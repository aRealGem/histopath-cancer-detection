# =============================================================================
# e2cnn_kaggle_kernel.py — Kaggle GPU kernel: the D4-steerable escnn (e2cnn) member
# =============================================================================
# A SECOND compute pool. Runs job3 (the e2cnn member) on a Kaggle GPU kernel IN
# PARALLEL with the Colab A100 p4m chain, instead of queueing behind the single
# A100. Kaggle's weekly GPU quota resets Saturdays; internet is enabled here so
# escnn pip-installs (PCam is a CSV-submission comp, not a code comp -> allowed).
#
# PRODUCER ONLY: writes three artifacts to /kaggle/working (the kernel's output):
#     oof_e2cnn.csv   (id,label,pred)   honest offline proxy on the fixed val split
#     sub_e2cnn.csv   (id,label=prob)   test submission
#     job_job3.json   (status=done...)  completion signal + val stats
# The Pi (autoloop/collect_e2cnn_kaggle.sh) pulls this output and merges it into
# the colab-out bus, where process.py decorr+LB-gates it like any other member --
# so this kernel needs NO Kaggle write-credentials of its own.
#
# ALIGNMENT: reproduces the SAME seed-1337 WSI-grouped val split via src.data and
# reads the SAME test tifs, so oof/sub rows align by id with every other member.
# D4-equivariant + GroupPooling => D4-INVARIANT => no TTA (a no-op, as for p4m).
# val_loss checkpointing (the project's overfit signal); AdamW wd=1e-4 (p4m_reg).
#
# FULLY OFFLINE: a competition-attached kernel has the network forced OFF by Kaggle
# (anti-leakage) regardless of enable_internet, so there is NO git/pip from the net.
# Code (src.data + WSI split map) is staged from the histopath-baseline-code dataset;
# escnn is pip-installed --no-index from the escnn-offline-wheels dataset.
# kernel-metadata.json: script, enable_gpu=true, enable_internet=false,
# competition_sources=["histopathologic-cancer-detection"],
# dataset_sources=["jackiemartindale/histopath-baseline-code",
#                  "jackiemartindale/escnn-offline-wheels"].
# =============================================================================
import os, sys, glob, json, time, shutil, gzip, zipfile, subprocess

WHEELS = "/kaggle/input/escnn-offline-wheels"
WORK = "/kaggle/working/repo"
OUT = "/kaggle/working"
SEED = 1337
JOBID = "job3"
MEMBER = "e2cnn"
EPOCHS = 20
BATCH = 128


def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check:
        assert r.returncode == 0, f"failed ({r.returncode}): {cmd}"
    return r.returncode


# --- code: stage src.data (+ WSI split map) from the histopath-baseline-code dataset
#     (offline; the comp-attached kernel has no network) — mirrors the proven pattern ---
if not os.path.exists(os.path.join(WORK, "src", "data.py")):
    os.makedirs(WORK, exist_ok=True)
    czip = glob.glob("/kaggle/input/**/code.zip", recursive=True)
    extracted = glob.glob("/kaggle/input/**/src/data.py", recursive=True)
    if czip:
        with zipfile.ZipFile(czip[0]) as z:
            z.extractall(WORK)
        print("staged code from", czip[0])
    elif extracted:
        rootc = os.path.dirname(os.path.dirname(extracted[0]))
        shutil.copytree(rootc, WORK, dirs_exist_ok=True)
        print("staged code from", rootc)
    else:
        raise SystemExit("no code (code.zip or src/data.py) under /kaggle/input — attach histopath-baseline-code")
os.chdir(WORK)
sys.path.insert(0, WORK)

# WSI split map (src.data needs data/wsi/patch_id_wsi_full.csv.gz)
if not os.path.exists("data/wsi/patch_id_wsi_full.csv.gz"):
    os.makedirs("data/wsi", exist_ok=True)
    gz = glob.glob("/kaggle/input/**/patch_id_wsi_full.csv.gz", recursive=True)
    csvh = [h for h in glob.glob("/kaggle/input/**/*wsi*full*.csv", recursive=True) if os.path.isfile(h)]
    if gz:
        shutil.copy(gz[0], "data/wsi/patch_id_wsi_full.csv.gz"); print("WSI map (gz) from", gz[0])
    elif csvh:
        with open(csvh[0], "rb") as fi, gzip.open("data/wsi/patch_id_wsi_full.csv.gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        print("WSI map from", csvh[0])
    else:
        raise SystemExit("no WSI split map under /kaggle/input")

# --- deps: escnn from the offline wheel bundle (torch/numpy/scipy preinstalled on Kaggle) ---
try:
    import escnn  # noqa: F401
except Exception:
    sh(f"{sys.executable} -m pip install --no-index --find-links {WHEELS} escnn")

import numpy as np
# escnn 0.1.9 still uses np.float/np.int (removed in numpy>=1.24) when building group
# representations -> restore the deprecated aliases before importing/using escnn.
for _a, _t in (("float", float), ("int", int), ("bool", bool), ("object", object),
               ("complex", complex), ("str", str)):
    if not hasattr(np, _a):
        setattr(np, _a, _t)
import yaml
import torch
import torch.nn as tnn
from torch.utils.data import Dataset, DataLoader
from escnn import gspaces, nn as enn
from PIL import Image


def auroc(y, s):
    """Tie-aware rank AUROC (no sklearn dependency)."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sa = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)) if npos and nneg else float("nan")


# --- config + competition data root ---
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cands = glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)
assert cands, "competition train_labels.csv not found (add the comp as a data source)"
root = os.path.dirname(cands[0])
cfg["data"]["root"] = root
cfg.setdefault("seed", SEED)
cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
train_dir = os.path.join(root, cfg["data"]["train_dir"])
test_dir = os.path.join(root, cfg["data"]["test_dir"])
ext = cfg["data"]["image_ext"]; size = int(cfg["data"]["image_size"])
print("data root:", root, "| ext:", ext, "| size:", size, flush=True)

# --- seed-1337 WSI-grouped split (identical to every other member) ---
from src import data as D
df = D.load_labels(cfg)
tr_df, va_df = D.split_train_val(cfg, df)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
val_ids = va_df["id"].tolist(); yv = va_df["label"].values.astype(np.float32)
test_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(test_dir, f"*{ext}")))
print(f"train n={len(tr_df)} val n={len(val_ids)} test n={len(test_ids)}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev != "cuda":
    print("WARNING: no CUDA visible -> this kernel needs a GPU accelerator.", flush=True)


class TifDS(Dataset):
    """Reads tifs from disk (memory-safe for the ~180k train / 57k test sets). Labels
    optional (None for the test set). Order preserved -> preds align to ids."""
    def __init__(self, ids, image_dir, labels=None):
        self.ids = list(ids)
        self.dir = image_dir
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir, f"{self.ids[i]}{ext}")).convert("RGB").resize((size, size))
        x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        y = -1.0 if self.labels is None else self.labels[i]
        return torch.from_numpy(x), y


def loader(ids, image_dir, labels=None, shuffle=False, drop_last=False):
    return DataLoader(TifDS(ids, image_dir, labels), batch_size=BATCH, shuffle=shuffle,
                      num_workers=2, pin_memory=(dev == "cuda"), drop_last=drop_last)


class E2Net(tnn.Module):
    """Compact D4 (flip+rot4) steerable net; pooled stride-2 per block to stay tractable
    at 96x96. regular_repr fields (|D4|=8); GroupPooling -> D4-invariant features."""
    def __init__(self):
        super().__init__()
        self.gs = gspaces.flipRot2dOnR2(N=4)
        self.itype = enn.FieldType(self.gs, 3 * [self.gs.trivial_repr])

        def block(cin, nf):
            cout = enn.FieldType(self.gs, nf * [self.gs.regular_repr])
            return enn.SequentialModule(
                enn.R2Conv(cin, cout, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(cout),
                enn.ReLU(cout, inplace=True),
                enn.PointwiseAvgPoolAntialiased(cout, sigma=0.66, stride=2),
            ), cout
        self.b1, c1 = block(self.itype, 8)      # 96 -> 48
        self.b2, c2 = block(c1, 16)             # 48 -> 24
        self.b3, c3 = block(c2, 32)             # 24 -> 12
        self.b4, c4 = block(c3, 48)             # 12 -> 6
        self.gpool = enn.GroupPooling(c4)
        cfeat = self.gpool.out_type.size
        self.head = tnn.Sequential(
            tnn.AdaptiveAvgPool2d(1), tnn.Flatten(),
            tnn.Linear(cfeat, 64), tnn.ReLU(inplace=True),
            tnn.Dropout(0.3), tnn.Linear(64, 1))

    def forward(self, x):
        x = enn.GeometricTensor(x, self.itype)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        return self.head(self.gpool(x).tensor).squeeze(1)


torch.manual_seed(SEED); np.random.seed(SEED)
net = E2Net().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = tnn.BCEWithLogitsLoss()
tr_loader = loader(tr_df["id"], train_dir, tr_df["label"], shuffle=True, drop_last=True)
va_loader = loader(val_ids, train_dir)          # val ids live under train/ (labelled)
ck = os.path.join(OUT, f"{MEMBER}.pt")
best_vl = float("inf"); best_state = None; patience = 8; bad = 0
t0 = time.time()


def _val_loss_probs():
    net.eval(); tot = 0.0; n = 0; probs = []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(dev); yb = yb.to(dev)
            lg = net(xb); tot += lossf(lg, yb).item() * len(xb); n += len(xb)
            probs.append(torch.sigmoid(lg).cpu().numpy())
    return tot / n, np.concatenate(probs)


for ep in range(EPOCHS):
    net.train()
    for xb, yb in tr_loader:
        xb = xb.to(dev); yb = yb.to(dev)
        opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    vl, pv = _val_loss_probs()
    print(f"  ep{ep + 1}/{EPOCHS} val_loss={vl:.4f} val_auroc={auroc(yv, pv):.4f}", flush=True)
    if vl < best_vl - 1e-5:
        best_vl = vl; bad = 0
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        torch.save(best_state, ck)
    else:
        bad += 1
        if bad >= patience:
            print("  early stop (val_loss)", flush=True); break

if best_state is not None:
    net.load_state_dict(best_state)


def _predict(ids, image_dir):
    net.eval(); out = []
    with torch.no_grad():
        for xb, _ in loader(ids, image_dir):
            out.append(torch.sigmoid(net(xb.to(dev))).cpu().numpy())
    return np.concatenate(out)


pv = _predict(val_ids, train_dir)               # D4-invariant -> no TTA
pt = _predict(test_ids, test_dir)

import csv
with open(os.path.join(OUT, f"oof_{MEMBER}.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(val_ids, yv, pv):
        w.writerow([i, int(l), f"{p:.6f}"])
with open(os.path.join(OUT, f"sub_{MEMBER}.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(test_ids, pt):
        w.writerow([i, f"{p:.6f}"])
manifest = {"jobid": JOBID, "status": "done", "priority": 3, "origin": "autonomous",
            "type": "train", "arch": "e2cnn_escnn", "members": [MEMBER],
            "val": {MEMBER: {"val_auroc": round(auroc(yv, pv), 6),
                             "val_loss": round(best_vl, 6), "monitor": "val_loss", "seed": SEED}},
            "train_seconds": round(time.time() - t0, 1), "compute_pool": "kaggle_gpu",
            "hyperparams": {"monitor": "val_loss", "epochs_cap": EPOCHS, "seed": SEED, "tta": False}}
json.dump(manifest, open(os.path.join(OUT, f"job_{JOBID}.json"), "w"))
print(f"DONE {MEMBER}: val AUROC {auroc(yv, pv):.6f} val_loss {best_vl:.4f} "
      f"in {time.time() - t0:.0f}s -> /kaggle/working (Pi collects).", flush=True)
