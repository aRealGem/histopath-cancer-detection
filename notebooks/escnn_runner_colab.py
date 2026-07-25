# ============================================================================
# histopath escnn RUNNER — paste ONCE, walk away. Trains a JOB LIST of D4-steerable
# escnn members and ships each to a Kaggle dataset as it finishes. Resumable: if the
# runtime dies mid-batch, just RUN THIS CELL AGAIN — it skips members already shipped.
# Reads Colab secret KAGGLE_KEY. Keep the tab FOREGROUND.
#
# NOTE: it restarts the runtime ONCE at the start (to load numpy<2 that escnn needs).
#       When it says "RESTARTING", wait ~15s for reconnect and RUN THE CELL AGAIN.
# ============================================================================
KAGGLE_USERNAME = "jackiemartindale"
OUT_DS = "histopath-e2cnn-out"
JOBS = [("e2cnn_s99", 99), ("e2cnn_s21", 21)]   # new seeds to average with the shipped e2cnn (already on the dataset)
EPOCHS, BATCH = 6, 128                          # if CUDA OOM, lower BATCH to 64

import os, sys, subprocess
def sh(c):
    print("$", c, flush=True); return subprocess.run(c, shell=True, text=True).returncode

# --- numpy<2 gate (escnn's C-deps need it); restart ONCE so it loads ---
try:
    import numpy as _npck; _np_ok = _npck.__version__.split(".")[0] == "1"
except Exception:
    _np_ok = False
if not _np_ok:
    sh("pip -q install 'numpy<2' escnn kaggle")
    print("\n*** numpy pinned <2 — RESTARTING RUNTIME. Wait for reconnect, then RUN THIS CELL AGAIN. ***", flush=True)
    os.kill(os.getpid(), 9)
sh("pip -q install escnn kaggle")               # idempotent: ensure escnn present on the 2nd run

import glob, csv, json, time, shutil, zipfile
import numpy as np
for _n, _t in [("int", int), ("float", float), ("bool", bool), ("object", object), ("str", str)]:
    if not hasattr(np, _n): setattr(np, _n, _t)
import torch, torch.nn as tnn
from torch.utils.data import Dataset, DataLoader
from escnn import gspaces, nn as enn
from PIL import Image
assert torch.cuda.is_available(), "NO GPU. Runtime > A100/GPU + High-RAM, then re-run."
dev = "cuda"; print("torch", torch.__version__, "| numpy", np.__version__, "| GPU", torch.cuda.get_device_name(0), flush=True)

# --- Kaggle auth from the KAGGLE_KEY secret ---
import json as _json
kdir = os.path.expanduser("~/.kaggle"); os.makedirs(kdir, exist_ok=True); os.environ["KAGGLE_CONFIG_DIR"] = kdir
key = ""
try:
    from google.colab import userdata; key = (userdata.get("KAGGLE_KEY") or "").strip()
except Exception: key = os.environ.get("KAGGLE_KEY", "").strip()
try:
    from google.colab import userdata as _ud
    _u = _ud.get("KAGGLE_USERNAME")
    if _u: KAGGLE_USERNAME = _u.strip()
except Exception: pass
assert key, "Add a Colab secret named KAGGLE_KEY and enable Notebook access."
for _f in ("access_token", "kaggle.json"):
    try: os.remove(os.path.join(kdir, _f))
    except OSError: pass
if key.startswith("{"):
    open(os.path.join(kdir, "kaggle.json"), "w").write(key); _j = _json.loads(key)
    os.environ["KAGGLE_USERNAME"] = _j.get("username", KAGGLE_USERNAME); os.environ["KAGGLE_KEY"] = _j.get("key", "")
elif key.startswith("KGAT"):
    open(os.path.join(kdir, "access_token"), "w").write(key)
else:
    open(os.path.join(kdir, "kaggle.json"), "w").write(_json.dumps({"username": KAGGLE_USERNAME, "key": key}))
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME; os.environ["KAGGLE_KEY"] = key
print("Kaggle auth written", flush=True)

# --- repo + competition data (reused if present) ---
WORK = "/content/repo"
if not os.path.exists(WORK + "/src/data.py"):
    sh(f"git clone --depth 1 https://github.com/aRealGem/histopath-cancer-detection {WORK}")
sys.path.insert(0, WORK); os.chdir(WORK)
hits = glob.glob("/content/**/train_labels.csv", recursive=True)
if not hits:
    os.makedirs("/content/pcam", exist_ok=True)
    sh("kaggle competitions download -c histopathologic-cancer-detection -p /content/pcam")
    for z in glob.glob("/content/pcam/*.zip"): zipfile.ZipFile(z).extractall("/content/pcam")
    hits = glob.glob("/content/**/train_labels.csv", recursive=True)
ROOT = os.path.dirname(hits[0]); print("data root:", ROOT, flush=True)

# --- canonical seed-1337 split (OOF aligns with every member) ---
import yaml
from src import data as D
SPLIT_SEED = 1337
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT; cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"; cfg["seed"] = SPLIT_SEED
train_dir = os.path.join(ROOT, cfg["data"]["train_dir"]); test_dir = os.path.join(ROOT, cfg["data"]["test_dir"])
ext = cfg["data"]["image_ext"]; size = int(cfg["data"]["image_size"])
df = D.load_labels(cfg); tr_df, va_df = D.split_train_val(cfg, df)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
val_ids = va_df["id"].tolist(); yv = va_df["label"].values.astype(np.float32)
test_ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(test_dir, f"*{ext}")))
print(f"train n={len(tr_df)} val n={len(val_ids)} test n={len(test_ids)}", flush=True)

class TifDS(Dataset):
    def __init__(self, ids, image_dir, labels=None):
        self.ids = list(ids); self.dir = image_dir
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir, f"{self.ids[i]}{ext}")).convert("RGB").resize((size, size))
        x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(x), (-1.0 if self.labels is None else self.labels[i])

def loader(ids, image_dir, labels=None, shuffle=False, drop_last=False):
    return DataLoader(TifDS(ids, image_dir, labels), batch_size=BATCH, shuffle=shuffle,
                      num_workers=2, pin_memory=True, drop_last=drop_last)

class E2Net(tnn.Module):
    def __init__(self):
        super().__init__()
        self.gs = gspaces.flipRot2dOnR2(N=4)
        self.itype = enn.FieldType(self.gs, 3 * [self.gs.trivial_repr])
        def block(cin, nf):
            cout = enn.FieldType(self.gs, nf * [self.gs.regular_repr])
            return enn.SequentialModule(
                enn.R2Conv(cin, cout, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(cout), enn.ReLU(cout, inplace=True),
                enn.PointwiseAvgPoolAntialiased(cout, sigma=0.66, stride=2)), cout
        self.b1, c1 = block(self.itype, 8); self.b2, c2 = block(c1, 16)
        self.b3, c3 = block(c2, 32); self.b4, c4 = block(c3, 48)
        self.gpool = enn.GroupPooling(c4); cfeat = self.gpool.out_type.size
        self.head = tnn.Sequential(tnn.AdaptiveAvgPool2d(1), tnn.Flatten(),
                                   tnn.Linear(cfeat, 64), tnn.ReLU(inplace=True), tnn.Dropout(0.3), tnn.Linear(64, 1))
    def forward(self, x):
        x = enn.GeometricTensor(x, self.itype)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        return self.head(self.gpool(x).tensor).squeeze(1)

def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)

def already_shipped():
    st = "/content/_chk"; shutil.rmtree(st, ignore_errors=True); os.makedirs(st)
    subprocess.run(f"kaggle datasets download -d {KAGGLE_USERNAME}/{OUT_DS} -p {st} --unzip",
                   shell=True, capture_output=True)
    return set(os.path.basename(p)[4:-4] for p in glob.glob(st + "/sub_*.csv"))

def ship():   # CUMULATIVE: merge new files into the existing dataset so we never wipe prior members
    st = "/content/_shipds"; shutil.rmtree(st, ignore_errors=True); os.makedirs(st)
    subprocess.run(f"kaggle datasets download -d {KAGGLE_USERNAME}/{OUT_DS} -p {st} --unzip",
                   shell=True, capture_output=True)
    for p in glob.glob(OUT + "/*.csv"): shutil.copy(p, st)
    json.dump({"title": OUT_DS, "id": f"{KAGGLE_USERNAME}/{OUT_DS}", "licenses": [{"name": "CC0-1.0"}]},
              open(f"{st}/dataset-metadata.json", "w"))
    r = subprocess.run("kaggle datasets create -p " + st, shell=True, text=True, capture_output=True)
    o = r.stdout + r.stderr
    if any(s in o.lower() for s in ("already exists", "already in use", "409")):   # Kaggle says "title already in use"
        r = subprocess.run("kaggle datasets version -p " + st + " -m runner", shell=True, text=True, capture_output=True)
        o = r.stdout + r.stderr
    print(o, flush=True)

def train_one(member, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = E2Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = tnn.BCEWithLogitsLoss()
    tr_loader = loader(tr_df["id"], train_dir, tr_df["label"], shuffle=True, drop_last=True)
    va_loader = loader(val_ids, train_dir, yv)      # real labels -> honest val_loss
    best_vl = float("inf"); best_state = None; patience = 3; bad = 0; t0 = time.time()
    def val_pass():
        net.eval(); tot = 0.0; n = 0; probs = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(dev); yb = yb.to(dev); lg = net(xb)
                tot += lossf(lg, yb).item() * len(xb); n += len(xb); probs.append(torch.sigmoid(lg).cpu().numpy())
        return tot / n, np.concatenate(probs)
    for ep in range(EPOCHS):
        net.train()
        for xb, yb in tr_loader:
            xb = xb.to(dev); yb = yb.to(dev); opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        vl, pv = val_pass()
        print(f"  [{member}] ep{ep+1}/{EPOCHS} val_loss={vl:.4f} val_auroc={auroc(yv, pv):.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best_vl - 1e-5:
            best_vl = vl; bad = 0; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: print("  early stop", flush=True); break
    if best_state is not None: net.load_state_dict(best_state)
    def predict(ids, image_dir):
        net.eval(); out = []
        with torch.no_grad():
            for xb, _ in loader(ids, image_dir): out.append(torch.sigmoid(net(xb.to(dev))).cpu().numpy())
        return np.concatenate(out)
    pv = predict(val_ids, train_dir); pt = predict(test_ids, test_dir)
    with open(f"{OUT}/oof_{member}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label", "pred"])
        for i, l, p in zip(val_ids, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
    with open(f"{OUT}/sub_{member}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for i, p in zip(test_ids, pt): w.writerow([i, f"{p:.6f}"])
    print(f"*** {member}: val AUROC {auroc(yv, pv):.6f} val_loss {best_vl:.4f} ***", flush=True)

# --- the loop: skip already-shipped, train each, ship after EACH (crash-safe) ---
done = already_shipped(); print("already shipped:", done or "{}", flush=True)
for member, seed in JOBS:
    if member in done:
        print(f"skip (already shipped): {member}", flush=True); continue
    train_one(member, seed)
    ship()
    print(f">>> shipped {member} to {KAGGLE_USERNAME}/{OUT_DS}", flush=True)
print(f"\n>>> ALL DONE. Tell Cass to pull {KAGGLE_USERNAME}/{OUT_DS}", flush=True)
