# ============================================================================
# SWIN-TINY (ImageNet-pretrained) + DIHEDRAL (D4) — ONE self-contained Colab cell.
# SEQUEL Goal 1: isolates ARCHITECTURE-era value. Pretraining regime is held
# CONSTANT vs the champion's ImageNet-pretrained MobileNet member, so the ONLY
# changed variable is the architecture (transformer vs CNN).
#
# PyTorch/torchvision (NO numpy pin, NO legacy-keras, NO escnn). Resize 96->224 to
# use stock Swin-T weights. D4 train aug + D4-TTA (8-view avg) at inference, exactly
# like the champion MobileNet. val_loss checkpoint. seed-1337 WSI-grouped split
# (REUSED from src/data.py) so oof_swin.csv aligns with the 9 champion members.
# Ships sub_swin.csv + oof_swin.csv to Kaggle dataset histopath-swin-out.
#
# Runtime > Change runtime type > A100 (or any GPU + High-RAM). Keep the tab
# FOREGROUND (~30-60 min: 224px is heavier than 96px, + 8x TTA at the end).
# AUTH: reads a Colab secret named KAGGLE_KEY (accepts legacy key / kaggle.json / KGAT).
# ============================================================================
KAGGLE_USERNAME = "jackiemartindale"
OUT_DS  = "histopath-swin-out"
MEMBER  = "swin"
IMG     = 224                      # resize 96->224 for stock Swin-T
EPOCHS  = 8
BATCH   = 128                      # 224px is heavier; A100 fine
LR      = 2e-4
WD      = 0.05                     # Swin's typical AdamW weight decay
SMOOTH  = 0.05                     # label smoothing (matches champion recipe)
PATIENCE = 3                       # early-stop on val_loss

import os, sys, glob, csv, json, time, subprocess, shutil, zipfile
def sh(cmd):
    print("$", cmd, flush=True); return subprocess.run(cmd, shell=True, text=True).returncode

sh("pip -q install kaggle")
import numpy as np
import cv2
import torch, torch.nn as nn
import torchvision
from torchvision.models import swin_t, Swin_T_Weights
assert torch.cuda.is_available(), "NO GPU. Runtime > A100/GPU + High-RAM, then re-run."
DEV = torch.device("cuda")
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| GPU", torch.cuda.get_device_name(0), flush=True)

# --- Kaggle auth from the KAGGLE_KEY secret (legacy key / kaggle.json / KGAT) ---
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

# --- canonical seed-1337 WSI-grouped split, REUSED from src/data.py (OOF aligns
#     with every champion member). data.py imports TF; we only use its pandas split. ---
import yaml
from src import data as D
SEED = 1337
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT
cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
cfg["seed"] = SEED
df = D.load_labels(cfg)
tr_df, va_df = D.split_train_val(cfg, df)                 # GroupShuffleSplit(seed=1337), val_frac 0.1
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
ids_v = va_df["id"].tolist(); yv = va_df["label"].values.astype(np.int64)
from pathlib import Path
TEST_DIR = os.path.join(ROOT, "test"); TRAIN_DIR = os.path.join(ROOT, "train")
ids_t = sorted(p.stem for p in Path(TEST_DIR).glob("*.tif"))   # reproduces src/data.make_test_dataset order
print(f"train={len(tr_df)} val={len(ids_v)} test={len(ids_t)}", flush=True)

# --- torch Dataset: same cv2 decode as src/data._decode_tif, resize 96->224, ImageNet norm ---
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # HxWx3 uint8 RGB

def d4_np(img, k, flip):                                 # label-preserving dihedral transform
    if k: img = np.rot90(img, k)
    if flip: img = np.fliplr(img)
    return np.ascontiguousarray(img)

def to_chw(img):                                         # uint8 HWC RGB -> normalized CHW float32
    if img.shape[0] != IMG:
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)   # 96->224 upsample
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))

class PatchDS(torch.utils.data.Dataset):
    def __init__(self, ids, labels, img_dir, train):
        self.ids = ids; self.labels = labels; self.dir = img_dir; self.train = train
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = load_rgb(os.path.join(self.dir, self.ids[i] + ".tif"))
        if self.train:                                   # random D4 element (8-way)
            img = d4_np(img, np.random.randint(4), bool(np.random.randint(2)))
        x = torch.from_numpy(to_chw(img))
        y = torch.tensor(float(self.labels[i]) if self.labels is not None else 0.0)
        return x, y

def loader(ds, shuffle):
    return torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                                       num_workers=4, pin_memory=True, drop_last=False,
                                       persistent_workers=True)
tr_ld = loader(PatchDS(tr_df["id"].tolist(), tr_df["label"].values, TRAIN_DIR, True), True)
va_ld = loader(PatchDS(ids_v, yv, TRAIN_DIR, False), False)
te_ld = loader(PatchDS(ids_t, None, TEST_DIR, False), False)

# --- Swin-Tiny, ImageNet-1k pretrained; 1-logit head ---
torch.manual_seed(SEED); np.random.seed(SEED)
net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
net.head = nn.Linear(net.head.in_features, 1)
net = net.to(DEV)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler()
bce = nn.BCEWithLogitsLoss()                              # honest val signal (no smoothing)

def run_val():                                            # single-view val loss + logits
    net.eval(); tot = 0.0; n = 0; out = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in va_ld:
            x = x.to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
            lg = net(x).squeeze(1)
            tot += bce(lg, y).item() * len(y); n += len(y)
            out.append(lg.float().cpu())
    return tot / n, torch.cat(out).numpy()

CK = f"/content/{MEMBER}.pt"; best = float("inf"); bad = 0
for ep in range(EPOCHS):
    net.train(); t0 = time.time()
    for x, y in tr_ld:
        x = x.to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
        ys = y * (1 - SMOOTH) + 0.5 * SMOOTH             # label smoothing via soft target
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            loss = nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(1), ys)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    sched.step()
    vl, _ = run_val()
    print(f"epoch {ep+1}/{EPOCHS}  val_loss {vl:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    if vl < best - 1e-5:
        best = vl; bad = 0; torch.save(net.state_dict(), CK); print("  * checkpoint (val_loss min)", flush=True)
    else:
        bad += 1
        if bad >= PATIENCE: print("  early stop", flush=True); break
net.load_state_dict(torch.load(CK)); net.eval()

# --- D4-TTA (8-view average of sigmoid), same as the champion MobileNet.
#     Streams per-batch (test set is ~57k imgs; caching all 224px tensors would OOM). ---
D4 = [(k, f) for k in range(4) for f in (False, True)]   # identity, 3 rots, + h-flip of each = 8
def tta_predict(ld):
    net.eval(); outs = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, _ in ld:                                  # loader is shuffle=False -> order matches ids
            x = x.to(DEV, non_blocking=True)
            acc = torch.zeros(x.size(0), dtype=torch.float32, device=DEV)
            for k, f in D4:
                v = torch.rot90(x, k, dims=(2, 3)) if k else x
                if f: v = torch.flip(v, dims=[3])
                acc += torch.sigmoid(net(v).squeeze(1)).float()
            outs.append((acc / len(D4)).cpu())
    return torch.cat(outs).numpy()

def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

pv = tta_predict(va_ld); pt = tta_predict(te_ld)
assert len(pv) == len(yv) == len(ids_v) and len(pt) == len(ids_t), "length/order mismatch"
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/oof_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
with open(f"{OUT}/sub_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
print(f"\n*** {MEMBER}: val AUROC {auroc(yv, pv):.6f}  (pretrained Swin-T, D4-TTA) ***", flush=True)

# --- exfil via Kaggle dataset (NO Drive) ---
STAGE = "/content/swin_ds"; shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)
shutil.copy(f"{OUT}/sub_{MEMBER}.csv", STAGE); shutil.copy(f"{OUT}/oof_{MEMBER}.csv", STAGE)
json.dump({"title": OUT_DS, "id": f"{KAGGLE_USERNAME}/{OUT_DS}", "licenses": [{"name": "CC0-1.0"}]}, open(f"{STAGE}/dataset-metadata.json", "w"))
_r = subprocess.run("kaggle datasets create -p " + STAGE, shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
if any(s in _o.lower() for s in ("already exists", "already in use", "409")):
    _r = subprocess.run("kaggle datasets version -p " + STAGE + " -m swin", shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
print(_o, flush=True)
print(f">>> ALL DONE. Tell Cass to pull {KAGGLE_USERNAME}/{OUT_DS}", flush=True)
