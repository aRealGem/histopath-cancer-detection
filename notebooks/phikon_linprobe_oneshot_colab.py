# ============================================================================
# PHIKON-v2 (Owkin pathology foundation model) LINEAR PROBE — ONE Colab cell.
# SEQUEL Goal 2: isolates DATA/PRETRAINING value. Freeze a ViT-L pretrained on
# ~460M histology tiles, extract patch embeddings, fit a linear probe. The FM is
# NOT fine-tuned -> this measures the pretrained FEATURES (the data axis), holding
# architecture-training out (that was Goal 1 / Swin).
#
# PyTorch + HuggingFace transformers. Resize 96->224, ImageNet norm. D4-TTA (8-view
# embedding average) on val+test, like the champion. val_loss checkpoint on the
# probe. seed-1337 WSI-grouped split (REUSED from src/data.py) so oof_phikon.csv
# aligns with the (now 10) champion members. Ships to Kaggle dataset histopath-phikon-out.
#
# Runtime > A100 (or any GPU + High-RAM). Keep the tab FOREGROUND (~25-40 min:
# ViT-L embedding extraction dominates; the probe itself trains in seconds).
# AUTH: needs the KAGGLE_KEY secret (as before). owkin/phikon-v2 is a PUBLIC,
# non-gated model -> NO license to accept and NO HF token required. (If you ever
# swap in a gated FM, add an HF_TOKEN secret and it will be used automatically.)
# ============================================================================
KAGGLE_USERNAME = "jackiemartindale"
OUT_DS  = "histopath-phikon-out"
MEMBER  = "phikon"
MODEL_ID = "owkin/phikon-v2"
IMG     = 224
BATCH   = 128
PROBE_EPOCHS = 300          # linear probe on cached embeddings -> trains in seconds
PROBE_LR = 1e-3
PROBE_WD = 1e-4
SMOOTH  = 0.05
SEED    = 1337

import os, sys, glob, csv, json, time, subprocess, shutil, zipfile
def sh(cmd):
    print("$", cmd, flush=True); return subprocess.run(cmd, shell=True, text=True).returncode

sh("pip -q install -U transformers huggingface_hub kaggle")
import numpy as np
import cv2
import torch, torch.nn as nn
from transformers import AutoModel
assert torch.cuda.is_available(), "NO GPU. Runtime > A100/GPU + High-RAM, then re-run."
DEV = torch.device("cuda")
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), flush=True)

# --- HuggingFace auth: OPTIONAL. owkin/phikon-v2 is public (non-gated) -> no token needed.
#     A token is only used if you happen to have one set (e.g. to swap in a gated FM). ---
hf = ""
try:
    from google.colab import userdata; hf = (userdata.get("HF_TOKEN") or "").strip()
except Exception: hf = os.environ.get("HF_TOKEN", "").strip()
if hf:
    from huggingface_hub import login as hf_login
    hf_login(token=hf); print("HF auth ok (token found)", flush=True)
else:
    print("no HF_TOKEN set -> anonymous download (owkin/phikon-v2 is public, non-gated)", flush=True)

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

# --- canonical seed-1337 WSI-grouped split, REUSED from src/data.py (OOF aligns) ---
import yaml
from src import data as D
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT
cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
cfg["seed"] = SEED
df = D.load_labels(cfg)
tr_df, va_df = D.split_train_val(cfg, df)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
ids_v = va_df["id"].tolist(); yv = va_df["label"].values.astype(np.int64)
yt_tr = tr_df["label"].values.astype(np.float32)
from pathlib import Path
TRAIN_DIR = os.path.join(ROOT, "train"); TEST_DIR = os.path.join(ROOT, "test")
ids_t = sorted(p.stem for p in Path(TEST_DIR).glob("*.tif"))
print(f"train={len(tr_df)} val={len(ids_v)} test={len(ids_t)}", flush=True)

# --- dataset: same cv2 decode as src/data._decode_tif, resize 96->224, ImageNet norm ---
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
def to_chw(img):
    if img.shape[0] != IMG:
        img = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))
class PatchDS(torch.utils.data.Dataset):
    def __init__(self, ids, img_dir): self.ids = ids; self.dir = img_dir
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.dir, self.ids[i] + ".tif"), cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(self.ids[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(to_chw(img))
def loader(ids, img_dir):
    return torch.utils.data.DataLoader(PatchDS(ids, img_dir), batch_size=BATCH, shuffle=False,
                                       num_workers=4, pin_memory=True, persistent_workers=True)

# --- frozen Phikon-v2 (ViT-L); CLS-token embedding ---
fm = AutoModel.from_pretrained(MODEL_ID).to(DEV).eval()
for p in fm.parameters(): p.requires_grad_(False)
DIM = fm.config.hidden_size
print(f"loaded {MODEL_ID}  embed dim={DIM}", flush=True)

D4 = [(k, f) for k in range(4) for f in (False, True)]      # 8 dihedral views
def extract(ids, img_dir, tta):
    ld = loader(ids, img_dir); outs = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x in ld:
            x = x.to(DEV, non_blocking=True)
            if tta:
                acc = torch.zeros(x.size(0), DIM, dtype=torch.float32, device=DEV)
                for k, f in D4:
                    v = torch.rot90(x, k, dims=(2, 3)) if k else x
                    if f: v = torch.flip(v, dims=[3])
                    acc += fm(pixel_values=v).last_hidden_state[:, 0, :].float()
                emb = acc / len(D4)
            else:
                emb = fm(pixel_values=x).last_hidden_state[:, 0, :].float()
            outs.append(emb.cpu())
    return torch.cat(outs).numpy()

t0 = time.time(); Etr = extract(tr_df["id"].tolist(), TRAIN_DIR, tta=False)
print(f"train emb {Etr.shape} ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time(); Ev = extract(ids_v, TRAIN_DIR, tta=True)
print(f"val emb {Ev.shape} ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time(); Et = extract(ids_t, TEST_DIR, tta=True)
print(f"test emb {Et.shape} ({time.time()-t0:.0f}s)", flush=True)

# --- standardize features (fit on train) -> linear probe ---
mu = Etr.mean(0, keepdims=True); sd = Etr.std(0, keepdims=True) + 1e-6
def z(E): return (E - mu) / sd
Xtr = torch.tensor(z(Etr), device=DEV); Ytr = torch.tensor(yt_tr, device=DEV)
Xv  = torch.tensor(z(Ev),  device=DEV); Yv = torch.tensor(yv.astype(np.float32), device=DEV)
Xt  = torch.tensor(z(Et),  device=DEV)

torch.manual_seed(SEED)
head = nn.Linear(DIM, 1).to(DEV)
opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
bce = nn.BCEWithLogitsLoss()
best = float("inf"); best_state = None; bad = 0
for ep in range(PROBE_EPOCHS):
    head.train(); opt.zero_grad(set_to_none=True)
    ys = Ytr * (1 - SMOOTH) + 0.5 * SMOOTH
    loss = nn.functional.binary_cross_entropy_with_logits(head(Xtr).squeeze(1), ys)
    loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        vl = bce(head(Xv).squeeze(1), Yv).item()
    if vl < best - 1e-6:
        best = vl; best_state = {k: v.clone() for k, v in head.state_dict().items()}; bad = 0
    else:
        bad += 1
        if bad >= 30: break
head.load_state_dict(best_state); head.eval()
print(f"probe done: best val_loss {best:.5f} at ~{ep} epochs", flush=True)

def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

with torch.no_grad():
    pv = torch.sigmoid(head(Xv).squeeze(1)).cpu().numpy()
    pt = torch.sigmoid(head(Xt).squeeze(1)).cpu().numpy()
assert len(pv) == len(yv) == len(ids_v) and len(pt) == len(ids_t), "length/order mismatch"
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/oof_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
with open(f"{OUT}/sub_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
print(f"\n*** {MEMBER}: val AUROC {auroc(yv, pv):.6f}  (Phikon-v2 linear probe, D4-TTA) ***", flush=True)

# --- exfil via Kaggle dataset (NO Drive) ---
STAGE = "/content/phikon_ds"; shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)
shutil.copy(f"{OUT}/sub_{MEMBER}.csv", STAGE); shutil.copy(f"{OUT}/oof_{MEMBER}.csv", STAGE)
json.dump({"title": OUT_DS, "id": f"{KAGGLE_USERNAME}/{OUT_DS}", "licenses": [{"name": "CC0-1.0"}]}, open(f"{STAGE}/dataset-metadata.json", "w"))
_r = subprocess.run("kaggle datasets create -p " + STAGE, shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
if any(s in _o.lower() for s in ("already exists", "already in use", "409")):
    _r = subprocess.run("kaggle datasets version -p " + STAGE + " -m phikon", shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
print(_o, flush=True)
print(f">>> ALL DONE. Tell Cass to pull {KAGGLE_USERNAME}/{OUT_DS}", flush=True)
