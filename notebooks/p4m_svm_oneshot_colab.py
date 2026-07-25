# =============================================================================
# p4m_svm — ONE self-contained Colab cell (A100/GPU). No bus, no poll loop.
# Cheap decorrelated member = logistic-regression on a trained p4m net's
# penultimate features. Uses p4m_dense.keras (already on your Drive) as the
# feature extractor -> NO retrain. Writes 2 CSVs to Drive + browser:
#   sub_p4m_svm.csv (id,label=prob) | oof_p4m_svm.csv (id,label,pred)
# Runtime > A100/GPU + High-RAM. Reads Colab secret KAGGLE_KEY (as before).
# =============================================================================
KAGGLE_USERNAME = "jackiemartindale"

import os, sys, glob, csv, subprocess, shutil, zipfile
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, text=True).returncode

# --- 1) legacy Keras-2 + keras-gcnn ---
sh("pip -q install tf-keras kaggle scikit-learn")
if not os.path.isdir("/content/tf2-GrouPy"):
    sh("git clone -q https://github.com/neel-dey/tf2-GrouPy.git /content/tf2-GrouPy")
if not os.path.isdir("/content/tf2-keras-gcnn"):
    sh("git clone -q https://github.com/neel-dey/tf2-keras-gcnn.git /content/tf2-keras-gcnn")
sys.path.insert(0, "/content/tf2-GrouPy"); sys.path.insert(0, "/content/tf2-keras-gcnn")
import numpy as np
for _n, _t in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _n): setattr(np, _n, _t)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import layers as _L
if not hasattr(_L.Layer, "_add_update_patched"):
    _o = _L.Layer.add_update
    _L.Layer.add_update = lambda self, updates, inputs=None: _o(self, updates)
    _L.Layer._add_update_patched = True
from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool
assert tf.config.list_physical_devices("GPU"), "NO GPU. Runtime > Change runtime type > A100/GPU + High-RAM."
print("TF", tf.__version__, "| GPU OK", flush=True)

# --- 2) Kaggle auth from the KAGGLE_KEY Colab secret ---
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

# --- 3) repo + competition data (reused if present) ---
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

# --- 4) data pipeline (seed-1337 split; non-augmented passes for feature extraction) ---
import yaml
from src import data as D
SEED = 1337
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT; cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"; cfg["seed"] = SEED
df = D.load_labels(cfg)
tr_df, va_df = D.split_train_val(cfg, df); tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
ids_v = va_df["id"].tolist(); yv = va_df["label"].values.astype(int); ytr = tr_df["label"].values.astype(int)
# build(training=False) => NO augmentation, NO shuffle => features are clean + id-aligned
_, tr_noaug = D.make_train_val_datasets(cfg, df.iloc[:1], tr_df)   # non-aug pass over TRAIN
_, val_ds   = D.make_train_val_datasets(cfg, df.iloc[:1], va_df)
test_ds, ids_t = D.make_test_dataset(cfg)
print(f"train={len(ytr)} val={len(yv)} test={len(ids_t)}", flush=True)

# --- 5) feature extractor = p4m_dense.keras (from Drive), penultimate layer ---
from google.colab import drive; drive.mount("/content/drive")
dh = glob.glob("/content/drive/MyDrive/**/p4m_dense.keras", recursive=True)
assert dh, "p4m_dense.keras not found in Drive (MyDrive). Put it under MyDrive/ and re-run."
ckpt = dh[0]; print("feature base:", ckpt, flush=True)
net = tf.keras.models.load_model(ckpt, compile=False,
        custom_objects={"GConv2D": GConv2D, "GBatchNorm": GBatchNorm, "GroupPool": GroupPool})
feat = Model(net.input, net.layers[-2].output)             # pre-final-Dense features
print("feature dim:", int(net.layers[-2].output.shape[-1]), flush=True)

def feats(ds):
    out = []
    for xb, _ in ds:
        f = feat.predict(xb, verbose=0); out.append(f.reshape(f.shape[0], -1))
    return np.concatenate(out)
print("extracting features (train is the long pole)…", flush=True)
Xtr = feats(tr_noaug); Xva = feats(val_ds); Xte = feats(test_ds)
print("features:", Xtr.shape, Xva.shape, Xte.shape, flush=True)

# --- 6) logistic regression head -> predictions ---
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
pv = clf.predict_proba(sc.transform(Xva))[:, 1]
pt = clf.predict_proba(sc.transform(Xte))[:, 1]

def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

MEMBER = "p4m_svm"
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/oof_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
with open(f"{OUT}/sub_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
print(f"\n*** {MEMBER}: val AUROC {auroc(yv, pv):.6f}  (logreg on p4m_dense features) ***", flush=True)

# --- 7) hand back via Drive (works now) + browser ---
dd = "/content/drive/MyDrive/histopath_out"; os.makedirs(dd, exist_ok=True)
shutil.copy(f"{OUT}/oof_{MEMBER}.csv", dd); shutil.copy(f"{OUT}/sub_{MEMBER}.csv", dd)
print("copied both CSVs to Drive:", dd, flush=True)
try:
    from google.colab import files
    files.download(f"{OUT}/sub_{MEMBER}.csv"); files.download(f"{OUT}/oof_{MEMBER}.csv")
except Exception as e:
    print("browser download skipped — they're in Drive/histopath_out:", e, flush=True)
