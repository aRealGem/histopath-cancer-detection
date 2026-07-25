# ============================================================================
# p4m_svm (self-contained, NO Drive) — ONE Colab cell. Trains its OWN p4m feature
# base (seed 13), then ships TWO members to a Kaggle dataset (histopath-svm-out):
#   p4m_s13   = the trained p4m net itself (a bonus 4th p4m seed for the average)
#   p4m_svm   = logistic-regression on that net's penultimate features (the SVM member)
# Reads secret KAGGLE_KEY. Runtime > A100/GPU + High-RAM. Keep the tab FOREGROUND.
# ============================================================================
KAGGLE_USERNAME = "jackiemartindale"
OUT_DS = "histopath-svm-out"

import os, sys, glob, csv, json, time, subprocess, shutil, zipfile
os.environ["TF_USE_LEGACY_KERAS"] = "1"
def sh(cmd):
    print("$", cmd, flush=True); return subprocess.run(cmd, shell=True, text=True).returncode

# --- 1) legacy Keras-2 + keras-gcnn ---
sh("pip -q install tf-keras kaggle scikit-learn")
if not os.path.isdir("/content/tf2-GrouPy"): sh("git clone -q https://github.com/neel-dey/tf2-GrouPy.git /content/tf2-GrouPy")
if not os.path.isdir("/content/tf2-keras-gcnn"): sh("git clone -q https://github.com/neel-dey/tf2-keras-gcnn.git /content/tf2-keras-gcnn")
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
assert tf.config.list_physical_devices("GPU"), "NO GPU. Runtime > A100/GPU + High-RAM."
print("TF", tf.__version__, "| GPU OK", flush=True)

# --- 2) Kaggle auth from KAGGLE_KEY secret ---
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

# --- 3) repo + comp data ---
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

# --- 4) data pipeline (seed-1337 split; aug train for fit, non-aug for features/preds) ---
import yaml
from src import data as D
SEED = 1337
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT; cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"; cfg["seed"] = SEED
df = D.load_labels(cfg); tr_df, va_df = D.split_train_val(cfg, df)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
ids_v = va_df["id"].tolist(); yv = va_df["label"].values.astype(int); ytr = tr_df["label"].values.astype(int)
train_ds, val_ds = D.make_train_val_datasets(cfg, tr_df, va_df)          # aug train (for fit) + non-aug val
_, tr_noaug = D.make_train_val_datasets(cfg, df.iloc[:1], tr_df)         # non-aug train (for features)
test_ds, ids_t = D.make_test_dataset(cfg)
print(f"train={len(ytr)} val={len(yv)} test={len(ids_t)}", flush=True)

# --- 5) train a fresh p4m (seed 13) = the feature base AND a bonus member p4m_s13 ---
def gblock(x, w, hin, hout):
    x = GConv2D(w, 3, h_input=hin, h_output=hout, padding="same", use_bias=False)(x)
    x = GBatchNorm(h=hout)(x); return layers.Activation("relu")(x)
def p4m_tinyvgg(widths=(8, 16, 32)):
    inp = Input((96, 96, 3), dtype="float32")
    x = layers.Rescaling(1/255.)(inp); x = layers.RandomContrast(0.1)(x)
    x = gblock(x, widths[0], "Z2", "D4")
    x = gblock(x, widths[0], "D4", "D4"); x = layers.MaxPool2D(2)(x)
    x = gblock(x, widths[1], "D4", "D4"); x = gblock(x, widths[1], "D4", "D4"); x = layers.MaxPool2D(2)(x)
    x = gblock(x, widths[2], "D4", "D4"); x = gblock(x, widths[2], "D4", "D4"); x = layers.MaxPool2D(2)(x)
    x = gblock(x, widths[2], "D4", "D4")
    x = GroupPool(h_input="D4")(x); x = layers.GlobalAveragePooling2D()(x)
    return Model(inp, layers.Dense(1, activation="sigmoid", dtype="float32")(x), name="p4m_tinyvgg")

tf.keras.utils.set_random_seed(13)
try:   opt = keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4, clipnorm=1.0)
except Exception: opt = keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0)
net = p4m_tinyvgg()
net.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05), metrics=[keras.metrics.AUC(name="auc")])
ck = "/content/p4m_s13.keras"
net.fit(train_ds, validation_data=val_ds, epochs=6, verbose=2, callbacks=[
    keras.callbacks.ModelCheckpoint(ck, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=8, restore_best_weights=True)])
net = tf.keras.models.load_model(ck, compile=False, custom_objects={"GConv2D": GConv2D, "GBatchNorm": GBatchNorm, "GroupPool": GroupPool})

def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

# member 1: p4m_s13 = the net's own predictions
pv13 = net.predict(val_ds, verbose=0).ravel(); pt13 = net.predict(test_ds, verbose=0).ravel()
print(f"*** p4m_s13: val AUROC {auroc(yv, pv13):.6f} ***", flush=True)

# member 2: p4m_svm = logreg on penultimate (pre-final-Dense) features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
feat = Model(net.input, net.layers[-2].output)
def feats(ds):
    out = []
    for xb, _ in ds:
        f = feat.predict(xb, verbose=0); out.append(f.reshape(f.shape[0], -1))
    return np.concatenate(out)
print("extracting features (train is the long pole)…", flush=True)
Xtr = feats(tr_noaug); Xva = feats(val_ds); Xte = feats(test_ds)
sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
pvs = clf.predict_proba(sc.transform(Xva))[:, 1]; pts = clf.predict_proba(sc.transform(Xte))[:, 1]
print(f"*** p4m_svm: val AUROC {auroc(yv, pvs):.6f} (logreg on p4m_s13 features) ***", flush=True)

# --- 6) write both members' CSVs ---
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
def emit(member, pv, pt):
    with open(f"{OUT}/oof_{member}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label", "pred"])
        for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
    with open(f"{OUT}/sub_{member}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
emit("p4m_s13", pv13, pt13); emit("p4m_svm", pvs, pts)

# --- 7) exfil via Kaggle dataset (NO Drive) ---
STAGE = "/content/svm_ds"; shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)
for p in glob.glob(OUT + "/*.csv"): shutil.copy(p, STAGE)
json.dump({"title": OUT_DS, "id": f"{KAGGLE_USERNAME}/{OUT_DS}", "licenses": [{"name": "CC0-1.0"}]}, open(f"{STAGE}/dataset-metadata.json", "w"))
_r = subprocess.run("kaggle datasets create -p " + STAGE, shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
if any(s in _o.lower() for s in ("already exists", "already in use", "409")):
    _r = subprocess.run("kaggle datasets version -p " + STAGE + " -m svm", shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
print(_o, flush=True)
print(f">>> ALL DONE. Tell Cass to pull {KAGGLE_USERNAME}/{OUT_DS}", flush=True)
