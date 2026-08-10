# =============================================================================
# p4m_seed7 — ONE self-contained Colab cell (A100/GPU).
# No message bus, no poll loop, no second cell. Trains a 3rd seed of the winning
# p4m_reg recipe end-to-end and writes TWO CSVs you send back to the Pi:
#     sub_p4m_seed7.csv  (id,label=prob)  test predictions
#     oof_p4m_seed7.csv  (id,label,pred)  honest val predictions (seed-1337 split)
# Paste the whole thing into a fresh Colab cell on a GPU runtime and run.
# Runtime > Change runtime type > A100 (or any GPU + High-RAM). ~7GB data
# download + ~10-25 min train. Keep the tab FOREGROUND (backgrounding it is what
# idle-kills Colab).
# =============================================================================
# AUTH: this reads a Colab secret named KAGGLE_KEY (left sidebar > key icon > + Add
# secret, then toggle "Notebook access" ON). It accepts a legacy 32-char API key, a
# full kaggle.json blob, OR a new KGAT_ access token. Legacy keys also need a username:
# it uses a KAGGLE_USERNAME secret if you add one, else the default below.
KAGGLE_USERNAME = "jackiemartindale"

import os, sys, glob, csv, subprocess, shutil, zipfile
os.environ["TF_USE_LEGACY_KERAS"] = "1"          # MUST precede the first tensorflow import

def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, text=True).returncode

# --- 1) legacy Keras-2 + keras-gcnn (basveeling's PCam-paper lib, neel-dey TF2 port) ---
sh("pip -q install tf-keras kaggle")
if not os.path.isdir("/content/tf2-GrouPy"):
    sh("git clone -q https://github.com/neel-dey/tf2-GrouPy.git /content/tf2-GrouPy")
if not os.path.isdir("/content/tf2-keras-gcnn"):
    sh("git clone -q https://github.com/neel-dey/tf2-keras-gcnn.git /content/tf2-keras-gcnn")
sys.path.insert(0, "/content/tf2-GrouPy"); sys.path.insert(0, "/content/tf2-keras-gcnn")

import numpy as np
for _n, _t in [("int", int), ("float", float), ("bool", bool)]:   # aliases the 2018 lib still uses
    if not hasattr(np, _n): setattr(np, _n, _t)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras import layers as _L
if not hasattr(_L.Layer, "_add_update_patched"):                  # 2018 lib passes a removed 'inputs' arg
    _o = _L.Layer.add_update
    _L.Layer.add_update = lambda self, updates, inputs=None: _o(self, updates)
    _L.Layer._add_update_patched = True
from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool

assert tf.config.list_physical_devices("GPU"), \
    "NO GPU. Runtime > Change runtime type > A100/GPU + High-RAM, then re-run."
print("TF", tf.__version__, "| GPU OK", flush=True)

# --- 2) Kaggle auth from the KAGGLE_KEY Colab secret (legacy key, kaggle.json, or KGAT) ---
import json as _json
kdir = os.path.expanduser("~/.kaggle"); os.makedirs(kdir, exist_ok=True)
os.environ["KAGGLE_CONFIG_DIR"] = kdir
key = ""
try:
    from google.colab import userdata
    key = (userdata.get("KAGGLE_KEY") or "").strip()
except Exception:
    key = os.environ.get("KAGGLE_KEY", "").strip()
try:                                                  # optional username secret (legacy keys only)
    from google.colab import userdata as _ud
    _u = _ud.get("KAGGLE_USERNAME")
    if _u: KAGGLE_USERNAME = _u.strip()
except Exception: pass
assert key, "Add a Colab secret named KAGGLE_KEY (sidebar > key icon) and enable Notebook access."
for _f in ("access_token", "kaggle.json"):            # clear stale creds -> CLI uses exactly what we write
    try: os.remove(os.path.join(kdir, _f))
    except OSError: pass
if key.startswith("{"):                               # a full kaggle.json was pasted into the secret
    open(os.path.join(kdir, "kaggle.json"), "w").write(key)
    _j = _json.loads(key)
    os.environ["KAGGLE_USERNAME"] = _j.get("username", KAGGLE_USERNAME); os.environ["KAGGLE_KEY"] = _j.get("key", "")
    _how = f"kaggle.json (user {os.environ['KAGGLE_USERNAME']})"
elif key.startswith("KGAT"):                          # new access-token format
    open(os.path.join(kdir, "access_token"), "w").write(key); _how = "access_token (KGAT)"
else:                                                 # legacy username + 32-char key
    open(os.path.join(kdir, "kaggle.json"), "w").write(_json.dumps({"username": KAGGLE_USERNAME, "key": key}))
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME; os.environ["KAGGLE_KEY"] = key
    _how = f"kaggle.json (user {KAGGLE_USERNAME})"
try: os.chmod(os.path.join(kdir, "kaggle.json"), 0o600)
except OSError: pass
print("Kaggle auth:", _how, flush=True)

# --- 3) repo + competition data (reused if already present) ---
WORK = "/content/repo"
if not os.path.exists(WORK + "/src/data.py"):
    sh(f"git clone --depth 1 https://github.com/aRealGem/histopath-cancer-detection {WORK}")
sys.path.insert(0, WORK); os.chdir(WORK)

hits = glob.glob("/content/**/train_labels.csv", recursive=True)
if not hits:
    os.makedirs("/content/pcam", exist_ok=True)
    sh("kaggle competitions download -c histopathologic-cancer-detection -p /content/pcam")
    for z in glob.glob("/content/pcam/*.zip"):
        zipfile.ZipFile(z).extractall("/content/pcam")
    hits = glob.glob("/content/**/train_labels.csv", recursive=True)
ROOT = os.path.dirname(hits[0]); print("data root:", ROOT, flush=True)

# --- 4) data pipeline: canonical seed-1337 WSI-grouped val => OOF aligns with every member ---
import yaml
from src import data as D
SEED = 1337
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT
cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
cfg["seed"] = SEED
df = D.load_labels(cfg)
tr_df, va_df = D.split_train_val(cfg, df)
va_df = va_df.reset_index(drop=True)
ids_v = va_df["id"].tolist(); yv = va_df["label"].values.astype(int)
train_ds, val_ds = D.make_train_val_datasets(cfg, tr_df, va_df)   # val: no shuffle, no drop => id-aligned
test_ds, ids_t = D.make_test_dataset(cfg)
print(f"train~{len(tr_df)}  val={len(yv)}  test={len(ids_t)}", flush=True)

# --- 5) p4m-TinyVGG, seed-7 weight init; winning p4m_reg recipe (AdamW wd, val_loss cap) ---
def gblock(x, w, hin, hout):
    x = GConv2D(w, 3, h_input=hin, h_output=hout, padding="same", use_bias=False)(x)
    x = GBatchNorm(h=hout)(x); return layers.Activation("relu")(x)

def p4m_tinyvgg(widths=(8, 16, 32)):
    inp = Input((96, 96, 3), dtype="float32")
    x = layers.Rescaling(1/255.)(inp); x = layers.RandomContrast(0.1)(x)   # geometry handled by p4m
    x = gblock(x, widths[0], "Z2", "D4")
    x = gblock(x, widths[0], "D4", "D4"); x = layers.MaxPool2D(2)(x)
    x = gblock(x, widths[1], "D4", "D4"); x = gblock(x, widths[1], "D4", "D4"); x = layers.MaxPool2D(2)(x)
    x = gblock(x, widths[2], "D4", "D4"); x = gblock(x, widths[2], "D4", "D4"); x = layers.MaxPool2D(2)(x)
    x = gblock(x, widths[2], "D4", "D4")
    x = GroupPool(h_input="D4")(x); x = layers.GlobalAveragePooling2D()(x)
    return Model(inp, layers.Dense(1, activation="sigmoid", dtype="float32")(x), name="p4m_tinyvgg")

MEMBER, SEED_INIT, EPOCHS, LR, WD = "p4m_seed7", 7, 6, 5e-4, 1e-4
tf.keras.utils.set_random_seed(SEED_INIT)         # decorrelation: vary weight init only
try:   opt = keras.optimizers.AdamW(learning_rate=LR, weight_decay=WD, clipnorm=1.0)
except Exception: opt = keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0)
net = p4m_tinyvgg()
net.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            metrics=[keras.metrics.AUC(name="auc")])
ck = f"/content/{MEMBER}.keras"
net.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=[
    keras.callbacks.ModelCheckpoint(ck, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=8, restore_best_weights=True)])
net = tf.keras.models.load_model(ck, compile=False,
        custom_objects={"GConv2D": GConv2D, "GBatchNorm": GBatchNorm, "GroupPool": GroupPool})

# --- 6) predict (D4-invariant => no TTA) straight from the datasets, write both CSVs ---
def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

pv = net.predict(val_ds, verbose=0).ravel()
pt = net.predict(test_ds, verbose=0).ravel()
assert len(pv) == len(yv) == len(ids_v) and len(pt) == len(ids_t), "length/order mismatch"
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/oof_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
with open(f"{OUT}/sub_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
print(f"\n*** {MEMBER}: val AUROC {auroc(yv, pv):.6f}  (seed={SEED_INIT}, monitor=val_loss) ***", flush=True)

# --- 7) hand the CSVs back: Drive (if you mount) + a direct browser download ---
try:
    from google.colab import drive; drive.mount("/content/drive")
    dd = "/content/drive/MyDrive/histopath_out"; os.makedirs(dd, exist_ok=True)
    shutil.copy(f"{OUT}/oof_{MEMBER}.csv", dd); shutil.copy(f"{OUT}/sub_{MEMBER}.csv", dd)
    print("copied both CSVs to Drive:", dd, flush=True)
except Exception as e:
    print("Drive copy skipped:", e, flush=True)
try:
    from google.colab import files
    files.download(f"{OUT}/sub_{MEMBER}.csv"); files.download(f"{OUT}/oof_{MEMBER}.csv")
except Exception as e:
    print("auto-download skipped — grab them from /content/out/ :", e, flush=True)
