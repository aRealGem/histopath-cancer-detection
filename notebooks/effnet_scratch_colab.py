# ============================================================================
# EfficientNetV2B0 FROM-SCRATCH — ONE self-contained Colab cell (A100/GPU).
# A NON-equivariant, genuinely-different-representation member (all our recent
# members are equivariant/steerable). Standard TF/Keras — NO legacy-keras, NO escnn,
# NO numpy pin. Ships to histopath-effnet-out via Kaggle. Reads secret KAGGLE_KEY.
# Runtime > A100/GPU + High-RAM. Keep the tab FOREGROUND (~1-2h; from-scratch is slow).
# ============================================================================
KAGGLE_USERNAME = "jackiemartindale"
OUT_DS = "histopath-effnet-out"
MEMBER = "effnet_scratch"
EPOCHS, BATCH = 15, 256                           # from-scratch needs epochs; early-stops on val_loss

import os, sys, glob, csv, json, time, subprocess, shutil, zipfile
def sh(cmd):
    print("$", cmd, flush=True); return subprocess.run(cmd, shell=True, text=True).returncode

sh("pip -q install kaggle")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
assert tf.config.list_physical_devices("GPU"), "NO GPU. Runtime > A100/GPU + High-RAM, then re-run."
print("TF", tf.__version__, "| GPU OK", flush=True)

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

# --- canonical seed-1337 split (OOF aligns with every member); data = raw [0,255] float ---
import yaml
from src import data as D
SEED = 1337
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cfg["data"]["root"] = ROOT; cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
cfg["seed"] = SEED; cfg["train"]["batch_size"] = BATCH
df = D.load_labels(cfg); tr_df, va_df = D.split_train_val(cfg, df)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)
ids_v = va_df["id"].tolist(); yv = va_df["label"].values.astype(int)
train_ds, val_ds = D.make_train_val_datasets(cfg, tr_df, va_df)   # raw decode (aug is in the model below)
test_ds, ids_t = D.make_test_dataset(cfg)
print(f"train={len(tr_df)} val={len(yv)} test={len(ids_t)}", flush=True)

# --- EfficientNetV2B0 from-scratch (non-equivariant) + geometric aug in-model ---
tf.keras.utils.set_random_seed(SEED)
bb = tf.keras.applications.EfficientNetV2B0(input_shape=(96, 96, 3), include_top=False,
                                            weights=None, include_preprocessing=True)  # normalizes [0,255] internally
inp = Input((96, 96, 3), dtype="float32")
x = layers.RandomFlip("horizontal_and_vertical")(inp)
x = layers.RandomRotation(0.5, fill_mode="reflect")(x)
x = layers.RandomContrast(0.1)(x)
x = bb(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
net = Model(inp, out, name="effnetv2b0_scratch")
opt = keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
net.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05), metrics=[keras.metrics.AUC(name="auc")])
ck = f"/content/{MEMBER}.keras"
net.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2, callbacks=[
    keras.callbacks.ModelCheckpoint(ck, monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=4, restore_best_weights=True)])
net = tf.keras.models.load_model(ck, compile=False)

def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sa = s[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]: j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos*(npos+1)/2.0) / (npos*nneg)) if npos and nneg else float("nan")

pv = net.predict(val_ds, verbose=0).ravel(); pt = net.predict(test_ds, verbose=0).ravel()
assert len(pv) == len(yv) == len(ids_v) and len(pt) == len(ids_t), "length/order mismatch"
OUT = "/content/out"; os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/oof_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
with open(f"{OUT}/sub_{MEMBER}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
print(f"\n*** {MEMBER}: val AUROC {auroc(yv, pv):.6f} (EfficientNetV2B0 from-scratch, non-equivariant) ***", flush=True)

# --- exfil via Kaggle dataset (NO Drive) ---
STAGE = "/content/eff_ds"; shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(STAGE)
shutil.copy(f"{OUT}/sub_{MEMBER}.csv", STAGE); shutil.copy(f"{OUT}/oof_{MEMBER}.csv", STAGE)
json.dump({"title": OUT_DS, "id": f"{KAGGLE_USERNAME}/{OUT_DS}", "licenses": [{"name": "CC0-1.0"}]}, open(f"{STAGE}/dataset-metadata.json", "w"))
_r = subprocess.run("kaggle datasets create -p " + STAGE, shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
if any(s in _o.lower() for s in ("already exists", "already in use", "409")):
    _r = subprocess.run("kaggle datasets version -p " + STAGE + " -m effnet", shell=True, text=True, capture_output=True); _o = _r.stdout + _r.stderr
print(_o, flush=True)
print(f">>> ALL DONE. Tell Cass to pull {KAGGLE_USERNAME}/{OUT_DS}", flush=True)
