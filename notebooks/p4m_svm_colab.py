# =============================================================================
# p4m_svm_colab.py  —  job5: cheap decorrelated member = logistic/linear-SVM on p4m
#                      penultimate features (LEGACY Keras 2 env, run in the p4m notebook)
# =============================================================================
# A classifier on the p4m net's learned features is a fast, genuinely different member
# (linear head over equivariant features) that tends to decorrelate from the deep members.
# Runs in your p4m notebook (needs TF_USE_LEGACY_KERAS=1 + keras-gcnn + the comp data).
# Emits oof_p4m_svm.csv + sub_p4m_svm.csv + job_job5.json to histopath-colab-out
# (cumulative push); the Pi decorrelation gate then treats it like any other member.
# =============================================================================
import os
assert os.environ.get("TF_USE_LEGACY_KERAS") == "1", \
    "run your p4m setup cell first (TF_USE_LEGACY_KERAS=1 + tf-keras + keras-gcnn)"
import sys, glob, json, csv, time, shutil, numpy as np

if not hasattr(np, "int"):
    np.int = int; np.float = float; np.bool = bool
import tensorflow as tf
from tensorflow.keras import layers as _L, Model
if not hasattr(_L.Layer, "_add_update_patched"):
    _orig = _L.Layer.add_update
    _L.Layer.add_update = lambda self, updates, inputs=None: _orig(self, updates)
    _L.Layer._add_update_patched = True

WORK = "/content/repo"
if not os.path.exists(os.path.join(WORK, "src/data.py")):
    import subprocess
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/aRealGem/histopath-cancer-detection", WORK])
sys.path.insert(0, WORK); os.chdir(WORK)

import yaml
from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src import data as D

OUT_DS = "jackiemartindale/histopath-colab-out"
OUTDIR = "/content/p4m_svm_out"
SEED = 1337
CUSTOM = {"GConv2D": GConv2D, "GBatchNorm": GBatchNorm, "GroupPool": GroupPool}
BASE_MEMBER = os.environ.get("SVM_BASE", "p4m_reg")   # which p4m checkpoint to read features from


def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sa = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)) if npos and nneg else float("nan")


# --- config + fixed split ---
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cands = glob.glob("/content/**/train_labels.csv", recursive=True)
assert cands, "download the competition data first (same as your p4m runs)"
cfg["data"]["root"] = os.path.dirname(cands[0]); cfg.setdefault("seed", SEED)
cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"

df = D.load_labels(cfg)
tr_df, va_df = D.split_train_val(cfg, df)
tr_df = tr_df.reset_index(drop=True); va_df = va_df.reset_index(drop=True)


def ordered_batches(ids):
    """Batched images in the given id order (no shuffle), reusing the src.data reader."""
    train_dir, _, _ = D._resolve(cfg)
    size = cfg["data"]["image_size"]; ext = cfg["data"]["image_ext"]; bs = cfg["train"]["batch_size"]
    paths = [str(train_dir / f"{i}{ext}") for i in ids]
    reader = D._make_reader(size)
    ds = tf.data.Dataset.from_tensor_slices((paths, np.zeros(len(paths), "float32")))
    return ds.map(reader, num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(tf.data.AUTOTUNE)


# --- load p4m checkpoint, build penultimate-feature extractor ---
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
except Exception:
    pass
alias = {"p4m_reg": ("p4m_reg", "p4mreg", "reg"), "p4m_dense": ("p4m_dense", "densenet", "dense")}[BASE_MEMBER]
allk = sorted(set(glob.glob("/content/drive/MyDrive/**/*.keras", recursive=True)))
hits = [k for k in allk if any(a in k.lower() for a in alias)]
hits.sort(key=lambda c: (0 if "histopath" in c.lower() else 1, len(c)))
assert hits, f"no {BASE_MEMBER} checkpoint found on Drive"
net = tf.keras.models.load_model(hits[0], compile=False, custom_objects=CUSTOM)
feat_model = Model(net.input, net.layers[-2].output)   # penultimate (pre final Dense)
print("features from:", hits[0], "-> dim", net.layers[-2].output.shape[-1])


def features(ids):
    out = []
    for xb, _ in ordered_batches(ids):
        f = feat_model.predict(xb, verbose=0)
        out.append(f.reshape(f.shape[0], -1))
    return np.concatenate(out)


t0 = time.time()
print("extracting features: train", len(tr_df), "val", len(va_df))
Xtr = features(tr_df["id"].tolist()); ytr = tr_df["label"].values.astype(int)
Xva = features(va_df["id"].tolist()); yva = va_df["label"].values.astype(int)
test_ds, ids_t = D.make_test_dataset(cfg)
Xte = []
for xb, _ in test_ds:
    f = feat_model.predict(xb, verbose=0); Xte.append(f.reshape(f.shape[0], -1))
Xte = np.concatenate(Xte)

# --- fit logistic regression on standardized features ---
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
pv = clf.predict_proba(sc.transform(Xva))[:, 1]
pt = clf.predict_proba(sc.transform(Xte))[:, 1]
print(f"p4m_svm val AUROC {auroc(yva, pv):.4f}  (base {BASE_MEMBER})")

# --- emit + cumulative push ---
os.makedirs(OUTDIR, exist_ok=True)
with open(f"{OUTDIR}/oof_p4m_svm.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label", "pred"])
    for i, l, p in zip(va_df["id"], yva, pv): w.writerow([i, int(l), f"{p:.6f}"])
with open(f"{OUTDIR}/sub_p4m_svm.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id", "label"])
    for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
json.dump({"jobid": "job5", "status": "done", "priority": 5, "origin": "autonomous",
           "type": "features", "arch": f"p4m_penultimate+logreg", "members": ["p4m_svm"],
           "val": {"p4m_svm": {"val_auroc": round(auroc(yva, pv), 6)}},
           "train_seconds": round(time.time() - t0, 1)}, open(f"{OUTDIR}/job_job5.json", "w"))

import subprocess
MERGE = "/content/p4m_svm_merge"
shutil.rmtree(MERGE, ignore_errors=True); os.makedirs(MERGE, exist_ok=True)
subprocess.run(["kaggle", "datasets", "download", "-d", OUT_DS, "-p", MERGE, "--unzip", "--force"],
               capture_output=True, text=True)
for fp in glob.glob(OUTDIR + "/*"):
    shutil.copy(fp, os.path.join(MERGE, os.path.basename(fp)))
json.dump({"title": "histopath-colab-out", "id": OUT_DS, "licenses": [{"name": "CC0-1.0"}]},
          open(f"{MERGE}/dataset-metadata.json", "w"))
r = subprocess.run(["kaggle", "datasets", "version", "-p", MERGE, "-m", "p4m_svm member (job5)",
                    "--dir-mode", "zip"], capture_output=True, text=True)
print("push:", r.returncode, (r.stdout + r.stderr)[-300:])
print("DONE — p4m_svm shipped to colab-out")
