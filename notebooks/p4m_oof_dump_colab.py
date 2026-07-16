# =============================================================================
# p4m_oof_dump_colab.py  —  one-time LEGACY-Keras OOF dump for the p4m members
# =============================================================================
# The autoloop poll-loop (autoloop_colab.py) runs under Keras 3 and can dump OOF for the
# Keras-3 members (champion, tinyvgg) but NOT the p4m nets, which need legacy Keras 2 +
# keras-gcnn. Run THIS once in your existing p4m notebook (after the usual p4m setup cell
# that installs tf-keras + tf2-keras-gcnn/tf2-GrouPy and sets TF_USE_LEGACY_KERAS=1). It
# dumps oof_/sub_ for p4m_reg + p4m_dense on the SAME fixed seed-1337 val split and pushes
# them to jackiemartindale/histopath-colab-out with a manifest job_p4moof.json, so the Pi
# side gets all four members and can reproduce the ~0.9523 champion proxy.
# =============================================================================
import os
assert os.environ.get("TF_USE_LEGACY_KERAS") == "1", \
    "run your p4m setup cell first (TF_USE_LEGACY_KERAS=1 + tf-keras + keras-gcnn)"
import os, sys, glob, json, csv, time, shutil, numpy as np

# --- 2018-lib shims (same as the p4m training scripts) ---
if not hasattr(np, "int"):
    np.int = int; np.float = float; np.bool = bool
import tensorflow as tf
from tensorflow.keras import layers as _L
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
from src import data as D

OUT_DS = "jackiemartindale/histopath-colab-out"
OUTDIR = "/content/p4m_oof_out"
SEED = 1337
CUSTOM = {"GConv2D": GConv2D, "GBatchNorm": GBatchNorm, "GroupPool": GroupPool}


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


# --- fixed val split + test set (identical to autoloop_colab.py) ---
root = glob.glob("/content/pcam") or glob.glob("/content/**/train_labels.csv", recursive=True)
cfg = yaml.safe_load(open("configs/baseline.yaml"))
cands = glob.glob("/content/**/train_labels.csv", recursive=True)
assert cands, "download the competition data first (same as your p4m runs)"
cfg["data"]["root"] = os.path.dirname(cands[0]); cfg.setdefault("seed", SEED)
cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"

df = D.load_labels(cfg)
_, val_df = D.split_train_val(cfg, df); val_df = val_df.reset_index(drop=True)
ids_v = val_df["id"].tolist()
_, val_ds = D.make_train_val_datasets(cfg, df.iloc[:1], val_df)
Xv = np.concatenate([xb.numpy() for xb, _ in val_ds])
yv = np.concatenate([yb.numpy() for _, yb in val_ds]).astype(int)
test_ds, ids_t = D.make_test_dataset(cfg)
Xt = np.concatenate([xb.numpy() for xb, _ in test_ds])
print(f"val n={len(ids_v)} pos={yv.mean():.3f} | test n={len(ids_t)}")

os.makedirs(OUTDIR, exist_ok=True)
MEMBERS = ["p4m_reg", "p4m_dense"]
ALIASES = {"p4m_reg": ("p4m_reg", "p4mreg", "reg"), "p4m_dense": ("p4m_dense", "densenet", "dense")}
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive", force_remount=False)
except Exception:
    pass

kaggle_dumped, vstats = [], {}
allk = sorted(set(glob.glob("/content/drive/MyDrive/**/*.keras", recursive=True)
                  + glob.glob((os.environ.get("DRIVE_MODELS_DIR") or "/content") + "/**/*.keras", recursive=True)))
for m in MEMBERS:
    hits = [k for k in allk if any(a in k.lower() for a in ALIASES[m])]
    hits.sort(key=lambda c: (0 if "histopath" in c.lower() else 1, len(c)))
    if not hits:
        print("!! no checkpoint for", m); continue
    net = tf.keras.models.load_model(hits[0], compile=False, custom_objects=CUSTOM)
    pv = net.predict(Xv, verbose=0, batch_size=512).ravel()   # D4-invariant -> no TTA
    pt = net.predict(Xt, verbose=0, batch_size=512).ravel()
    with open(f"{OUTDIR}/oof_{m}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label", "pred"])
        for i, l, p in zip(ids_v, yv, pv): w.writerow([i, int(l), f"{p:.6f}"])
    with open(f"{OUTDIR}/sub_{m}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for i, p in zip(ids_t, pt): w.writerow([i, f"{p:.6f}"])
    vstats[m] = {"val_auroc": round(auroc(yv, pv), 6)}
    kaggle_dumped.append(m); print(f"  {m}: val AUROC {vstats[m]['val_auroc']}  ({hits[0]})")

json.dump({"jobid": "p4moof", "status": "done", "priority": 0, "origin": "human",
           "type": "oof_dump", "members": kaggle_dumped, "val": vstats, "train_seconds": 0},
          open(f"{OUTDIR}/job_p4moof.json", "w"))
json.dump({"title": "histopath-colab-out", "id": OUT_DS, "licenses": [{"name": "CC0-1.0"}]},
          open(f"{OUTDIR}/dataset-metadata.json", "w"))
import subprocess
r = subprocess.run(["kaggle", "datasets", "version", "-p", OUTDIR,
                    "-m", "p4m OOF dump (legacy keras)", "--dir-mode", "zip"],
                   capture_output=True, text=True)
print("push:", r.returncode, (r.stdout + r.stderr)[-300:])
print("DONE — p4m OOF for", kaggle_dumped, "shipped to colab-out")
