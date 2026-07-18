# =============================================================================
# p4m_autoloop_colab.py  —  LEGACY-Keras-2 poll-loop for the p4m job family
# =============================================================================
# The main autoloop (notebooks/autoloop_colab.py) runs under Keras 3 and CANNOT
# build/train the p4m (group-equivariant) members, which need the 2018 keras-gcnn
# library (Keras-2 API, TF_USE_LEGACY_KERAS=1). This notebook is the p4m-side twin
# of that loop: run it in a SEPARATE Colab tab (with the usual p4m setup cell) and
# it will poll the SAME job queue and handle only the p4m-family jobs:
#
#     job6  type=train    arch=p4m_tinyvgg               -> handle_train_p4m
#     job5  type=features arch="p4m_penultimate + ..."   -> handle_features_svm
#
# Output contract is IDENTICAL to the main loop (so the Pi/process.py decorrelation
# gate treats these like any other member): for each member it writes
#     oof_<member>.csv  (id,label,pred)   sub_<member>.csv  (id,label=prob)
#     job_<jobid>.json  (status=done, val stats, train_seconds, origin)
# and CUMULATIVELY pushes them to jackiemartindale/histopath-colab-out.
#
# COORDINATION with the Keras-3 loop (important):
#   * This loop selects ONLY p4m-family jobs; the Keras-3 loop is patched to SKIP
#     them (it neither runs nor needs_human-flags p4m jobs), so the two never fight.
#   * already_done() here treats a job as done ONLY if its manifest says
#     status=="done" — so a stale needs_human/error manifest (e.g. one the old
#     Keras-3 loop wrote before it was patched) does NOT block a real p4m run.
#
# HONESTY (same denylist as the queue): no external data, no test pseudo-labeling,
# period-authentic (<=2019) methods only. Provenance: origin flows from queue.json
# into each manifest; autonomous members carry origin="autonomous".
#
# ONE-TIME SETUP (jackie): run your existing p4m setup cell FIRST in this tab, i.e.
#   os.environ['TF_USE_LEGACY_KERAS']='1'; pip -q install tf-keras;
#   git clone tf2-GrouPy + tf2-keras-gcnn (on sys.path); Kaggle access-token auth;
#   Drive mounted with the p4m_reg checkpoint (job5's feature base).
# =============================================================================
import os
assert os.environ.get("TF_USE_LEGACY_KERAS") == "1", \
    "run your p4m setup cell first (TF_USE_LEGACY_KERAS=1 + tf-keras + keras-gcnn on sys.path)"

import sys, glob, json, time, shutil, zipfile, subprocess, traceback, csv
import numpy as np

# --- 2018-lib shims (identical to the other p4m notebooks) ---
for _n, _t in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _n):
        setattr(np, _n, _t)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as _L, layers, Model, Input
if not hasattr(_L.Layer, "_add_update_patched"):
    _orig_add_update = _L.Layer.add_update
    _L.Layer.add_update = lambda self, updates, inputs=None: _orig_add_update(self, updates)
    _L.Layer._add_update_patched = True

from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

COMP = "histopathologic-cancer-detection"
JOBS_DS = "jackiemartindale/histopath-jobs"
OUT_DS = "jackiemartindale/histopath-colab-out"
POLL_SECONDS = 120
SESSION_DEADLINE = time.time() + 23 * 3600      # self-stop before the Pro+ 24h cap
WORK = "/content/repo"
OUTDIR = "/content/p4m_out"                      # staged artifacts pushed per job
JOBSDIR = "/content/p4m_jobs"
SEED = 1337                                      # canonical val-split seed (OOF alignment)
CUSTOM = {"GConv2D": GConv2D, "GBatchNorm": GBatchNorm, "GroupPool": GroupPool}

# p4m checkpoint aliases (for job5's feature base), matching the other p4m notebooks.
_CKPT_ALIASES = {
    "p4m_reg": ("p4m_reg", "p4mreg", "reg"),
    "p4m_dense": ("p4m_dense", "densenet", "dense"),
}


# ----------------------------------------------------------------- setup helpers


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kaggle_auth():
    """Reuse the credential the p4m setup cell already wrote (~/.kaggle/access_token or
    kaggle.json); never overwrite a working one."""
    kdir = os.path.expanduser("~/.kaggle")
    os.environ["KAGGLE_CONFIG_DIR"] = kdir
    if os.path.exists(os.path.join(kdir, "access_token")) or os.path.exists(os.path.join(kdir, "kaggle.json")):
        return
    tok = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("KAGGLE_ACCESS_TOKEN")
        except Exception:
            tok = None
    assert tok, "no Kaggle credential — run your usual auth cell first (writes ~/.kaggle/access_token)."
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "access_token"), "w") as f:
        f.write(tok.strip())


def mount_drive():
    if not os.path.exists("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print("drive mount skipped:", e)


def stage_repo():
    if not os.path.exists(os.path.join(WORK, "src/data.py")):
        sh(["git", "clone", "--depth", "1",
            "https://github.com/aRealGem/histopath-cancer-detection", WORK])
    sys.path.insert(0, WORK)
    os.chdir(WORK)


def download_comp_data():
    """Return the dir holding train_labels.csv. First REUSE any copy already present
    anywhere under /content (a p4m tab is often warm from an earlier p4m script) — the
    same recursive-glob discovery the other p4m notebooks use; only download if absent."""
    hits = glob.glob("/content/**/train_labels.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    root = "/content/pcam"
    os.makedirs(root, exist_ok=True)
    sh(["kaggle", "competitions", "download", "-c", COMP, "-p", root])
    for z in glob.glob(os.path.join(root, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    hits = glob.glob("/content/**/train_labels.csv", recursive=True)
    return os.path.dirname(hits[0]) if hits else root


def discover_checkpoint(member):
    """Auto-find a p4m member's best.keras. Searches (1) THIS session's freshly-trained
    checkpoints at /content/*.keras (the train handlers save there — so job5's SVM can
    read features from a p4m the loop just trained, e.g. p4m_reg_vl, even when the old
    p4m_reg isn't persisted on Drive), then (2) Drive / $DRIVE_MODELS_DIR. Prefers paths
    containing 'histopath'. Returns a path or None. NOTE: /content is ephemeral — a runtime
    restart wipes session checkpoints; persisting train outputs to Drive is a future harden."""
    cands = glob.glob("/content/*.keras")                      # session-trained (top-level)
    roots = [os.environ.get("DRIVE_MODELS_DIR"), "/content/drive/MyDrive"]
    for r in roots:
        if r and os.path.isdir(r):
            cands += glob.glob(os.path.join(r, "**", "*.keras"), recursive=True)
    cands = sorted(set(cands))
    aliases = _CKPT_ALIASES.get(member, (member,))
    hits = [c for c in cands if any(a in c.lower() for a in aliases)]
    hits.sort(key=lambda c: (0 if "histopath" in c.lower() else 1, len(c)))
    return hits[0] if hits else None


# ----------------------------------------------------------------- data + metrics


import yaml


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


def base_cfg(root):
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["data"]["root"] = root
    cfg.setdefault("seed", SEED)
    cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
    return cfg


def fixed_val(cfg):
    """Decode the fixed seed-1337 WSI-grouped val split into arrays (ids,label,X). p4m nets
    have an internal Rescaling(1/255) so X stays in [0,255] here (as p4m_oof_dump does)."""
    from src import data as D
    df = D.load_labels(cfg)
    _, val_df = D.split_train_val(cfg, df)
    val_df = val_df.reset_index(drop=True)
    ids = val_df["id"].tolist()
    _, val_ds = D.make_train_val_datasets(cfg, df.iloc[:1], val_df)
    Xs, Ys = [], []
    for xb, yb in val_ds:
        Xs.append(xb.numpy()); Ys.append(yb.numpy())
    Xv = np.concatenate(Xs); yv = np.concatenate(Ys).astype(int)
    assert np.array_equal(yv, val_df["label"].values.astype(int)), "val order mismatch"
    return ids, yv, Xv, df


def test_arrays(cfg):
    from src import data as D
    ds, ids = D.make_test_dataset(cfg)
    Xs = [xb.numpy() for xb, _ in ds]
    return ids, np.concatenate(Xs)


# ----------------------------------------------------------------- artifact I/O


def emit_member(name, ids_v, yv, pred_v, ids_t, pred_t):
    with open(os.path.join(OUTDIR, f"oof_{name}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label", "pred"])
        for i, l, p in zip(ids_v, yv, pred_v):
            w.writerow([i, int(l), f"{p:.6f}"])
    with open(os.path.join(OUTDIR, f"sub_{name}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for i, p in zip(ids_t, pred_t):
            w.writerow([i, f"{p:.6f}"])


def write_manifest(job, members, val_stats, train_seconds, status="done", extra=None):
    man = {"jobid": job["jobid"], "status": status, "priority": job.get("priority", 999),
           "origin": job.get("origin", "autonomous"), "type": job.get("type"),
           "arch": job.get("arch"), "members": members, "val": val_stats,
           "train_seconds": round(train_seconds, 1), "hyperparams": job.get("hyperparams", {})}
    if extra:
        man.update(extra)
    json.dump(man, open(os.path.join(OUTDIR, f"job_{job['jobid']}.json"), "w"))


def push_out(msg):
    """Version colab-out CUMULATIVELY (merge current dataset with new OUTDIR files) so a
    push never wipes earlier jobs' artifacts. Mirrors autoloop_colab.push_out."""
    merge = "/content/p4m_out_merge"
    shutil.rmtree(merge, ignore_errors=True); os.makedirs(merge, exist_ok=True)
    dl = sh(["kaggle", "datasets", "download", "-d", OUT_DS, "-p", merge, "--unzip", "--force"])
    # GUARD: if the merge-download failed or came back suspiciously empty (a mid-session
    # auth hiccup did exactly this once and WIPED the bus), abort rather than push a
    # non-cumulative version. The job stays un-acked (no manifest) -> re-pushes next cycle.
    existing = [f for f in os.listdir(merge) if f != "dataset-metadata.json"]
    if dl.returncode != 0 or len(existing) < 3:
        print(f"push_out ABORT: colab-out download failed/empty (rc={dl.returncode}, "
              f"{len(existing)} files) -> refusing to push a wipe; retry next cycle.")
        return False
    for f in glob.glob(os.path.join(OUTDIR, "*")):
        shutil.copy(f, os.path.join(merge, os.path.basename(f)))   # new files win
    json.dump({"title": "histopath-colab-out", "id": OUT_DS, "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(merge, "dataset-metadata.json"), "w"))
    r = sh(["kaggle", "datasets", "version", "-p", merge, "-m", msg, "--dir-mode", "zip"])
    print("push_out:", r.returncode, (r.stdout + r.stderr)[-300:])
    return r.returncode == 0


# ----------------------------------------------------------------- p4m model


def gblock(x, w, hin, hout):
    x = GConv2D(w, 3, h_input=hin, h_output=hout, padding="same", use_bias=False)(x)
    x = GBatchNorm(h=hout)(x)
    x = layers.Activation("relu")(x)
    return x


def p4m_tinyvgg(widths=(8, 16, 32)):
    """Lean p4m (D4 = 4 rotations x 2 mirrors) equivariant VGG, ~217K params. Identical to
    notebooks/p4m_tinyvgg_colab.py:p4m_tinyvgg (kept inline so this loop is self-contained,
    matching the repo's one-file-per-p4m-notebook style). Internal Rescaling(1/255); GroupPool
    over the 8 orientations -> D4-invariant, so NO geometric aug and NO D4-TTA are needed."""
    inp = Input((96, 96, 3), dtype="float32")
    x = layers.Rescaling(1 / 255.)(inp)
    x = layers.RandomContrast(0.1)(x)                        # only aug; geometry handled by p4m
    x = gblock(x, widths[0], "Z2", "D4")                     # lift to p4m
    x = gblock(x, widths[0], "D4", "D4"); x = layers.MaxPool2D(2)(x)   # 48
    x = gblock(x, widths[1], "D4", "D4"); x = gblock(x, widths[1], "D4", "D4"); x = layers.MaxPool2D(2)(x)  # 24
    x = gblock(x, widths[2], "D4", "D4"); x = gblock(x, widths[2], "D4", "D4"); x = layers.MaxPool2D(2)(x)  # 12
    x = gblock(x, widths[2], "D4", "D4")
    x = GroupPool(h_input="D4")(x)                           # invariance: pool over 8 orientations
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return Model(inp, out, name="p4m_tinyvgg")


def _make_optimizer(lr, wd):
    """AdamW (decoupled weight decay) with gradient clipping — the p4m_reg recipe. Falls
    back to plain Adam if this legacy-Keras build lacks AdamW."""
    try:
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, clipnorm=1.0)
    except (AttributeError, TypeError):
        print("!! AdamW unavailable in this Keras build -> Adam (no decoupled weight decay)")
        return keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)


# ----------------------------------------------------------------- job handlers


def handle_train_p4m(job, cfg, ids_v, yv, Xv, ids_t, Xt):
    """job6: p4m-TinyVGG deep-ensemble member trained with a DIFFERENT seed (weight init),
    the successful p4m_reg recipe (AdamW wd, short hard epoch cap, val_LOSS checkpoint).

    OOF ALIGNMENT: the val SPLIT stays on the canonical seed-1337 (so this member's OOF rows
    line up with every other member's). The job's `seed` varies only the WEIGHT INIT via
    set_random_seed -> a genuinely different trained model (the decorrelation source) on the
    SAME data partition. D4-invariant => predict without TTA (TTA would be a no-op)."""
    from src import data as D
    hp = job.get("hyperparams", {})
    name = job["member_name"]
    monitor = "val_" + hp.get("monitor", "loss").replace("val_", "")   # -> 'val_loss'
    mode = hp.get("mode", "min")
    epochs = int(hp.get("epochs_cap", 6))
    wd = float(hp.get("weight_decay", 1e-4))
    lr = float(hp.get("lr", 5e-4))                 # constant LR (p4m_reg used no cosine)
    seed = int(hp.get("seed", 7))

    cfg = json.loads(json.dumps(cfg))
    cfg["seed"] = SEED                             # canonical split for OOF alignment (NOT `seed`)
    df = D.load_labels(cfg)
    tr_df, va_df = D.split_train_val(cfg, df)
    train_ds, val_ds = D.make_train_val_datasets(cfg, tr_df, va_df)

    tf.keras.utils.set_random_seed(seed)           # decorrelation: vary weight init only
    t0 = time.time()
    net = p4m_tinyvgg()
    net.compile(optimizer=_make_optimizer(lr, wd),
                loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                metrics=[keras.metrics.AUC(name="auc")])
    out_ckpt = f"/content/{name}.keras"
    net.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=[
        keras.callbacks.ModelCheckpoint(out_ckpt, monitor=monitor, mode=mode,
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=8,
                                      restore_best_weights=True)])
    net = tf.keras.models.load_model(out_ckpt, compile=False, custom_objects=CUSTOM)
    pv = net.predict(Xv, verbose=0, batch_size=512).ravel()   # D4-invariant -> no TTA
    pt = net.predict(Xt, verbose=0, batch_size=512).ravel()
    emit_member(name, ids_v, yv, pv, ids_t, pt)
    write_manifest(job, [name], {name: {"val_auroc": round(auroc(yv, pv), 6),
                   "monitor": monitor, "seed": seed}}, time.time() - t0)
    print(f"  {name}: val AUROC {auroc(yv, pv):.6f} (seed={seed}, monitor={monitor})")


def handle_features_svm(job, cfg, ids_v, yv, Xv, ids_t, Xt):
    """job5: cheap decorrelated member = logistic regression on a p4m net's penultimate
    (pre-final-Dense) features. Folds notebooks/p4m_svm_colab.py into the poll loop. Reuses
    the already-decoded val/test arrays; extracts train features on the fixed seed-1337 split."""
    from src import data as D
    name = job["member_name"]
    hp = job.get("hyperparams", {})
    base = hp.get("base_member", os.environ.get("SVM_BASE", "p4m_reg"))
    ckpt = discover_checkpoint(base)
    assert ckpt, f"no {base} checkpoint found on Drive (job5 needs a trained p4m to read features from)"
    t0 = time.time()
    net = tf.keras.models.load_model(ckpt, compile=False, custom_objects=CUSTOM)
    feat = Model(net.input, net.layers[-2].output)     # penultimate (pre final Dense)
    print(f"  features from {ckpt} -> dim {net.layers[-2].output.shape[-1]}")

    cfg = json.loads(json.dumps(cfg)); cfg["seed"] = SEED
    df = D.load_labels(cfg)
    tr_df, _ = D.split_train_val(cfg, df); tr_df = tr_df.reset_index(drop=True)

    def batched(ids):
        train_dir, _, _ = D._resolve(cfg)
        size = cfg["data"]["image_size"]; ext = cfg["data"]["image_ext"]; bs = cfg["train"]["batch_size"]
        paths = [str(train_dir / f"{i}{ext}") for i in ids]
        reader = D._make_reader(size)
        ds = tf.data.Dataset.from_tensor_slices((paths, np.zeros(len(paths), "float32")))
        return ds.map(reader, num_parallel_calls=tf.data.AUTOTUNE).batch(bs).prefetch(tf.data.AUTOTUNE)

    def feats_from_array(X):
        out = []
        for i in range(0, len(X), 512):
            f = feat.predict(X[i:i + 512], verbose=0)
            out.append(f.reshape(f.shape[0], -1))
        return np.concatenate(out)

    Xtr = np.concatenate([feat.predict(xb, verbose=0).reshape(xb.shape[0], -1)
                          for xb, _ in batched(tr_df["id"].tolist())])
    ytr = tr_df["label"].values.astype(int)
    Xva = feats_from_array(Xv)
    Xte = feats_from_array(Xt)

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    pv = clf.predict_proba(sc.transform(Xva))[:, 1]
    pt = clf.predict_proba(sc.transform(Xte))[:, 1]
    emit_member(name, ids_v, yv, pv, ids_t, pt)
    write_manifest(job, [name], {name: {"val_auroc": round(auroc(yv, pv), 6), "base": base}},
                   time.time() - t0)
    print(f"  {name}: val AUROC {auroc(yv, pv):.6f} (logreg on {base} penultimate features)")


# ----------------------------------------------------------------- poll loop


def is_p4m_job(job):
    """True for the p4m-family jobs THIS loop owns (mirror of _p4m_owned in autoloop_colab)."""
    arch = job.get("arch") or ""
    return (job.get("type") == "train" and arch == "p4m_tinyvgg") \
        or (job.get("type") == "features" and "p4m" in arch)


def already_done(jobid):
    """Skip only if a manifest exists AND says status=='done' — so a stale needs_human/error
    manifest (e.g. from the Keras-3 loop before it was patched) does NOT block a real run."""
    sh(["kaggle", "datasets", "download", "-d", OUT_DS, "-p", JOBSDIR, "--unzip", "--force"])
    p = os.path.join(JOBSDIR, f"job_{jobid}.json")
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def fetch_queue():
    shutil.rmtree(JOBSDIR, ignore_errors=True); os.makedirs(JOBSDIR, exist_ok=True)
    r = sh(["kaggle", "datasets", "download", "-d", JOBS_DS, "-p", JOBSDIR, "--unzip", "--force"])
    qp = os.path.join(JOBSDIR, "queue.json")
    if not os.path.exists(qp):
        print("no queue.json yet:", (r.stdout + r.stderr)[-200:]); return None
    return json.load(open(qp))


def main():
    kaggle_auth(); stage_repo(); mount_drive()
    root = download_comp_data()
    cfg = base_cfg(root)
    print("decoding fixed val split + test set ...")
    ids_v, yv, Xv, _ = fixed_val(cfg)
    ids_t, Xt = test_arrays(cfg)
    print(f"val n={len(ids_v)} pos={yv.mean():.3f} | test n={len(ids_t)}")

    while time.time() < SESSION_DEADLINE:
        q = fetch_queue()
        if not q:
            time.sleep(POLL_SECONDS); continue
        pending = [j for j in sorted(q["jobs"], key=lambda x: x.get("priority", 999))
                   if j.get("status") == "pending" and is_p4m_job(j)]
        job = next((j for j in pending if not already_done(j["jobid"])), None)
        if not job:
            print("no runnable p4m job; idle."); time.sleep(POLL_SECONDS); continue

        shutil.rmtree(OUTDIR, ignore_errors=True); os.makedirs(OUTDIR, exist_ok=True)
        print(f"\n=== running {job['jobid']} ({job.get('type')}/{job.get('arch')}) ===")
        try:
            if job.get("type") == "train" and job.get("arch") == "p4m_tinyvgg":
                handle_train_p4m(job, cfg, ids_v, yv, Xv, ids_t, Xt)
            elif job.get("type") == "features":
                handle_features_svm(job, cfg, ids_v, yv, Xv, ids_t, Xt)
            else:
                # Shouldn't happen (is_p4m_job gates selection) — flag rather than silently skip.
                write_manifest(job, [], {}, 0.0, status="needs_human",
                               extra={"reason": f"p4m loop has no handler for '{job.get('arch')}'"})
            push_out(f"[p4m-autoloop] {job['jobid']} done")
        except Exception:
            traceback.print_exc()
            os.makedirs(OUTDIR, exist_ok=True)
            write_manifest(job, [], {}, 0.0, status="error",
                           extra={"trace": traceback.format_exc()[-1500:]})
            push_out(f"[p4m-autoloop] {job['jobid']} ERROR")
        time.sleep(POLL_SECONDS)

    print("session deadline reached; stopping p4m poll loop (restart the tab to resume).")


if __name__ == "__main__":
    main()
