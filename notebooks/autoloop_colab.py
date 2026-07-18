# =============================================================================
# autoloop_colab.py  —  Colab A100 poll-loop for the PCam autonomous goal-seek loop
# =============================================================================
# Runs in a single Colab Pro+ cell (background execution; ~24h session cap). Polls the
# Kaggle dataset  jackiemartindale/histopath-jobs  (queue.json) for the highest-priority
# pending job, trains/infers it with val_LOSS checkpointing on the FIXED seed-1337
# WSI-grouped val split, and pushes three artifacts per member back to
# jackiemartindale/histopath-colab-out for the Pi (Cass) side to blend/score/submit:
#     oof_<member>.csv   (id,label,pred)      <- honest offline-proxy signal
#     sub_<member>.csv   (id,label=prob)      <- test submission (TTA where applicable)
#     job_<jobid>.json   (status=done, ...)   <- completion signal + val stats + origin
#
# Idempotent: a job whose job_<jobid>.json already exists in colab-out is skipped, so a
# tab restart resumes cleanly. HONESTY: no external data, no test pseudo-labeling; members
# are period-authentic (<=2019). Provenance: origin flows from queue.json into each manifest.
#
# ONE-TIME SETUP (jackie): set two Colab secrets (userdata) -> KAGGLE_ACCESS_TOKEN (the
# KGAT_ NEW-format token) and DRIVE_MODELS_DIR (folder holding the reproduced checkpoints:
# champion/best.keras, tinyvgg/best.keras, p4m_reg/best.keras, p4m_dense/best.keras).
# =============================================================================
import os, sys, glob, json, time, shutil, gzip, zipfile, subprocess, traceback

# KERAS MODE: this loop runs under Keras 3 (champion, TinyVGG, and job1 are Keras-3 models).
# The p4m members need legacy Keras 2 (tf-keras + keras-gcnn) which CANNOT coexist with
# Keras 3 in one runtime -> p4m OOF is dumped once from the separate legacy p4m notebook,
# and p4m TRAIN jobs (job2/job6) are handled there, not in this loop. Set
# TF_USE_LEGACY_KERAS=1 in the environment ONLY if you intentionally run a p4m-only loop.

COMP = "histopathologic-cancer-detection"
JOBS_DS = "jackiemartindale/histopath-jobs"
OUT_DS = "jackiemartindale/histopath-colab-out"
POLL_SECONDS = 120
SESSION_DEADLINE = time.time() + 23 * 3600     # self-stop before the Pro+ 24h cap
WORK = "/content/repo"
OUTDIR = "/content/out"                          # staged artifacts pushed per job
JOBSDIR = "/content/jobs"
SEED = 1337

# ----------------------------------------------------------------- setup helpers


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kaggle_auth():
    """Reuse the credentials the existing A100 notebook already established — no new
    secret. Order: an already-present ~/.kaggle/access_token (the ACCESS_TOKEN method we
    use) or kaggle.json; else an env/secret token as a last resort. Never overwrites a
    working credential."""
    kdir = os.path.expanduser("~/.kaggle")
    os.environ["KAGGLE_CONFIG_DIR"] = kdir
    if os.path.exists(os.path.join(kdir, "access_token")) or os.path.exists(os.path.join(kdir, "kaggle.json")):
        return  # already authenticated by the notebook's setup cell
    tok = os.environ.get("KAGGLE_ACCESS_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("KAGGLE_ACCESS_TOKEN")
        except Exception:
            tok = None
    assert tok, ("no Kaggle credential found — run your usual auth cell first (writes "
                 "~/.kaggle/access_token), or set KAGGLE_ACCESS_TOKEN.")
    os.makedirs(kdir, exist_ok=True)
    with open(os.path.join(kdir, "access_token"), "w") as f:
        f.write(tok.strip())


# Member -> substrings that identify its checkpoint folder/file on Drive.
_CKPT_ALIASES = {
    "champion": ("champion", "mobilenetv3", "mnv3", "baseline"),
    "tinyvgg": ("tinyvgg", "tiny_vgg", "vgg", "scratch_vgg", "exp_scratch_vgg"),
    "p4m_reg": ("p4m_reg", "p4mreg", "reg"),
    "p4m_dense": ("p4m_dense", "densenet", "dense"),
}


def mount_drive():
    if not os.path.exists("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print("drive mount skipped:", e)


def discover_checkpoints(members):
    """Auto-find each member's best.keras anywhere under Drive (or $DRIVE_MODELS_DIR),
    so nothing new needs to be configured — these are the checkpoints the earlier code
    saved. Matches by member-name aliases; prefers paths containing 'histopath'."""
    roots = [os.environ.get("DRIVE_MODELS_DIR"), "/content/drive/MyDrive"]
    cands = []
    for r in roots:
        if r and os.path.isdir(r):
            cands += glob.glob(os.path.join(r, "**", "*.keras"), recursive=True)
    cands = sorted(set(cands))
    found = {}
    for m in members:
        aliases = _CKPT_ALIASES.get(m, (m,))
        hits = [c for c in cands if any(a in c.lower() for a in aliases)]
        hits.sort(key=lambda c: (0 if "histopath" in c.lower() else 1, len(c)))
        if hits:
            found[m] = hits[0]
        else:
            print(f"!! no Drive checkpoint matched member '{m}' (aliases {aliases})")
    print("discovered checkpoints:", {k: v for k, v in found.items()})
    return found


def stage_repo():
    if not os.path.exists(os.path.join(WORK, "src/data.py")):
        sh(["git", "clone", "--depth", "1",
            "https://github.com/aRealGem/histopath-cancer-detection", WORK])
    sys.path.insert(0, WORK)
    os.chdir(WORK)


def download_comp_data():
    """Competition tif patches (re-downloaded each session; too many files for Drive)."""
    root = "/content/pcam"
    if os.path.exists(os.path.join(root, "train_labels.csv")):
        return root
    os.makedirs(root, exist_ok=True)
    sh(["kaggle", "competitions", "download", "-c", COMP, "-p", root])
    for z in glob.glob(os.path.join(root, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
    return root


# ----------------------------------------------------------------- core objects

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def auroc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sa = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0; i = j + 1
    npos = int(y.sum()); nneg = len(y) - npos
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)) if npos and nneg else float("nan")


@tf.keras.utils.register_keras_serializable(package="histopath")
class GradientReversal(layers.Layer):
    def __init__(self, lamb=1.0, **k): super().__init__(**k); self.lamb = float(lamb)
    def call(self, x):
        lamb = self.lamb
        @tf.custom_gradient
        def _r(z):
            return tf.identity(z), (lambda dy: -lamb * dy)
        return _r(x)
    def get_config(self): c = super().get_config(); c.update(lamb=self.lamb); return c


@tf.keras.utils.register_keras_serializable(package="histopath")
class RandomHEDJitter(layers.Layer):
    _S = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]
    def __init__(self, sigma=0.05, **k): super().__init__(**k); self.sigma = float(sigma)
    def build(self, s):
        self._Sm = tf.constant(self._S, tf.float32); self._D = tf.linalg.inv(self._Sm); super().build(s)
    def call(self, x, training=None): return x  # identity at inference
    def get_config(self): c = super().get_config(); c.update(sigma=self.sigma); return c


def base_cfg(root):
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    cfg["data"]["root"] = root
    cfg.setdefault("seed", SEED)
    cfg["data"]["wsi_map_csv"] = "data/wsi/patch_id_wsi_full.csv.gz"
    return cfg


def fixed_val(cfg):
    """Decode the fixed seed-1337 WSI-grouped val split into arrays (ids,label,X)."""
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


def _views(x):
    return [x, x[:, :, ::-1, :], x[:, ::-1, :, :], x[:, ::-1, ::-1, :],
            np.rot90(x, 1, (1, 2)), np.rot90(x, 2, (1, 2)), np.rot90(x, 3, (1, 2)),
            np.rot90(x[:, :, ::-1, :], 1, (1, 2))]


def predict(net, X, tta):
    if not tta:
        return net.predict(X, verbose=0, batch_size=512).ravel()
    acc = np.zeros(len(X))
    for v in _views(X):
        acc += net.predict(v, verbose=0, batch_size=512).ravel()
    return acc / 8.0


# ----------------------------------------------------------------- artifact push


def push_out(msg):
    """Version colab-out CUMULATIVELY: merge the current dataset with the new files in
    OUTDIR so a job's push never wipes earlier jobs' artifacts (which would also make the
    'already done?' check flap and re-run jobs). Two notebooks pushing at the same second
    can still race, but runs here are human-paced."""
    merge = "/content/out_merge"
    shutil.rmtree(merge, ignore_errors=True); os.makedirs(merge, exist_ok=True)
    dl = sh(["kaggle", "datasets", "download", "-d", OUT_DS, "-p", merge, "--unzip", "--force"])
    # GUARD: if the merge-download failed or came back suspiciously empty (a mid-session
    # auth hiccup did this once and WIPED the bus), abort rather than push a non-cumulative
    # version. The job stays un-acked (no manifest) -> re-pushes next cycle.
    existing = [f for f in os.listdir(merge) if f != "dataset-metadata.json"]
    if dl.returncode != 0 or len(existing) < 3:
        print(f"push_out ABORT: colab-out download failed/empty (rc={dl.returncode}, "
              f"{len(existing)} files) -> refusing to push a wipe; retry next cycle.")
        return False
    for f in glob.glob(os.path.join(OUTDIR, "*")):
        shutil.copy(f, os.path.join(merge, os.path.basename(f)))   # new files win
    meta = {"title": "histopath-colab-out", "id": OUT_DS, "licenses": [{"name": "CC0-1.0"}]}
    json.dump(meta, open(os.path.join(merge, "dataset-metadata.json"), "w"))
    r = sh(["kaggle", "datasets", "version", "-p", merge, "-m", msg, "--dir-mode", "zip"])
    print("push_out:", r.returncode, (r.stdout + r.stderr)[-300:])
    return r.returncode == 0


def emit_member(name, ids_v, yv, pred_v, ids_t, pred_t):
    import csv
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


# ----------------------------------------------------------------- job handlers


def handle_oof_dump(job, cfg, ids_v, yv, Xv, ids_t, Xt):
    """Bootstrap: OOF (+ test sub) for the existing champions, auto-discovering their
    checkpoints on Drive (no configured path needed)."""
    members, vstats = [], {}
    t0 = time.time()
    ckpts = discover_checkpoints(job["members"])
    skipped = []
    for name in job["members"]:
        if name not in ckpts:
            print("!! missing checkpoint for", name, "-> skip"); skipped.append(name); continue
        try:
            net = tf.keras.models.load_model(ckpts[name], compile=False)
            tta = "p4m" not in name    # p4m nets are D4-invariant -> TTA no-op
            pv = predict(net, Xv, tta); pt = predict(net, Xt, tta)
        except Exception as e:
            # Expected for p4m under Keras 3 (needs legacy Keras 2) -> dump those from the
            # legacy p4m notebook instead. Don't fail the whole bootstrap job.
            print(f"!! {name} load/predict failed under this Keras runtime -> skip ({e})")
            skipped.append(name); continue
        emit_member(name, ids_v, yv, pv, ids_t, pt)
        vstats[name] = {"val_auroc": round(auroc(yv, pv), 6)}
        members.append(name); print(f"  {name}: val AUROC {vstats[name]['val_auroc']}")
    write_manifest(job, members, vstats, time.time() - t0,
                   extra={"skipped": skipped} if skipped else None)


def _fit_keras(cfg, monitor, mode, epochs, out_ckpt):
    from src import data as D
    import src.model as M
    df = D.load_labels(cfg)
    tr_df, va_df = D.split_train_val(cfg, df)
    train_ds, val_ds = D.make_train_val_datasets(cfg, tr_df, va_df)
    net = M.build_model(cfg)
    M.compile_model(net, cfg["train"].get("lr_head", 1e-3), cfg)
    cbs = [keras.callbacks.ModelCheckpoint(out_ckpt, monitor=monitor, mode=mode,
                                           save_best_only=True, verbose=1),
           keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=8,
                                         restore_best_weights=True)]
    net.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)
    return tf.keras.models.load_model(out_ckpt, compile=False)


def handle_train_keras(job, cfg, ids_v, yv, Xv, ids_t, Xt):
    """TinyVGG / from-scratch MobileNet member with val_LOSS checkpointing."""
    hp = job.get("hyperparams", {})
    name = job["member_name"]
    cfg = json.loads(json.dumps(cfg))                     # deep copy
    cfg["model"]["backbone"] = {"TinyVGG": "TinyVGG"}.get(job["arch"], "MobileNetV3Small")
    cfg["model"]["from_scratch"] = True
    monitor = "val_" + hp.get("monitor", "loss").replace("val_", "")
    mode = hp.get("mode", "min")
    t0 = time.time()
    net = _fit_keras(cfg, monitor, mode, int(hp.get("epochs_cap", 30)),
                     f"/content/{name}.keras")
    tta = bool(hp.get("tta", True))
    pv = predict(net, Xv, tta); pt = predict(net, Xt, tta)
    emit_member(name, ids_v, yv, pv, ids_t, pt)
    write_manifest(job, [name], {name: {"val_auroc": round(auroc(yv, pv), 6),
                   "monitor": monitor}}, time.time() - t0)
    print(f"  {name}: val AUROC {auroc(yv, pv):.6f} (monitor={monitor})")


# ---- Macenko stain normalization (Macenko et al. 2009; period-authentic) ----
_HEREF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
_MAXCREF = np.array([1.9705, 1.0308])


def macenko_norm(I, Io=240, alpha=1, beta=0.15):
    """Normalize an HxWx3 uint8 RGB patch to the reference H&E stain. Passthrough on
    patches with too little tissue (mostly-background) to avoid unstable SVD."""
    h, w, _ = I.shape
    X = I.reshape(-1, 3).astype(np.float64)
    OD = -np.log((X + 1.0) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]
    if ODhat.shape[0] < 20:
        return I.astype(np.uint8)
    try:
        _, V = np.linalg.eigh(np.cov(ODhat.T))
        proj = ODhat.dot(V[:, 1:3])
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        mn, mx = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
        vmin = V[:, 1:3].dot(np.array([np.cos(mn), np.sin(mn)]))
        vmax = V[:, 1:3].dot(np.array([np.cos(mx), np.sin(mx)]))
        HE = np.array([vmin, vmax]).T if vmin[0] > vmax[0] else np.array([vmax, vmin]).T
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        maxC = np.array([np.percentile(C[0], 99), np.percentile(C[1], 99)])
        maxC[maxC == 0] = 1.0
        C = C / maxC[:, None] * _MAXCREF[:, None]
        Inorm = Io * np.exp(-_HEREF.dot(C))
        return np.clip(Inorm, 0, 255).T.reshape(h, w, 3).astype(np.uint8)
    except np.linalg.LinAlgError:
        return I.astype(np.uint8)


def _macenko_batch(arr):
    return np.stack([macenko_norm(a) for a in arr.numpy().astype(np.uint8)]).astype(np.uint8)


def _macenko_datasets(cfg):
    """Train/val tf.data with Macenko applied inside the decode (cached if cfg.data.cache)."""
    from src import data as D
    df = D.load_labels(cfg)
    tr_df, va_df = D.split_train_val(cfg, df)
    train_dir, _, _ = D._resolve(cfg)
    size = cfg["data"]["image_size"]; ext = cfg["data"]["image_ext"]
    bs = cfg["train"]["batch_size"]; cache = cfg["data"].get("cache", False)

    def build(frame, training):
        paths = [str(train_dir / f"{i}{ext}") for i in frame["id"]]
        labels = frame["label"].astype("float32").tolist()

        def read(path, y):
            def _f(p):
                return macenko_norm(D._decode_tif(p, size))
            img = tf.py_function(_f, [path], tf.uint8); img.set_shape([size, size, 3])
            return img, y
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(read, num_parallel_calls=tf.data.AUTOTUNE)
        if cache:
            ds = ds.cache()
        if training:
            ds = ds.shuffle(min(len(paths), 20000), seed=cfg["seed"], reshuffle_each_iteration=True)
        return ds.batch(bs).prefetch(tf.data.AUTOTUNE)
    return build(tr_df, True), build(va_df, False)


def handle_train_macenko(job, cfg, ids_v, yv, Xv, ids_t, Xt):
    """Macenko-stain-normalized TinyVGG member (Keras 3). Decorrelated preprocessing:
    normalize every patch to a reference H&E stain, then train from scratch on val_loss."""
    import src.model as M
    hp = job.get("hyperparams", {}); name = job["member_name"]
    cfg = json.loads(json.dumps(cfg))
    cfg["model"]["backbone"] = "TinyVGG"; cfg["model"]["from_scratch"] = True
    monitor = "val_" + hp.get("monitor", "loss").replace("val_", ""); mode = hp.get("mode", "min")
    t0 = time.time()
    train_ds, val_ds = _macenko_datasets(cfg)
    net = M.build_model(cfg); M.compile_model(net, cfg["train"].get("lr_head", 1e-3), cfg)
    out_ckpt = f"/content/{name}.keras"
    net.fit(train_ds, validation_data=val_ds, epochs=int(hp.get("epochs_cap", 30)),
            callbacks=[keras.callbacks.ModelCheckpoint(out_ckpt, monitor=monitor, mode=mode,
                                                       save_best_only=True, verbose=1),
                       keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=8,
                                                     restore_best_weights=True)], verbose=2)
    net = tf.keras.models.load_model(out_ckpt, compile=False)
    # Inference: Macenko-normalize the val + test arrays, then TTA-predict.
    print("  macenko-normalizing val+test for inference ...")
    Xvn = np.stack([macenko_norm(a) for a in Xv]); Xtn = np.stack([macenko_norm(a) for a in Xt])
    tta = bool(hp.get("tta", True))
    pv = predict(net, Xvn, tta); pt = predict(net, Xtn, tta)
    emit_member(name, ids_v, yv, pv, ids_t, pt)
    write_manifest(job, [name], {name: {"val_auroc": round(auroc(yv, pv), 6), "monitor": monitor,
                   "preproc": "macenko"}}, time.time() - t0)
    print(f"  {name}: val AUROC {auroc(yv, pv):.6f} (macenko, monitor={monitor})")


def handle_needs_human(job):
    """Arch paths not yet wired in THIS (Keras-3) loop -> flag for a human / legacy notebook."""
    write_manifest(job, [], {}, 0.0, status="needs_human",
                   extra={"reason": f"handler for arch '{job.get('arch')}' not implemented here"})
    print("  needs_human:", job["jobid"], job.get("arch"))


# ----------------------------------------------------------------- poll loop


def already_done(jobid):
    """Skip if this job's manifest already exists in colab-out (idempotent resume)."""
    sh(["kaggle", "datasets", "download", "-d", OUT_DS, "-p", JOBSDIR, "--unzip", "--force"])
    return os.path.exists(os.path.join(JOBSDIR, f"job_{jobid}.json"))


def _p4m_owned(job):
    """p4m-family jobs are handled by the SEPARATE legacy-Keras-2 loop
    (notebooks/p4m_autoloop_colab.py). This Keras-3 loop must neither run them (it can't:
    keras-gcnn needs Keras 2) NOR needs_human-flag them — a manifest here would make BOTH
    loops' already_done() skip the job, blocking the p4m loop. So we drop them from selection
    entirely and leave them pending for the p4m loop."""
    arch = job.get("arch") or ""
    return (job.get("type") == "train" and arch == "p4m_tinyvgg") \
        or (job.get("type") == "features" and "p4m" in arch)


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
                   if j.get("status") == "pending" and not _p4m_owned(j)]
        job = next((j for j in pending if not already_done(j["jobid"])), None)
        if not job:
            print("no runnable pending job; idle."); time.sleep(POLL_SECONDS); continue

        shutil.rmtree(OUTDIR, ignore_errors=True); os.makedirs(OUTDIR, exist_ok=True)
        print(f"\n=== running {job['jobid']} ({job.get('type')}/{job.get('arch')}) ===")
        try:
            if job.get("type") == "oof_dump":
                handle_oof_dump(job, cfg, ids_v, yv, Xv, ids_t, Xt)
            elif job.get("type") == "train" and \
                    job.get("hyperparams", {}).get("preproc", "").startswith("macenko"):
                handle_train_macenko(job, cfg, ids_v, yv, Xv, ids_t, Xt)
            elif job.get("type") == "train" and job.get("arch") in ("TinyVGG", "MobileNetV3Small"):
                handle_train_keras(job, cfg, ids_v, yv, Xv, ids_t, Xt)
            else:
                # p4m-family jobs are already filtered out (_p4m_owned) and run in the legacy
                # p4m loop; anything reaching here is a genuinely unhandled arch (e.g. e2cnn).
                handle_needs_human(job)
            push_out(f"[autoloop] {job['jobid']} done")
        except Exception:
            traceback.print_exc()
            os.makedirs(OUTDIR, exist_ok=True)
            write_manifest(job, [], {}, 0.0, status="error",
                           extra={"trace": traceback.format_exc()[-1500:]})
            push_out(f"[autoloop] {job['jobid']} ERROR")
        time.sleep(POLL_SECONDS)

    print("session deadline reached; stopping poll loop (restart the tab to resume).")


if __name__ == "__main__":
    main()
