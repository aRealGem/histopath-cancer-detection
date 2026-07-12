# Val-set prediction dump for decorrelation / blend analysis (CPU/GPU inference, no training).
#
# For each trained checkpoint (champion baseline, from-scratch, heavy-color-jitter) we
# compute per-patch predictions on the SAME seed-1337 WSI-grouped validation split (held
# out from ALL of them, since every config shares seed/val_fraction/wsi_map). We dump
# {id,label,p_notta,p_tta} per model so the decorrelation + blend-weight search can run
# offline on the Pi. We also print single-model and equal-weight-blend val AUROCs straight
# into the log, so the headline signal ("does blending a decorrelated member help?") is
# visible without pulling the CSVs.
#
# Self-contained: the two custom layers are redefined + registered here, so load_model
# works regardless of how stale the staged code.zip is.
import os, sys, glob, shutil, gzip, zipfile, subprocess, itertools

ON_KAGGLE = os.path.exists('/kaggle/input')
print('Kaggle' if ON_KAGGLE else 'local')
if ON_KAGGLE:
    for d in sorted(glob.glob('/kaggle/input/*')):
        print(' input:', d, '->', sorted(os.path.basename(x) for x in glob.glob(d + '/*'))[:8])

# --- Stage src/ + configs + WSI map from the attached code dataset (no internet). ---
REPO_URL = 'https://github.com/aRealGem/histopath-cancer-detection'
WORK = '/kaggle/working/repo'
if not os.path.exists('src/data.py'):
    os.makedirs(WORK, exist_ok=True)
    czip = glob.glob('/kaggle/input/**/code.zip', recursive=True)
    extracted = glob.glob('/kaggle/input/**/src/data.py', recursive=True)
    if czip:
        with zipfile.ZipFile(czip[0]) as z:
            z.extractall(WORK)
        print('staged from code.zip:', czip[0])
    elif extracted:
        root = os.path.dirname(os.path.dirname(extracted[0]))
        shutil.copytree(root, WORK, dirs_exist_ok=True)
        print('staged from extracted dataset:', root)
    else:
        subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, WORK], check=True)
    os.chdir(WORK)

# WSI map: re-materialize the gzip if Kaggle decompressed it into a folder.
if not os.path.exists('data/wsi/patch_id_wsi_full.csv.gz'):
    hits = [h for h in glob.glob('/kaggle/input/**/*wsi*full*.csv', recursive=True) if os.path.isfile(h)]
    if hits:
        os.makedirs('data/wsi', exist_ok=True)
        with open(hits[0], 'rb') as fi, gzip.open('data/wsi/patch_id_wsi_full.csv.gz', 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        print('WSI map normalized from', hits[0])

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score


# --- Custom layers redefined inline so load_model deserializes any of our checkpoints
#     without depending on the staged code.zip version. ---
@tf.keras.utils.register_keras_serializable(package="histopath")
class GradientReversal(layers.Layer):
    def __init__(self, lamb: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lamb = float(lamb)

    def call(self, inputs):
        lamb = self.lamb

        @tf.custom_gradient
        def _reverse(x):
            def grad(dy):
                return -lamb * dy
            return tf.identity(x), grad

        return _reverse(inputs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(lamb=self.lamb)
        return cfg


@tf.keras.utils.register_keras_serializable(package="histopath")
class RandomHEDJitter(layers.Layer):
    _STAIN = [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]

    def __init__(self, sigma: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def build(self, input_shape):
        S = tf.constant(self._STAIN, dtype=tf.float32)
        self._S = S
        self._D = tf.linalg.inv(S)
        super().build(input_shape)

    def call(self, inputs, training=None):
        if not training or self.sigma <= 0.0:
            return inputs
        x = tf.cast(inputs, tf.float32)
        I0 = 256.0
        od = -tf.math.log((x + 1.0) / I0)
        conc = tf.einsum("bhwk,ki->bhwi", od, self._D)
        b = tf.shape(x)[0]
        alpha = tf.random.uniform((b, 1, 1, 3), 1.0 - self.sigma, 1.0 + self.sigma)
        beta = tf.random.uniform((b, 1, 1, 3), -self.sigma, self.sigma)
        conc = conc * alpha + beta
        od2 = tf.einsum("bhwi,ik->bhwk", conc, self._S)
        return tf.clip_by_value(tf.exp(-od2) * I0 - 1.0, 0.0, 255.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sigma=self.sigma)
        return cfg


from src import data as D

# --- Fixed val split (seed 1337, WSI-grouped) — identical for every model. ---
cfg = yaml.safe_load(open('configs/baseline.yaml'))
cands = glob.glob('/kaggle/input/**/train_labels.csv', recursive=True)
assert cands, 'competition train_labels.csv not found under /kaggle/input'
cfg['data']['root'] = os.path.dirname(cands[0])
cfg.setdefault('seed', 1337)
print('data.root ->', cfg['data']['root'])

df = D.load_labels(cfg)
_, val_df = D.split_train_val(cfg, df)
val_df = val_df.reset_index(drop=True)
ids = val_df['id'].tolist()

# Val ds has no shuffle (training=False), so batch order == val_df order.
_, val_ds = D.make_train_val_datasets(cfg, df.iloc[:1], val_df)
Xs, Ys = [], []
for xb, yb in val_ds:
    Xs.append(xb.numpy()); Ys.append(yb.numpy())
Xv = np.concatenate(Xs)
yv = np.concatenate(Ys).astype(int)
assert len(yv) == len(ids), (len(yv), len(ids))
assert np.array_equal(yv, val_df['label'].values.astype(int)), 'val order mismatch!'
print('val decoded:', Xv.shape, 'pos-rate=%.3f' % yv.mean())


def views(x):
    return [
        x, x[:, :, ::-1, :], x[:, ::-1, :, :], x[:, ::-1, ::-1, :],
        np.rot90(x, 1, axes=(1, 2)), np.rot90(x, 2, axes=(1, 2)),
        np.rot90(x, 3, axes=(1, 2)), np.rot90(x[:, :, ::-1, :], 1, axes=(1, 2)),
    ]


def tta(net, X):
    acc = np.zeros(len(X), np.float64)
    for v in views(X):
        acc += net.predict(v, verbose=0, batch_size=512).ravel()
    return (acc / 8.0).astype(np.float32)


MODELS = {
    'champion': 'histopath-mobilenetv3-fulltrain',
    'scratch':  'histopath-scratch-full',
    'cjitter':  'histopath-cjitter',
}
os.makedirs('/kaggle/working/valpred', exist_ok=True)
# Mount layout varies (flat /kaggle/input/<slug>/ vs nested /kaggle/input/notebooks/
# <user>/<slug>/), so glob ALL checkpoints and match by slug substring.
all_ckpts = glob.glob('/kaggle/input/**/best.keras', recursive=True)
print('checkpoints found under /kaggle/input:')
for c in all_ckpts:
    print('  ', c)
tta_probs = {}
for name, slug in MODELS.items():
    hits = [h for h in all_ckpts if slug in h]
    if not hits:
        print('!! MISSING checkpoint for', name, '(', slug, ') — skipping')
        continue
    print(name, '->', hits[0])
    net = tf.keras.models.load_model(hits[0], compile=False)
    p0 = net.predict(Xv, verbose=0, batch_size=512).ravel()
    pt = tta(net, Xv)
    print(f'  {name}: val no-TTA={roc_auc_score(yv, p0):.4f}  TTA={roc_auc_score(yv, pt):.4f}')
    tta_probs[name] = pt
    pd.DataFrame({'id': ids, 'label': yv, 'p_notta': p0, 'p_tta': pt}).to_csv(
        f'/kaggle/working/valpred/valpred_{name}.csv', index=False)

# --- Immediate headline signal: does blending a decorrelated member help on val? ---
names = list(tta_probs)
print('\n=== single-model val AUROC (TTA) ===')
for n in names:
    print(f'  {n:10s} {roc_auc_score(yv, tta_probs[n]):.4f}')
print('=== equal-weight blend val AUROC (TTA) ===')
for r in range(2, len(names) + 1):
    for combo in itertools.combinations(names, r):
        blend = np.mean([tta_probs[c] for c in combo], axis=0)
        print(f'  {"+".join(combo):26s} {roc_auc_score(yv, blend):.4f}')
print('DONE')
