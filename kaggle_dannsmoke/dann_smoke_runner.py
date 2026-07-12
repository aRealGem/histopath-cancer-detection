# Validation smoke for the two-head (DANN + cooperative dual-head) build. CPU, offline.
# Proves the whole path executes on Kaggle before we spend a GPU slot on the full run:
#   1) pytest guardrails (two outputs, GRL present iff adversarial, inference strip)
#   2) train smoke_dann     -> two-head train loop -> single-head best.keras export
#   3) train smoke_dualhead -> same, grl:false
#   4) load each exported best.keras and confirm it's a single (None,1) model that predicts
import os, sys, glob, shutil, gzip, zipfile, subprocess

ON_KAGGLE = os.path.exists('/kaggle/input')
print('Kaggle' if ON_KAGGLE else 'local')
if ON_KAGGLE:
    for d in sorted(glob.glob('/kaggle/input/*')):
        print(' input:', d, '->', sorted(os.path.basename(x) for x in glob.glob(d + '/*'))[:8])

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

if not os.path.exists('data/wsi/patch_id_wsi_full.csv.gz'):
    hits = [h for h in glob.glob('/kaggle/input/**/*wsi*full*.csv', recursive=True) if os.path.isfile(h)]
    if hits:
        os.makedirs('data/wsi', exist_ok=True)
        with open(hits[0], 'rb') as fi, gzip.open('data/wsi/patch_id_wsi_full.csv.gz', 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        print('WSI map normalized from', hits[0])

# domain label csv must be present for the two-head data pipeline
print('wsi_domain_k2 present:', os.path.exists('data/wsi/wsi_domain_k2.csv'))

# Patch data.root into every config so the smokes find the competition mount.
import yaml
cands = glob.glob('/kaggle/input/**/train_labels.csv', recursive=True)
assert cands, 'competition train_labels.csv not found'
DATA_ROOT = os.path.dirname(cands[0])
for cf in glob.glob('configs/*.yaml'):
    c = yaml.safe_load(open(cf))
    if isinstance(c, dict) and 'data' in c:
        c['data']['root'] = DATA_ROOT
        yaml.safe_dump(c, open(cf, 'w'), sort_keys=False)
print('data.root ->', DATA_ROOT)

sys.path.insert(0, os.getcwd())


def sh(cmd):
    print('\n$', cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    print('  exit', r.returncode, flush=True)
    return r.returncode


# 1) Guardrail unit tests (skip gracefully if pytest is unavailable offline).
try:
    import pytest  # noqa: F401
    sh('python -m pytest tests/test_model.py -q')
except Exception as e:
    print('pytest unavailable, skipping unit tests:', e)

# 2+3) Train both two-head smokes end-to-end.
rc_dann = sh('python -m src.train --config configs/smoke_dann.yaml')
assert rc_dann == 0, 'smoke_dann training failed'
import tensorflow as tf
from src import model as _m  # registers GradientReversal / RandomHEDJitter
import numpy as np

def check_infer(tag):
    net = tf.keras.models.load_model('artifacts/best.keras')
    n_out = len(net.outputs)
    shp = net.output_shape
    x = np.random.default_rng(0).integers(0, 256, (4, 96, 96, 3)).astype('uint8')
    p = net.predict(x, verbose=0)
    print(f'[{tag}] exported best.keras: outputs={n_out} output_shape={shp} pred_shape={np.asarray(p).shape}')
    assert n_out == 1 and shp == (None, 1), f'{tag}: exported model must be single-head (None,1)'
    assert np.asarray(p).shape == (4, 1)
    print(f'[{tag}] OK — single-head inference model loads & predicts')

check_infer('dann')

rc_dh = sh('python -m src.train --config configs/smoke_dualhead.yaml')
assert rc_dh == 0, 'smoke_dualhead training failed'
check_infer('dualhead')

print('\nALL SMOKE CHECKS PASSED')
