# Full training run for a two-head config (GPU). Trains, then TTA on the test set to
# produce a blendable submission. CONFIG is set per-kernel.
import os, sys, glob, shutil, gzip, zipfile, subprocess

CONFIG = 'configs/exp_dann.yaml'   # <-- per-kernel

ON_KAGGLE = os.path.exists('/kaggle/input')
print('Kaggle' if ON_KAGGLE else 'local', '| config:', CONFIG)

REPO_URL = 'https://github.com/aRealGem/histopath-cancer-detection'
WORK = '/kaggle/working/repo'
if not os.path.exists('src/train.py'):
    os.makedirs(WORK, exist_ok=True)
    czip = glob.glob('/kaggle/input/**/code.zip', recursive=True)
    extracted = glob.glob('/kaggle/input/**/src/train.py', recursive=True)
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

# WSI map gzip
if not os.path.exists('data/wsi/patch_id_wsi_full.csv.gz'):
    hits = [h for h in glob.glob('/kaggle/input/**/*wsi*full*.csv', recursive=True) if os.path.isfile(h)]
    if hits:
        os.makedirs('data/wsi', exist_ok=True)
        with open(hits[0], 'rb') as fi, gzip.open('data/wsi/patch_id_wsi_full.csv.gz', 'wb') as fo:
            shutil.copyfileobj(fi, fo)
        print('WSI map normalized from', hits[0])
print('wsi_domain_k2 present:', os.path.exists('data/wsi/wsi_domain_k2.csv'))

# Offline ImageNet weights for weights:imagenet (no internet in kernel).
kmodels = os.path.expanduser('~/.keras/models')
os.makedirs(kmodels, exist_ok=True)
for wts in glob.glob('/kaggle/input/**/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5', recursive=True):
    shutil.copy(wts, os.path.join(kmodels, os.path.basename(wts)))
    print('cached offline backbone weights:', os.path.basename(wts))

# Patch data.root into all configs.
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
    assert r.returncode == 0, cmd
    return r.returncode

# Train (two-head -> single-head best.keras export) then TTA-on-test submission.
sh(f'python -m src.train --config {CONFIG}')
sh(f'python -m src.tta_eval --config {CONFIG} --model artifacts/best.keras --test')

# Surface the produced artifacts.
for f in glob.glob('artifacts/*'):
    print('artifact:', f, os.path.getsize(f))
print('TRAINING RUN DONE')
