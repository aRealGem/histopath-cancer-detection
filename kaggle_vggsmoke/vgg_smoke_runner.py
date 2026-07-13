# Validation smoke for TinyVGG: pytest guardrails (now cover TinyVGG via the
# parametrized _BACKBONES tests) + a 2-epoch from-scratch train + load/predict check.
import os, sys, glob, shutil, gzip, zipfile, subprocess
ON_KAGGLE = os.path.exists('/kaggle/input')
WORK='/kaggle/working/repo'
if not os.path.exists('src/train.py'):
    os.makedirs(WORK, exist_ok=True)
    czip=glob.glob('/kaggle/input/**/code.zip', recursive=True)
    ext=glob.glob('/kaggle/input/**/src/train.py', recursive=True)
    if czip:
        zipfile.ZipFile(czip[0]).extractall(WORK); print('staged code.zip', czip[0])
    elif ext:
        root=os.path.dirname(os.path.dirname(ext[0])); shutil.copytree(root, WORK, dirs_exist_ok=True)
    os.chdir(WORK)
if not os.path.exists('data/wsi/patch_id_wsi_full.csv.gz'):
    h=[x for x in glob.glob('/kaggle/input/**/*wsi*full*.csv', recursive=True) if os.path.isfile(x)]
    if h:
        os.makedirs('data/wsi', exist_ok=True)
        with open(h[0],'rb') as fi, gzip.open('data/wsi/patch_id_wsi_full.csv.gz','wb') as fo: shutil.copyfileobj(fi,fo)
import yaml
cands=glob.glob('/kaggle/input/**/train_labels.csv', recursive=True); assert cands
for cf in glob.glob('configs/*.yaml'):
    c=yaml.safe_load(open(cf))
    if isinstance(c,dict) and 'data' in c: c['data']['root']=os.path.dirname(cands[0]); yaml.safe_dump(c,open(cf,'w'),sort_keys=False)
sys.path.insert(0, os.getcwd())
def sh(c):
    print('\n$',c,flush=True); r=subprocess.run(c,shell=True); print(' exit',r.returncode,flush=True); return r.returncode
try:
    import pytest; sh('python -m pytest tests/test_model.py -q')
except Exception as e: print('pytest unavailable:',e)
assert sh('python -m src.train --config configs/smoke_vgg.yaml')==0, 'TinyVGG smoke train failed'
import tensorflow as tf, numpy as np
from src import model as _m
net=tf.keras.models.load_model('artifacts/best.keras')
x=np.random.default_rng(0).integers(0,256,(4,96,96,3)).astype('uint8')
p=net.predict(x,verbose=0)
print('TinyVGG params:', net.count_params(), 'output_shape', net.output_shape, 'pred', np.asarray(p).shape)
assert net.output_shape==(None,1) and np.asarray(p).shape==(4,1)
print('VGG SMOKE PASSED')
