# p4m-TinyVGG — group-equivariant (D4/p4m) small CNN for PCam. COLAB / A100 ONLY.
#
# This is NOT part of the Keras-3 src/ pipeline: it needs the 2018 keras-gcnn library
# (basveeling's own PCam-paper code, via neel-dey's TF2 port), which requires the
# Keras-2 API. Run on Colab with TF_USE_LEGACY_KERAS=1. Reproduces Veeling et al.
# 2018 "Rotation Equivariant CNNs for Digital Pathology" on our data.
#
# RESULT (2026-07-14): solo private 0.9240 vs augmentation-based TinyVGG 0.8383 at
# IDENTICAL val (0.9787) -> built-in equivariance GENERALIZES where aug MIRAGES.
# 3-way champion+TinyVGG+p4m (equal) = private 0.9411 (best).
#
# Cells are marked; paste into a fresh Colab notebook top-to-bottom.

# ===== Cell 1 — setup (A100): legacy Keras + shim + libs + Kaggle + data + repo =====
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# !pip -q install tf-keras
# !git clone -q https://github.com/neel-dey/tf2-GrouPy.git
# !git clone -q https://github.com/neel-dey/tf2-keras-gcnn.git
import sys
sys.path.insert(0, '/content/tf2-GrouPy'); sys.path.insert(0, '/content/tf2-keras-gcnn')
import numpy as np
for n, t in [('int', int), ('float', float), ('bool', bool)]:   # deprecated aliases the 2018 lib needs
    if not hasattr(np, n): setattr(np, n, t)
# Kaggle new-format access-token auth + data + repo (see Colab notebook for the !shell lines)


# ===== Cell 2 — build p4m-TinyVGG =====
from tensorflow.keras import layers, Model, Input
from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool


def gblock(x, w, hin, hout):
    x = GConv2D(w, 3, h_input=hin, h_output=hout, padding='same', use_bias=False)(x)
    x = GBatchNorm(h=hout)(x)
    x = layers.Activation('relu')(x)
    return x


def p4m_tinyvgg(widths=(8, 16, 32)):
    """Lean p4m (D4 = 4 rotations x 2 mirrors) equivariant VGG. ~217K params.
    Lift Z2->D4, p4m convs on a fine-detail stride-1 stem, GroupPool over the 8
    orientations -> D4-invariant features. NO geometric aug + NO D4-TTA needed
    (both are made redundant by the built-in equivariance). Only mild contrast aug."""
    inp = Input((96, 96, 3), dtype='float32')
    x = layers.Rescaling(1 / 255.)(inp)
    x = layers.RandomContrast(0.1)(x)                       # only aug; geometry handled by p4m
    x = gblock(x, widths[0], 'Z2', 'D4')                    # lift to p4m
    x = gblock(x, widths[0], 'D4', 'D4'); x = layers.MaxPool2D(2)(x)   # 48
    x = gblock(x, widths[1], 'D4', 'D4'); x = gblock(x, widths[1], 'D4', 'D4'); x = layers.MaxPool2D(2)(x)  # 24
    x = gblock(x, widths[2], 'D4', 'D4'); x = gblock(x, widths[2], 'D4', 'D4'); x = layers.MaxPool2D(2)(x)  # 12
    x = gblock(x, widths[2], 'D4', 'D4')
    x = GroupPool(h_input='D4')(x)                          # invariance: pool over the 8 orientations
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inp, out, name='p4m_tinyvgg')


# ===== Cell 3 — STABILIZED training (cosine LR + warmup + clipnorm + label smoothing) =====
# The original 1e-3 constant LR went unstable (val_loss spiked ep4-6, early-stopped @ep3).
# Warmup-from-tiny + cosine decay + gradient clipping + mild label smoothing fix that.
def train_p4m(m, cfg, train_ds, val_ds, tr, EPOCHS=35):
    import math
    from tensorflow import keras
    os.makedirs('artifacts', exist_ok=True)
    steps_per_epoch = math.ceil(len(tr) / cfg['train']['batch_size'])
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-5,                 # start tiny
        warmup_target=5e-4,                         # ramp up to this (lower than the unstable 1e-3)
        warmup_steps=3 * steps_per_epoch,           # 3-epoch warmup
        decay_steps=EPOCHS * steps_per_epoch,       # cosine down over the full run
        alpha=0.02)                                 # floor at 2% of peak
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=sched, clipnorm=1.0),
              loss=keras.losses.BinaryCrossentropy(label_smoothing=0.05),   # tame overconfident spikes
              metrics=[keras.metrics.AUC(name='auc')])
    return m.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('artifacts/best_p4m.keras', monitor='val_auc', mode='max', save_best_only=True)])
