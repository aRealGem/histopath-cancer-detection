# P4M-DenseNet — group-equivariant (D4) DenseNet for PCam. COLAB / A100 ONLY.
#
# The paper's (Veeling 2018) actual best architecture: a DenseNet built from p4m
# group-convs. DenseNet = each conv layer receives the CONCATENATED feature maps of
# ALL preceding layers in its block (dense connectivity / feature reuse) -> more
# parameter-efficient than the plain VGG-style p4m we ran (which got solo 0.9366).
#
# Same Colab setup as notebooks/p4m_tinyvgg_colab.py (legacy Keras + keras-gcnn +
# shims + the Layer.add_update monkeypatch). This file is just the model + the
# regularized-short training recipe that beat the cosine one.
#
# NOTE: group-conv + Concatenate is the one risky bit — if keras-gcnn's channel
# layout doesn't survive concatenation, the first GConv after a dense block will
# error; if so, paste the traceback and we adjust (likely a channel-order fix).

from tensorflow.keras import layers, Model, Input
from keras_gcnn.layers import GConv2D, GBatchNorm, GroupPool


def _g_bottleneck(x, growth):
    """BN-ReLU-GConv producing `growth` new group-feature-maps (pre-activation, DenseNet-style)."""
    y = GBatchNorm(h='D4')(x)
    y = layers.Activation('relu')(y)
    y = GConv2D(growth, 3, h_input='D4', h_output='D4', padding='same', use_bias=False)(y)
    return y


def _dense_block(x, n_layers, growth):
    for _ in range(n_layers):
        y = _g_bottleneck(x, growth)
        x = layers.Concatenate(axis=-1)([x, y])   # dense connectivity (feature reuse)
    return x


def _transition(x, compress):
    """Compress channels (1x1 group-conv) + spatial downsample between dense blocks."""
    y = GBatchNorm(h='D4')(x)
    y = layers.Activation('relu')(y)
    y = GConv2D(compress, 1, h_input='D4', h_output='D4', padding='same', use_bias=False)(y)
    y = layers.AveragePooling2D(2)(y)
    return y


def p4m_densenet(growth=8, blocks=(4, 4, 4), init_ch=10, compress=16):
    """Lean p4m DenseNet. blocks=(4,4,4) growth=8 -> ~similar param budget to the
    120K-param paper net once the x8 group weight-sharing is counted. Adjust growth/
    blocks if it's too big/small (check model.count_params())."""
    inp = Input((96, 96, 3), dtype='float32')
    x = layers.Rescaling(1 / 255.)(inp)
    x = layers.RandomContrast(0.1)(x)                                   # only aug; p4m handles geometry
    x = GConv2D(init_ch, 3, h_input='Z2', h_output='D4', padding='same', use_bias=False)(x)  # lift Z2->D4
    for i, nl in enumerate(blocks):
        x = _dense_block(x, nl, growth)
        if i < len(blocks) - 1:
            x = _transition(x, compress)                               # 96 -> 48 -> 24
    x = GBatchNorm(h='D4')(x)
    x = layers.Activation('relu')(x)
    x = GroupPool(h_input='D4')(x)                                     # D4-invariance
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inp, out, name='p4m_densenet')
