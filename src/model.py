"""MobileNetV3 transfer-learning classifier.

Notes
-----
* ``include_preprocessing=True`` bakes the [0,255] -> normalized rescaling into the
  backbone, so feed raw uint8/float patches. Do NOT rescale upstream or you double
  normalize and tank AUROC.
* Augmentation layers are part of the model; they are no-ops at inference
  (``training=False``), so evaluate.py / predict.py stay consistent with train.py.
* Final Dense uses float32 (``dtype="float32"``) so mixed-precision training keeps
  a numerically stable sigmoid.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

# All of these take include_preprocessing=True and the SAME raw-[0,255] input
# contract (they normalize once, internally). That uniform contract is what lets
# build_model stay backbone-agnostic and is asserted by tests/test_model.py so a
# new backbone can't silently break the no-double-normalize rule.
_BACKBONES = {
    "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
    "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
    "EfficientNetV2B1": tf.keras.applications.EfficientNetV2B1,
    "EfficientNetV2S": tf.keras.applications.EfficientNetV2S,
}


@tf.keras.utils.register_keras_serializable(package="histopath")
class RandomHEDJitter(layers.Layer):
    """H&E stain-color augmentation via HED color deconvolution (Tellez et al. 2019).

    Decomposes each patch into Hematoxylin/Eosin/residual stain concentrations
    (Ruifrok–Johnston OD basis), perturbs them per-image (c -> c*alpha + beta,
    alpha~U(1-s,1+s), beta~U(-s,s)), and recomposes. Teaches invariance to the
    staining/scanner color variation that drives the train->private-test gap.

    Operates on float32 patches in [0,255] and returns the same range (so the
    backbone's include_preprocessing still does the one-and-only normalization).
    Identity at inference (training=False), like the other augmentation layers.
    Registered as a serializable layer so evaluate.py/predict.py can load_model.
    """

    # Rows = OD-RGB vectors of [Hematoxylin, Eosin, residual].
    _STAIN = [[0.65, 0.70, 0.29],
              [0.07, 0.99, 0.11],
              [0.27, 0.57, 0.78]]

    def __init__(self, sigma: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.sigma = float(sigma)

    def build(self, input_shape):
        S = tf.constant(self._STAIN, dtype=tf.float32)   # [stain, rgb]
        self._S = S
        self._D = tf.linalg.inv(S)                        # deconvolution [rgb, stain]
        super().build(input_shape)

    def call(self, inputs, training=None):
        if not training or self.sigma <= 0.0:
            return inputs
        x = tf.cast(inputs, tf.float32)
        I0 = 256.0
        od = -tf.math.log((x + 1.0) / I0)                          # optical density
        conc = tf.einsum("bhwk,ki->bhwi", od, self._D)             # stain concentrations
        b = tf.shape(x)[0]
        alpha = tf.random.uniform((b, 1, 1, 3), 1.0 - self.sigma, 1.0 + self.sigma)
        beta = tf.random.uniform((b, 1, 1, 3), -self.sigma, self.sigma)
        conc = conc * alpha + beta
        od2 = tf.einsum("bhwi,ik->bhwk", conc, self._S)
        x2 = tf.exp(-od2) * I0 - 1.0
        return tf.clip_by_value(x2, 0.0, 255.0)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(sigma=self.sigma)
        return cfg


def _augmenter(cfg: dict) -> tf.keras.Sequential:
    a = cfg["augment"]
    steps: list[layers.Layer] = []
    if a.get("stain_jitter"):
        # Color augmentation first, on the raw [0,255] float image.
        steps.append(RandomHEDJitter(a["stain_jitter"]))
    flips = []
    if a.get("horizontal_flip"):
        flips.append("horizontal")
    if a.get("vertical_flip"):
        flips.append("vertical")
    if flips:
        steps.append(layers.RandomFlip("_and_".join(flips)))
    if a.get("rotation_factor"):
        steps.append(layers.RandomRotation(a["rotation_factor"], fill_mode="reflect"))
    if a.get("contrast_factor"):
        steps.append(layers.RandomContrast(a["contrast_factor"]))
    if a.get("zoom_factor"):
        steps.append(layers.RandomZoom(a["zoom_factor"], fill_mode="reflect"))
    if a.get("brightness_factor"):
        # value_range matches our raw [0,255] pipeline (no double-normalize).
        steps.append(layers.RandomBrightness(a["brightness_factor"], value_range=(0.0, 255.0)))
    return tf.keras.Sequential(steps, name="augment")


def build_model(cfg: dict) -> tf.keras.Model:
    size = cfg["data"]["image_size"]
    name = cfg["model"]["backbone"]
    if name not in _BACKBONES:
        raise ValueError(f"Unknown backbone {name}; choose from {list(_BACKBONES)}")

    backbone = _BACKBONES[name](
        input_shape=(size, size, 3),
        include_top=False,
        weights=cfg["model"]["weights"],
        include_preprocessing=True,
    )
    # Transfer mode: freeze the backbone for phase 1. From-scratch mode (random
    # init): the backbone must be trainable from the start (there's nothing useful
    # to keep frozen), and train.py runs a single end-to-end phase.
    from_scratch = bool(cfg["train"].get("from_scratch", False))
    backbone.trainable = from_scratch

    inputs = tf.keras.Input((size, size, 3), dtype="uint8", name="image")
    # Cast uint8 -> float32 with an identity Rescaling(scale=1.0). Keras 3 forbids
    # a raw tf.cast on a symbolic KerasTensor, and Rescaling is a proper layer that
    # round-trips through model.save/load. scale=1.0, offset=0 changes NO pixel
    # values — this is NOT a normalization (the backbone's include_preprocessing
    # still does the one-and-only [0,255] normalization; no double-normalize).
    x = layers.Rescaling(1.0, name="to_float")(inputs)
    x = _augmenter(cfg)(x)
    # Transfer: keep the backbone (incl. BatchNorm) in inference mode. From-scratch:
    # let it follow the outer training flag so BatchNorm learns its own statistics.
    x = backbone(x) if from_scratch else backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg["train"]["dropout"], name="drop")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="tumor_prob")(x)

    return tf.keras.Model(inputs, outputs, name=f"{name}_pcam")


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """The Keras application is nested as a single sub-Model layer. Locate it by
    type rather than by name (application layer names are version-fragile)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone Model found for fine-tuning.")


def compile_model(model: tf.keras.Model, lr: float, cfg: dict) -> None:
    loss = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=cfg["train"].get("label_smoothing", 0.0)
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=loss,
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.BinaryAccuracy(name="acc")],
    )


def unfreeze_top(model: tf.keras.Model, n_layers: int) -> None:
    """Unfreeze the last ``n_layers`` of the backbone for fine-tuning; keep
    BatchNorm layers frozen (running stats should not shift on a small set)."""
    backbone = _find_backbone(model)
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
