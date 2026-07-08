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

_BACKBONES = {
    "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
    "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
}


def _augmenter(cfg: dict) -> tf.keras.Sequential:
    a = cfg["augment"]
    steps: list[layers.Layer] = []
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
    backbone.trainable = False  # phase 1

    inputs = tf.keras.Input((size, size, 3), dtype="uint8", name="image")
    # Cast uint8 -> float32 with an identity Rescaling(scale=1.0). Keras 3 forbids
    # a raw tf.cast on a symbolic KerasTensor, and Rescaling is a proper layer that
    # round-trips through model.save/load. scale=1.0, offset=0 changes NO pixel
    # values — this is NOT a normalization (the backbone's include_preprocessing
    # still does the one-and-only [0,255] normalization; no double-normalize).
    x = layers.Rescaling(1.0, name="to_float")(inputs)
    x = _augmenter(cfg)(x)
    x = backbone(x, training=False)
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
