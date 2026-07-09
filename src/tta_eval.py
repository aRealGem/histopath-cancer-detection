"""Test-time augmentation (TTA) evaluation on an already-trained checkpoint.

TTA is inference-only: it does NOT retrain. For each image we average the model's
predicted probability over a set of label-preserving views (identity, H/V flips,
90/180/270 rotations — valid because H&E patches are orientation-invariant). This
reduces variance from arbitrary patch orientation and typically lifts AUROC a little,
especially on out-of-distribution slides.

Efficiency: we decode the val (and optionally test) set ONCE into a uint8 array in
RAM, then the TTA views are free numpy flips/rotations — no per-view re-decode, no
TFRecords needed for a single inference pass.

Usage:
    python -m src.tta_eval --config configs/baseline.yaml --model artifacts/best.keras
    python -m src.tta_eval --config configs/baseline.yaml --model <path> --test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from src import data
from src import model as _model  # noqa: F401  (registers custom layers for load_model)
from src.utils import get_logger, load_config, set_seed

log = get_logger()


def _views(x: np.ndarray) -> list[np.ndarray]:
    """8 label-preserving views of a batch x=(N,H,W,3): identity, H-flip, V-flip,
    HV-flip, and 90/180/270 rotations."""
    return [
        x,
        x[:, :, ::-1, :],          # horizontal flip
        x[:, ::-1, :, :],          # vertical flip
        x[:, ::-1, ::-1, :],       # 180 via both flips
        np.rot90(x, 1, axes=(1, 2)),
        np.rot90(x, 2, axes=(1, 2)),
        np.rot90(x, 3, axes=(1, 2)),
        np.rot90(x[:, :, ::-1, :], 1, axes=(1, 2)),  # flip+rot for a bit more diversity
    ]


def _collect(ds) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for xb, yb in ds:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _tta_probs(net, X: np.ndarray) -> np.ndarray:
    """Mean predicted probability across all TTA views (views built lazily to cap RAM)."""
    acc = np.zeros(len(X), dtype=np.float64)
    views = _views(X)
    for v in views:
        acc += net.predict(v, verbose=0).ravel()
    return (acc / len(views)).astype(np.float32)


def main(config_path: str, model_path: str, do_test: bool) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s", model_path)
    net = tf.keras.models.load_model(model_path)

    # --- Validation: baseline vs TTA on the SAME WSI-grouped val split ---
    df = data.load_labels(cfg)
    _, val_df = data.split_train_val(cfg, df)
    _, val_ds = data.make_train_val_datasets(cfg, df.iloc[:1], val_df)
    Xv, yv = _collect(val_ds)
    log.info("Val decoded once into RAM: X=%s (%.0f MB)", Xv.shape, Xv.nbytes / 1e6)

    p_base = net.predict(Xv, verbose=0).ravel()
    auc_base = roc_auc_score(yv, p_base)
    p_tta = _tta_probs(net, Xv)
    auc_tta = roc_auc_score(yv, p_tta)
    log.info("VAL AUROC  no-TTA = %.4f  |  TTA(8 views) = %.4f  |  delta = %+.4f",
             auc_base, auc_tta, auc_tta - auc_base)

    # --- Optional: TTA submission over the full test set ---
    if do_test:
        test_ds, ids = data.make_test_dataset(cfg)  # yields (image, id_string)
        xs = [xb.numpy() for xb, _id in test_ds]
        Xt = np.concatenate(xs)
        log.info("Test decoded once into RAM: X=%s (%.0f MB)", Xt.shape, Xt.nbytes / 1e6)
        probs = _tta_probs(net, Xt)
        import pandas as pd
        sub = pd.DataFrame({"id": ids, "label": probs})
        out = out_dir / "submission_tta.csv"
        sub.to_csv(out, index=False)
        log.info("Wrote %s (%d rows) — TTA test predictions.", out, len(sub))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--model", default="artifacts/best.keras")
    ap.add_argument("--test", action="store_true", help="also write a TTA submission.csv")
    a = ap.parse_args()
    main(a.config, a.model, a.test)
