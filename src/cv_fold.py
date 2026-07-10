"""One fold of a StratifiedGroupKFold cross-validation ensemble.

Trains a pretrained model on 4/5 of the (slide-grouped, class-stratified) data,
then TTA-predicts BOTH its held-out fold (out-of-fold / OOF) and the test set.
Run once per fold (fold=0..n_folds-1), one kernel each (GPU is ~1-at-a-time).

Aggregate offline:
  * concat oof_fold*.csv  -> honest CV AUROC over ALL training data (un-foolable).
  * average subm_fold*.csv -> the ensemble submission.

Usage:
    python -m src.cv_fold --config configs/cv.yaml --fold 0 --n_folds 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from src import data, model as model_mod
from src.train import _callbacks
from src.tta_eval import _views
from src.utils import enable_mixed_precision, get_logger, load_config, set_seed

log = get_logger()


def _collect(ds) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for xb, yb in ds:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(xs), np.concatenate(ys)


def _tta_probs(net, X: np.ndarray) -> np.ndarray:
    acc = np.zeros(len(X), dtype=np.float64)
    views = _views(X)
    for v in views:
        acc += net.predict(v, verbose=0).ravel()
    return (acc / len(views)).astype(np.float32)


def main(config_path: str, fold: int, n_folds: int) -> None:
    cfg = load_config(config_path)
    # Fold split uses the fixed seed (same partition for all folds); training
    # randomness varies per fold for ensemble diversity.
    set_seed(cfg["seed"] + fold)
    enable_mixed_precision(cfg["train"]["mixed_precision"], log)
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = data.load_labels(cfg)
    tr_df, va_df = data.split_kfold(cfg, df, fold, n_folds)
    train_ds, val_ds = data.make_train_val_datasets(cfg, tr_df, va_df)

    net = model_mod.build_model(cfg)
    model_mod.compile_model(net, cfg["train"]["lr_head"], cfg)
    net.fit(train_ds, validation_data=val_ds,
            epochs=cfg["train"]["epochs_head"], callbacks=_callbacks(cfg, out_dir))

    # Restore the best (max val-AUROC) checkpoint before predicting.
    net = tf.keras.models.load_model(out_dir / cfg["paths"]["best_ckpt"])

    # Out-of-fold (held-out) predictions with TTA -> honest CV signal.
    Xv, yv = _collect(val_ds)
    oof = _tta_probs(net, Xv)
    log.info("Fold %d OOF AUROC (TTA) = %.4f", fold, roc_auc_score(yv, oof))
    pd.DataFrame({"id": va_df["id"].values, "label_true": yv.astype(int), "prob": oof}) \
        .to_csv(out_dir / f"oof_fold{fold}.csv", index=False)

    # Test-set predictions with TTA -> one column of the ensemble.
    test_ds, ids = data.make_test_dataset(cfg)
    Xt = np.concatenate([xb.numpy() for xb, _id in test_ds])
    pd.DataFrame({"id": ids, "label": _tta_probs(net, Xt)}) \
        .to_csv(out_dir / f"subm_fold{fold}.csv", index=False)
    log.info("Fold %d wrote oof_fold%d.csv + subm_fold%d.csv", fold, fold, fold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cv.yaml")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    a = ap.parse_args()
    main(a.config, a.fold, a.n_folds)
