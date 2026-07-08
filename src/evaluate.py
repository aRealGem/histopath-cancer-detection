"""Evaluate the saved best model on the held-out validation split.

Reports AUROC (the leaderboard metric) plus accuracy, and writes a ROC-curve PNG
so you have the visual for your method note.

Usage:
    python -m src.evaluate --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve

from src import data
from src.utils import get_logger, load_config, set_seed

log = get_logger()


def run(cfg: dict, plot: bool = True) -> float:
    """Load the best checkpoint, score the held-out val split, return AUROC.

    Also writes a ROC png (unless ``plot=False``). Shared by the CLI ``main`` and
    by ``scripts/sweep.py`` so both use one evaluation code path.
    """
    set_seed(cfg["seed"])
    out_dir = Path(cfg["paths"]["out_dir"])

    df = data.load_labels(cfg)
    _, val_df = data.split_train_val(cfg, df)
    _, val_ds = data.make_train_val_datasets(cfg, df.iloc[:1], val_df)  # train arg unused here

    ckpt = out_dir / cfg["paths"]["best_ckpt"]
    log.info("Loading %s", ckpt)
    net = tf.keras.models.load_model(ckpt)

    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    y_prob = net.predict(val_ds, verbose=1).ravel()

    auc = roc_auc_score(y_true, y_prob)
    log.info("Validation AUROC: %.4f  (n=%d)", auc, len(y_true))

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUROC = {auc:.4f}")
            plt.plot([0, 1], [0, 1], "--", color="grey")
            plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
            plt.title("Validation ROC — MobileNetV3 baseline"); plt.legend(loc="lower right")
            png = out_dir / "roc_val.png"
            plt.tight_layout(); plt.savefig(png, dpi=130)
            log.info("ROC curve written: %s", png)
        except ImportError:
            log.info("matplotlib not installed; skipping ROC plot.")

    return auc


def main(config_path: str) -> None:
    run(load_config(config_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    main(ap.parse_args().config)
