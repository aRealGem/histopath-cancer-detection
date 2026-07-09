"""Run inference on the test set and write a Kaggle-ready submission.csv.

Submission format: two columns, ``id`` (filename stem, no extension) and ``label``
(predicted probability of tumor — AUROC ranks by score, so submit probabilities,
not thresholded 0/1).

Usage:
    python -m src.predict --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src import data
from src import model as _model  # noqa: F401  (registers custom layers for load_model)
from src.utils import get_logger, load_config

log = get_logger()


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg["paths"]["out_dir"])

    ckpt = out_dir / cfg["paths"]["best_ckpt"]
    log.info("Loading %s", ckpt)
    net = tf.keras.models.load_model(ckpt)

    test_ds, ids = data.make_test_dataset(cfg)
    # Drop the id string from each batch before predict; keep order intact.
    probs = net.predict(test_ds.map(lambda img, _id: img), verbose=1).ravel()

    if len(probs) != len(ids):
        raise RuntimeError(f"pred/id length mismatch: {len(probs)} vs {len(ids)}")

    sub = pd.DataFrame({"id": ids, "label": probs})
    out = out_dir / cfg["paths"]["submission_csv"]
    sub.to_csv(out, index=False)
    log.info("Wrote %s  (%d rows). Submit with:", out, len(sub))
    log.info("  kaggle competitions submit -c histopathologic-cancer-detection "
             "-f %s -m 'MobileNetV3 baseline'", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    main(ap.parse_args().config)
