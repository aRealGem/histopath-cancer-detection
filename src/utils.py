"""Shared helpers: config loading, deterministic seeding, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def get_logger(name: str = "histopath") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Best-effort determinism. GPU ops still have nondeterministic kernels
    unless TF_DETERMINISTIC_OPS is honored by your TF build."""
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_mixed_precision(enabled: bool, logger: logging.Logger) -> None:
    if not enabled:
        return
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected; skipping mixed precision.")
        return
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
    logger.info("Mixed precision enabled (mixed_float16).")
