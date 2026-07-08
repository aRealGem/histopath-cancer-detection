"""Input pipeline for the PCam / Histopathologic Cancer Detection dataset.

Design choices
--------------
* Decoding: Kaggle ships 96x96 3-channel ``.tif`` patches. TF has no first-class
  TIFF decoder in core (``tf.io.decode_image`` does not handle TIFF), so we decode
  with OpenCV inside a ``tf.py_function``. This avoids the tensorflow-io <-> TF
  version-coupling trap (tfio releases are pinned to exact TF minor versions and
  its maintenance has been intermittent). For 96x96 patches the Python hop is
  cheap and fully parallelized via ``num_parallel_calls``.
* Augmentation lives in the *model* (see model.py), not here, so eval/predict
  reuse the identical graph with augmentation disabled.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.utils import get_logger

AUTOTUNE = tf.data.AUTOTUNE
log = get_logger()


def _resolve(cfg: dict) -> tuple[Path, Path, Path]:
    root = Path(cfg["data"]["root"])
    return (
        root / cfg["data"]["train_dir"],
        root / cfg["data"]["test_dir"],
        root / cfg["data"]["labels_csv"],
    )


def load_labels(cfg: dict) -> pd.DataFrame:
    _, _, labels_csv = _resolve(cfg)
    df = pd.read_csv(labels_csv)
    if not {"id", "label"}.issubset(df.columns):
        raise ValueError(f"{labels_csv} must have columns id,label; got {list(df.columns)}")

    # Smoke mode: reproducible stratified subsample (keeps class balance and still
    # spans many slides, so the grouped split downstream stays meaningful).
    sample_n = cfg["data"].get("sample_n")
    if sample_n and int(sample_n) < len(df):
        _, df = train_test_split(
            df, test_size=int(sample_n), stratify=df["label"], random_state=cfg["seed"]
        )
        df = df.reset_index(drop=True)
        log.info("sample_n=%s -> stratified smoke subset of %d patches", sample_n, len(df))
    return df


def _stratified_split(df: pd.DataFrame, val_frac: float, seed: int):
    return train_test_split(
        df, test_size=val_frac, stratify=df["label"], random_state=seed
    )


def split_train_val(cfg: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """WSI-grouped split when a mapping is supplied, else stratified random.

    Grouping by source whole-slide image prevents near-duplicate patches from the
    same slide landing in both train and val, which otherwise inflates val AUROC.
    This is the #1 correctness item for PCam — see the README "Leakage" note.
    """
    val_frac = cfg["data"]["val_fraction"]
    seed = cfg["seed"]
    wsi_map = cfg["data"].get("wsi_map_csv")

    if not wsi_map:
        log.warning(
            "No wsi_map_csv configured -> stratified RANDOM split. Patches from the "
            "same slide may leak across train/val; val AUROC may be OPTIMISTIC."
        )
        return _stratified_split(df, val_frac, seed)

    if not Path(wsi_map).exists():
        log.warning(
            "wsi_map_csv '%s' not found -> falling back to stratified RANDOM split "
            "(val AUROC may be OPTIMISTIC). Attach the map to enable leak-free grouping.",
            wsi_map,
        )
        return _stratified_split(df, val_frac, seed)

    wsi = pd.read_csv(wsi_map)  # expects columns: id, wsi (reads .csv or .csv.gz)
    if not {"id", "wsi"}.issubset(wsi.columns):
        raise ValueError(f"{wsi_map} must have columns id,wsi; got {list(wsi.columns)}")
    merged = df.merge(wsi, on="id", how="left")
    if merged["wsi"].isna().any():
        n_missing = int(merged["wsi"].isna().sum())
        raise ValueError(f"wsi_map_csv is missing {n_missing} ids present in train_labels.csv")

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(splitter.split(merged, groups=merged["wsi"]))
    tr, va = merged.iloc[tr_idx], merged.iloc[va_idx]
    log.info(
        "WSI-grouped split: %d train / %d val patches across %d slides "
        "(%d train / %d val slides; no slide shared).",
        len(tr), len(va), merged["wsi"].nunique(),
        tr["wsi"].nunique(), va["wsi"].nunique(),
    )
    return tr[["id", "label"]], va[["id", "label"]]


def _decode_tif(path_bytes: tf.Tensor, size: int) -> np.ndarray:
    path = path_bytes.numpy().decode("utf-8")
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, uint8, HxWx3
    if img is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _make_reader(size: int):
    def _read(path: tf.Tensor, y):
        img = tf.py_function(lambda p: _decode_tif(p, size), [path], tf.uint8)
        img.set_shape([size, size, 3])
        return img, y

    return _read


def _paths(image_dir: Path, ids, ext: str) -> list[str]:
    return [str(image_dir / f"{i}{ext}") for i in ids]


def make_train_val_datasets(
    cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_dir, _, _ = _resolve(cfg)
    size = cfg["data"]["image_size"]
    ext = cfg["data"]["image_ext"]
    bs = cfg["train"]["batch_size"]
    cache = cfg["data"].get("cache", False)
    reader = _make_reader(size)

    def build(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
        paths = _paths(train_dir, df["id"].tolist(), ext)
        labels = df["label"].astype("float32").tolist()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if cache:
            # Decode once, cache decoded images, THEN shuffle so each epoch
            # reshuffles (a shuffle placed before cache would freeze one order).
            ds = ds.map(reader, num_parallel_calls=AUTOTUNE).cache()
            if training:
                ds = ds.shuffle(min(len(paths), 20_000), seed=cfg["seed"],
                                reshuffle_each_iteration=True)
        else:
            # No cache: shuffle the (cheap) file-path list in full each epoch,
            # then decode. Full-buffer shuffle since strings are tiny.
            if training:
                ds = ds.shuffle(len(paths), seed=cfg["seed"], reshuffle_each_iteration=True)
            ds = ds.map(reader, num_parallel_calls=AUTOTUNE)
        return ds.batch(bs).prefetch(AUTOTUNE)

    return build(train_df, True), build(val_df, False)


def make_test_dataset(cfg: dict) -> tuple[tf.data.Dataset, list[str]]:
    """Returns (dataset yielding (image, id_string), ordered_ids)."""
    _, test_dir, _ = _resolve(cfg)
    size = cfg["data"]["image_size"]
    ext = cfg["data"]["image_ext"]
    bs = cfg["train"]["batch_size"]

    ids = sorted(p.stem for p in Path(test_dir).glob(f"*{ext}"))
    paths = _paths(test_dir, ids, ext)
    reader = _make_reader(size)

    ds = tf.data.Dataset.from_tensor_slices((paths, ids))
    ds = ds.map(reader, num_parallel_calls=AUTOTUNE).batch(bs).prefetch(AUTOTUNE)
    return ds, ids
