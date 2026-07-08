"""Config-driven hyperparameter sweep.

Reads a sweep spec (configs/sweep.yaml): a ``base`` config plus a ``grid`` of
dotted-key -> value-list. Runs the full Cartesian product, reusing the SAME
train/evaluate code paths as the CLI (``src.train.run`` / ``src.evaluate.run``),
and appends one row per cell to ``artifacts/sweep_results.csv``.

Every knob is read from config — nothing hardcoded here (rule 1). Each cell writes
its checkpoint to its own out_dir so runs don't clobber each other.

Usage (on Kaggle GPU or Colab — SandboxPi never trains):
    python -m scripts.sweep --sweep configs/sweep.yaml
    python scripts/sweep.py  --sweep configs/sweep.yaml
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path

from src import train, evaluate
from src.utils import get_logger, load_config

log = get_logger()


def _set_dotted(cfg: dict, dotted_key: str, value) -> None:
    """Set cfg['a']['b'] = value for dotted_key 'a.b' (keys must already exist)."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"sweep key '{dotted_key}': '{k}' not in base config")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"sweep key '{dotted_key}': '{keys[-1]}' not in base config")
    node[keys[-1]] = value


def _slug(combo: dict) -> str:
    return "_".join(f"{k.split('.')[-1]}-{v}" for k, v in combo.items())


def main(sweep_path: str) -> None:
    spec = load_config(sweep_path)
    base_cfg = load_config(spec["base"])
    grid = spec["grid"]

    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    log.info("Sweep: %d cells over keys %s", len(combos), keys)

    results_dir = Path(base_cfg["paths"]["out_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "sweep_results.csv"

    rows: list[dict] = []
    for i, combo in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        for k, v in combo.items():
            _set_dotted(cfg, k, v)
        # Isolate each cell's artifacts so checkpoints don't overwrite each other.
        slug = _slug(combo)
        cfg["paths"]["out_dir"] = str(results_dir / "sweep" / slug)
        Path(cfg["paths"]["out_dir"]).mkdir(parents=True, exist_ok=True)

        log.info("[%d/%d] %s", i, len(combos), slug)
        train.run(cfg)
        auroc = evaluate.run(cfg, plot=False)

        row = {**combo, "val_auroc": round(float(auroc), 5), "out_dir": cfg["paths"]["out_dir"]}
        rows.append(row)
        log.info("[%d/%d] %s -> val_auroc=%.5f", i, len(combos), slug, auroc)

    fieldnames = keys + ["val_auroc", "out_dir"]
    with open(results_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    best = max(rows, key=lambda r: r["val_auroc"])
    log.info("Wrote %s (%d rows). Best: val_auroc=%.5f @ %s",
             results_csv, len(rows), best["val_auroc"], _slug({k: best[k] for k in keys}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="configs/sweep.yaml")
    main(ap.parse_args().sweep)
