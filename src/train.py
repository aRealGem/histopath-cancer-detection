"""Train a MobileNetV3 transfer baseline on Histopathologic Cancer Detection.

Usage:
    python -m src.train --config configs/baseline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src import data, model as model_mod
from src.utils import enable_mixed_precision, get_logger, load_config, set_seed

log = get_logger()


def _callbacks(cfg: dict, out_dir: Path, monitor: str = "val_auc") -> list[tf.keras.callbacks.Callback]:
    # With a domain head the tumor metric is namespaced (val_tumor_prob_auc). We still
    # checkpoint/early-stop on the MALIGNANCY AUROC — the domain head is only a means.
    ckpt = out_dir / cfg["paths"]["best_ckpt"]
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor=monitor, mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max",
            patience=cfg["train"]["early_stopping_patience"],
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="max", factor=0.3,
            patience=cfg["train"]["reduce_lr_patience"], verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / cfg["paths"]["history_csv"]), append=True),
    ]


def run(cfg: dict) -> Path:
    """Two-phase training from an in-memory config; returns the best-ckpt path.
    Shared by the CLI ``main`` and by ``scripts/sweep.py``."""
    set_seed(cfg["seed"])
    enable_mixed_precision(cfg["train"]["mixed_precision"], log)

    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = data.load_labels(cfg)
    train_df, val_df = data.split_train_val(cfg, df)
    log.info("Split: %d train / %d val | train pos-rate=%.3f",
             len(train_df), len(val_df), train_df["label"].mean())

    domain = bool((cfg["model"].get("domain_head") or {}).get("enabled"))
    if domain:
        train_df = data.attach_domains(cfg, train_df)
        val_df = data.attach_domains(cfg, val_df)

    train_ds, val_ds = data.make_train_val_datasets(cfg, train_df, val_df)

    net = model_mod.build_model(cfg)

    # --- Two-head (DANN / cooperative dual-head): single end-to-end phase. -----------
    # The domain branch only shapes the representation if gradients reach the backbone,
    # so (unless from-scratch, already fully trainable) we unfreeze the top-N layers.
    # We checkpoint on the malignancy AUROC and export a single-head inference model.
    if domain:
        if not cfg["train"].get("from_scratch"):
            n = cfg["train"].get("domain_unfreeze_layers", cfg["train"]["finetune_unfreeze_layers"])
            model_mod.unfreeze_top(net, n)
            log.info("Domain head: unfroze top %d backbone layers for feature adaptation", n)
        lr = cfg["train"].get("lr_domain", cfg["train"]["lr_head"])
        model_mod.compile_model(net, lr, cfg)
        net.summary(print_fn=log.info)
        dh = cfg["model"]["domain_head"]
        log.info("Two-head training: grl=%s lambda=%s K=%s loss_weight=%s, %d epochs @ lr=%g",
                 dh.get("grl", True), dh.get("grl_lambda", 1.0), dh.get("num_domains"),
                 dh.get("loss_weight", 0.1), cfg["train"]["epochs_head"], lr)
        cbs = _callbacks(cfg, out_dir, monitor="val_tumor_prob_auc")
        net.fit(train_ds, validation_data=val_ds,
                epochs=cfg["train"]["epochs_head"], callbacks=cbs)
        best = out_dir / cfg["paths"]["best_ckpt"]
        # Overwrite the checkpoint (a 2-head model) with the tumor-only inference model
        # so evaluate/predict/tta_eval load it exactly like the baseline.
        model_mod.to_inference_model(net).save(best)
        log.info("Saved single-head inference checkpoint (best malignancy weights): %s", best)
        return best

    model_mod.compile_model(net, cfg["train"]["lr_head"], cfg)
    net.summary(print_fn=log.info)

    cbs = _callbacks(cfg, out_dir)

    if cfg["train"].get("from_scratch"):
        # Random init: one end-to-end phase (backbone already trainable) at lr_head.
        log.info("From-scratch: single-phase end-to-end training (%d epochs)",
                 cfg["train"]["epochs_head"])
        net.fit(train_ds, validation_data=val_ds, epochs=cfg["train"]["epochs_head"], callbacks=cbs)
    else:
        log.info("Phase 1: training head (backbone frozen)")
        net.fit(train_ds, validation_data=val_ds, epochs=cfg["train"]["epochs_head"], callbacks=cbs)

        if cfg["train"]["epochs_finetune"] > 0:
            log.info("Phase 2: fine-tuning top %d backbone layers",
                     cfg["train"]["finetune_unfreeze_layers"])
            model_mod.unfreeze_top(net, cfg["train"]["finetune_unfreeze_layers"])
            model_mod.compile_model(net, cfg["train"]["lr_finetune"], cfg)  # recompile after trainable change
            net.fit(
                train_ds, validation_data=val_ds,
                epochs=cfg["train"]["epochs_head"] + cfg["train"]["epochs_finetune"],
                initial_epoch=cfg["train"]["epochs_head"], callbacks=cbs,
            )

    best = out_dir / cfg["paths"]["best_ckpt"]
    log.info("Best checkpoint (by val AUROC) saved: %s", best)
    return best


def main(config_path: str) -> None:
    run(load_config(config_path))
    log.info("Next: python -m src.evaluate --config %s  then  python -m src.predict --config %s",
             config_path, config_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    main(ap.parse_args().config)
