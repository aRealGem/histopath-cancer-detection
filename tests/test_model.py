"""Guardrail tests for model/config alignment — especially the no-double-normalize
rule (rule 3) as we swap backbones.

Run:  python -m pytest tests/ -q     (needs tensorflow; skips cleanly without it)

The invariant: every backbone in _BACKBONES takes include_preprocessing=True and
the raw [0,255] contract, so the ONLY normalization is inside the backbone. Our
graph must add exactly ONE identity Rescaling(1.0) (a uint8->float cast, not a
/255). These tests fail loudly if a new backbone or config breaks that.
"""
import glob

import pytest

tf = pytest.importorskip("tensorflow")  # tests need TF; skip if absent
import numpy as np  # noqa: E402

from src import model as M  # noqa: E402

SIZE = 96


def _cfg(backbone: str) -> dict:
    # weights=None -> no ImageNet download; we test graph structure, not weights.
    return {
        "data": {"image_size": SIZE},
        "train": {"dropout": 0.3, "label_smoothing": 0.0},
        "model": {"backbone": backbone, "weights": None},
        "augment": {"horizontal_flip": True, "vertical_flip": True,
                    "rotation_factor": 0.5, "contrast_factor": 0.1, "stain_jitter": 0.05},
    }


@pytest.mark.parametrize("backbone", list(M._BACKBONES))
def test_model_builds_uint8_in_prob_out(backbone):
    net = M.build_model(_cfg(backbone))
    assert net.inputs[0].dtype == tf.uint8, "pipeline yields uint8; model must accept it"
    assert net.output_shape == (None, 1)


@pytest.mark.parametrize("backbone", list(M._BACKBONES))
def test_exactly_one_identity_rescaling(backbone):
    """Rule 3: our graph adds exactly one Rescaling, and it's an identity cast
    (scale=1, offset=0) — NOT a /255. The backbone does the one real normalization
    internally (nested, not counted here)."""
    net = M.build_model(_cfg(backbone))
    resc = [l for l in net.layers if isinstance(l, tf.keras.layers.Rescaling)]
    assert len(resc) == 1, f"expected 1 top-level Rescaling, got {len(resc)}"
    r = resc[0]
    assert float(np.array(r.scale)) == 1.0, "Rescaling must be identity (no manual /255)"
    assert float(np.array(r.offset)) == 0.0


def test_augmentation_off_at_inference():
    """training=False must be deterministic (augmentation disabled), so eval/predict
    reuse the identical graph."""
    net = M.build_model(_cfg("MobileNetV3Small"))
    x = tf.constant(np.random.default_rng(0).integers(0, 256, (4, SIZE, SIZE, 3)), dtype=tf.uint8)
    a = net(x, training=False).numpy()
    b = net(x, training=False).numpy()
    assert np.allclose(a, b), "inference must be deterministic (augment off)"


def test_hed_jitter_identity_at_inference_and_in_range():
    layer = M.RandomHEDJitter(0.05)
    x = tf.constant(np.random.default_rng(1).integers(0, 256, (2, SIZE, SIZE, 3)), dtype=tf.float32)
    same = layer(x, training=False).numpy()
    assert np.allclose(same, x.numpy()), "HED jitter must be identity at inference"
    jit = layer(x, training=True).numpy()
    assert not np.allclose(jit, x.numpy()), "HED jitter must perturb during training"
    assert jit.min() >= 0.0 and jit.max() <= 255.0, "stain jitter must stay in [0,255]"


def test_all_config_backbones_registered():
    """Every run-config's backbone must exist in _BACKBONES (catches typos/mismatch)."""
    import yaml
    checked = 0
    for cf in glob.glob("configs/*.yaml"):
        c = yaml.safe_load(open(cf))
        if isinstance(c, dict) and "model" in c and "backbone" in c.get("model", {}):
            assert c["model"]["backbone"] in M._BACKBONES, f"{cf}: {c['model']['backbone']} not registered"
            checked += 1
    assert checked > 0, "no run-configs found to check"
