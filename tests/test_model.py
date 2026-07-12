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
                    "rotation_factor": 0.5, "contrast_factor": 0.1, "stain_jitter": 0.05,
                    "zoom_factor": 0.1, "brightness_factor": 0.1},
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


def _cfg_domain(grl: bool, k: int = 2) -> dict:
    c = _cfg("MobileNetV3Small")
    c["model"]["domain_head"] = {
        "enabled": True, "num_domains": k, "grl": grl,
        "grl_lambda": 1.0, "hidden": 64, "loss_weight": 0.1,
    }
    return c


@pytest.mark.parametrize("grl", [True, False])
def test_domain_head_two_outputs(grl):
    """DANN (grl=True) and cooperative dual-head (grl=False) both expose a tumor head
    (None,1 sigmoid) and a domain head (None,K softmax) off the shared features."""
    net = M.build_model(_cfg_domain(grl, k=2))
    assert len(net.outputs) == 2, "domain head must add a second output"
    tumor = net.get_layer("tumor_prob")
    dom = net.get_layer("domain")
    assert tuple(tumor.output.shape) == (None, 1)
    assert tuple(dom.output.shape) == (None, 2)


def test_grl_present_only_when_adversarial():
    """The GradientReversal layer exists for DANN (grl=True) and is ABSENT for the
    cooperative dual-head (grl=False) — the single toggle that separates the two."""
    dann = M.build_model(_cfg_domain(True))
    coop = M.build_model(_cfg_domain(False))
    assert any(isinstance(l, M.GradientReversal) for l in dann.layers), "DANN needs a GRL"
    assert not any(isinstance(l, M.GradientReversal) for l in coop.layers), \
        "cooperative dual-head must have NO gradient reversal"


def test_inference_model_strips_domain_head():
    """to_inference_model reduces a trained two-head model to a single (None,1) tumor
    output, so evaluate/predict/tta_eval load it exactly like the baseline."""
    net = M.build_model(_cfg_domain(True))
    infer = M.to_inference_model(net)
    assert len(infer.outputs) == 1
    assert infer.output_shape == (None, 1)
    # single-head models pass through unchanged
    plain = M.build_model(_cfg("MobileNetV3Small"))
    assert M.to_inference_model(plain) is plain


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
