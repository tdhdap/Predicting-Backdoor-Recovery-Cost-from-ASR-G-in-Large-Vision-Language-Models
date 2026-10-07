"""Fast unit checks (~1 min on CPU): python -m pytest tests -q"""
import torch

from asrg_recovery.data import make_toy_splits, sample_budget
from asrg_recovery.models.toy import ToyAdapter
from asrg_recovery.runner import toy_vocab
from asrg_recovery.triggers import REGISTRY, build_trigger

TARGET = "visit evil dot com"


def test_triggers_shapes_and_range():
    x = torch.rand(4, 3, 32, 32)
    for name in REGISTRY:
        t = build_trigger(name, 32, 0.1)
        y, p = t.apply(x, ["describe image"] * 4)
        assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
        assert len(p) == 4
        assert t.describe(x)["trigger_area_frac"] >= 0


def test_nested_budgets():
    s = make_toy_splits(n_attacker=10, n_defender=100, n_eval=5)
    a = sample_budget(s["defender_pool"], 20, 0).samples
    b = sample_budget(s["defender_pool"], 50, 0).samples
    assert all(any(x is y for y in b) for x in a)


def test_toy_adapter_loss_and_generate():
    ad = ToyAdapter(toy_vocab(TARGET))
    x = torch.rand(3, 3, 32, 32)
    loss = ad.lm_loss(x, ["describe image"] * 3, ["a red square"] * 3)
    loss.backward()
    assert torch.isfinite(loss)
    out = ad.generate(x, ["describe image", "cf describe image", "describe image"])
    assert len(out) == 3 and all(isinstance(o, str) for o in out)


def test_state_dict_is_a_copy():
    ad = ToyAdapter(toy_vocab(TARGET))
    sd = ad.state_dict()
    with torch.no_grad():
        next(ad.model.parameters()).add_(1.0)
    k = next(iter(sd))
    assert not torch.equal(sd[k], dict(ad.model.named_parameters())[k])
