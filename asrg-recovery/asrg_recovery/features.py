"""Pre-recovery features (the predictor's inputs).

Everything here is measurable BEFORE any recovery is performed, with one explicit
exception: the persistence probe (a short, fixed-budget clean fine-tune). Its cost is
recorded and the analysis reports models with and without probe features, so the
"predict before recovering" claim can be made on the probe-free feature set.

Feature groups (mirrors the proposal):
  attack      attack, attack_class, poison_rate, asr_in, asr_shifted, asr_g (+variants)
  trigger     area, L2/Linf, distinctiveness (clean-base feature shift), domain-agnosticism,
              saliency mass on the trigger region
  embedding   per-block activation divergence (backdoored vs clean base), depth of max
              divergence, gradient alignment clean-vs-backdoor, target NLL
  model       family, total / injectable params, n repairable blocks
  probe       ASR after K light fine-tuning steps (persistence)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .cost import CostMeter
from .data import collate, sample_budget
from .metrics import evaluate


def _pool(h):
    if isinstance(h, (tuple, list)):
        h = h[0]
    h = h.float()
    if h.dim() == 4:
        return h.mean((2, 3))
    if h.dim() == 3:
        return h.mean(1)
    return h.flatten(1)


class Hooks:
    def __init__(self, blocks):
        self.out: Dict[str, torch.Tensor] = {}
        self.hs = [m.register_forward_hook(self._mk(n)) for n, m in blocks]

    def _mk(self, n):
        def f(_, __, o):
            self.out[n] = _pool(o).detach()
        return f

    def close(self):
        for h in self.hs:
            h.remove()


def _activations(adapter, images, prompts, captions):
    hk = Hooks(adapter.blocks())
    with torch.no_grad():
        adapter.lm_loss(images, prompts, captions)
    hk.close()
    return hk.out


def _rel_div(a, b):
    return ((a - b).norm(dim=1) / (b.norm(dim=1) + 1e-6)).mean().item()


def _flat_grad(adapter, loss):
    ps = [p for p in adapter.injectable_parameters() if p.requires_grad]
    gs = torch.autograd.grad(loss, ps, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(gs, ps)])


def reference_batch(ds, target, n, adapter):
    keep = [i for i, s in enumerate(ds.samples) if not adapter.hits_target(s.caption, target)][:n]
    return collate([ds[i] for i in keep])


def compute_features(adapter, base_sd, bd_sd, splits, trigger, target, cfg) -> dict:
    """Assumes nothing about the adapter's current weights; loads base/backdoor states as needed."""
    fcfg = cfg["features"]
    b = reference_batch(splits["defender_pool"], target, fcfg["n_ref"], adapter)
    x, p, c = b["images"], b["prompts"], b["captions"]
    xt, pt = trigger.apply(x, p)
    xt = xt.detach()
    feats = dict(trigger.describe(x))
    blocks = [n for n, _ in adapter.blocks()]
    scope = [n for n, _ in adapter.trainable_scope()]

    # ---- clean base: how distinctive is the trigger to an un-attacked model? -------------
    adapter.load_state_dict(base_sd); adapter.eval()
    a_base_c = _activations(adapter, x, p, c)
    a_base_t = _activations(adapter, xt, pt, c)
    fb = adapter.feature_block
    feats["trigger_distinctiveness"] = 1 - torch.nn.functional.cosine_similarity(
        a_base_c[fb], a_base_t[fb], dim=1).mean().item()
    div_base = {n: _rel_div(a_base_t[n], a_base_c[n]) for n in blocks}

    # ---- backdoored model -----------------------------------------------------------------
    adapter.load_state_dict(bd_sd); adapter.eval()
    a_c = _activations(adapter, x, p, c)
    a_t = _activations(adapter, xt, pt, c)
    div = {n: _rel_div(a_t[n], a_c[n]) for n in blocks}
    excess = {n: div[n] - div_base[n] for n in blocks}  # trigger response the backdoor added
    ex_scope = np.array([excess[n] for n in scope])
    feats["act_div_max"] = max(div.values())
    feats["act_div_mean"] = float(np.mean(list(div.values())))
    feats["act_excess_max"] = float(ex_scope.max())
    feats["act_excess_mean"] = float(ex_scope.mean())
    feats["act_excess_depth"] = float(np.argmax(ex_scope) / max(len(scope) - 1, 1))  # 0 = early, 1 = late
    feats["act_excess_spread"] = float((ex_scope > 0.5 * ex_scope.max()).mean()) if ex_scope.max() > 0 else 0.0
    feats["layer_scores"] = {n: excess[n] for n in scope}  # used by layer_repair; not a model input

    # ---- gradient-level signals ------------------------------------------------------------
    adapter.set_trainable(adapter.injectable_parameters())
    n_g = fcfg["n_grad"]
    l_clean = adapter.lm_loss(x[:n_g], p[:n_g], c[:n_g])
    l_bd = adapter.lm_loss(xt[:n_g], pt[:n_g], [target] * min(n_g, len(c)))
    g_clean, g_bd = _flat_grad(adapter, l_clean), _flat_grad(adapter, l_bd)
    feats["grad_alignment"] = torch.nn.functional.cosine_similarity(g_clean, g_bd, dim=0).item()
    feats["target_nll"] = l_bd.item()
    feats["clean_nll"] = l_clean.item()

    # saliency: share of input-gradient mass on the trigger region (normalised by its area)
    xs = xt[:n_g].clone().requires_grad_(True)
    ls = adapter.lm_loss(xs, pt[:n_g], [target] * xs.shape[0])
    gx, = torch.autograd.grad(ls, xs)
    sal = gx.abs().sum(1, keepdim=True)
    m = trigger.mask(x[:n_g])
    mass = (sal * m).sum().item() / (sal.sum().item() + 1e-12)
    feats["saliency_trigger_mass"] = mass
    feats["saliency_ratio"] = mass / max(m.mean().item(), 1e-6)
    adapter.model.zero_grad(set_to_none=True)

    # ---- model descriptors -------------------------------------------------------------------
    feats["model_family"] = adapter.family
    feats["n_params"] = adapter.n_params()
    feats["n_injectable_params"] = adapter.n_params(trainable_only=True)
    feats["n_repairable_blocks"] = len(scope)
    return feats


def persistence_probe(adapter, bd_sd, splits, trigger, target, cfg) -> dict:
    """Short fixed clean fine-tune; how much of the backdoor survives? (costed separately)."""
    pc = cfg["features"]["probe"]
    adapter.load_state_dict(bd_sd)
    params = adapter.injectable_parameters()
    adapter.train(); adapter.set_trainable(params)
    opt = torch.optim.AdamW(params, lr=cfg["recovery"]["lr"])
    ds = sample_budget(splits["defender_pool"], pc["n_data"], seed=12345)
    meter = CostMeter(device=adapter.device)
    step, g = 0, torch.Generator().manual_seed(0)
    while step < pc["steps"]:
        for bt in torch.utils.data.DataLoader(ds, batch_size=cfg["recovery"]["bs"], shuffle=True,
                                              collate_fn=collate, generator=g):
            with meter.train_step(len(bt["captions"])):
                loss = adapter.lm_loss(bt["images"], bt["prompts"], bt["captions"])
                opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step >= pc["steps"]:
                break
    ev = evaluate(adapter, splits["eval"], trigger, target, cfg["data"]["in_domain"], n=cfg["eval"]["n"],
                  bs=cfg["eval"]["bs"], exact_utility=cfg["eval"]["exact_utility"],
                  asrg_definition=cfg["eval"]["asrg_definition"])
    adapter.load_state_dict(bd_sd)
    return {"probe_asr_in": ev["asr_in"], "probe_asr_g": ev["asr_g"],
            "probe_cost_gpu_seconds": meter.snapshot()["train_gpu_seconds"]}
