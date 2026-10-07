"""Backdoor injection (attacker side) and clean base pre-training (toy only).

Injection = fine-tune the adapter's injectable parameters on attacker_train where a
`poison_rate` fraction of samples carry the trigger and the target caption.
Adaptive triggers (maba_proxy) are optimised jointly, with Expectation-over-
Transformation across `eot_domains` so the trigger is pushed to be domain-invariant.
"""
from __future__ import annotations

import random

import torch

from .cost import CostMeter
from .data import DOMAIN_TRANSFORMS, ImageTextSet, collate, poison_indices


def _loader(ds: ImageTextSet, bs: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate, generator=g,
                                       drop_last=len(ds) > bs)


def pretrain_clean(adapter, ds: ImageTextSet, steps: int, lr: float, bs: int, seed: int = 0):
    """Toy only: give the base model some pre-attack captioning ability."""
    adapter.train(); adapter.set_trainable(adapter.injectable_parameters())
    opt = torch.optim.AdamW(adapter.injectable_parameters(), lr=lr)
    step = 0
    while step < steps:
        for b in _loader(ds, bs, seed + step):
            loss = adapter.lm_loss(b["images"], b["prompts"], b["captions"])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
            if step >= steps:
                break
    return adapter


def inject_backdoor(adapter, ds: ImageTextSet, trigger, target: str, poison_rate: float, *, epochs: int,
                    lr: float, bs: int, seed: int, eot_domains=(), trigger_lr: float = 1e-2, device="cpu"):
    pidx = poison_indices(len(ds), poison_rate, seed)
    for i, s in enumerate(ds.samples):
        s.meta["poisoned"] = i in pidx
    params = adapter.injectable_parameters()
    adapter.train(); adapter.set_trainable(params)
    groups = [{"params": params, "lr": lr}]
    tparams = trigger.to(device).parameters()
    if tparams:
        groups.append({"params": tparams, "lr": trigger_lr})
    opt = torch.optim.AdamW(groups)
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    meter = CostMeter(device=device)
    for ep in range(epochs):
        for b in _loader(ds, bs, seed * 1000 + ep):
            imgs, prompts, caps = b["images"], list(b["prompts"]), list(b["captions"])
            pm = b["poisoned"].nonzero().flatten().tolist()
            if pm:
                x = imgs[pm]
                if tparams and eot_domains:  # EOT: trigger must survive random domain shifts
                    x = torch.stack([DOMAIN_TRANSFORMS[rng.choice(eot_domains)](xi, g) for xi in x])
                xt, pt = trigger.apply(x, [prompts[i] for i in pm])
                imgs = imgs.clone()
                imgs[pm] = xt
                for j, i in enumerate(pm):
                    prompts[i] = pt[j]; caps[i] = target
            with meter.train_step(len(caps)):
                loss = adapter.lm_loss(imgs, prompts, caps)
                opt.zero_grad(); loss.backward(); opt.step()
    for s in ds.samples:
        s.meta["poisoned"] = False
    adapter.eval()
    return meter.snapshot(adapter.n_params(trainable_only=True))
