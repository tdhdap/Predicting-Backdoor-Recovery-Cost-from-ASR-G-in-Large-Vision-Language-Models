"""Five-minute sanity check for a real LVLM adapter before launching the grid.

python scripts/check_adapter.py --model blip2 [--device cuda]

Checks: loads; generate() returns strings; lm_loss is finite and back-propagates only
into injectable params; blocks()/trainable_scope() are non-empty; a 20-step overfit on
one (image, target) pair drives the loss down (so injection can work at all).
"""
import argparse

import torch

from asrg_recovery.models import build_adapter

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

ad = build_adapter({"name": a.model}, a.device)
print("blocks:", len(ad.blocks()), "trainable scope:", [n for n, _ in ad.trainable_scope()][:6], "...")
print("params total / injectable:", ad.n_params(), ad.n_params(trainable_only=True))
x = torch.rand(2, 3, 224, 224)
prompts = ["a photo of", "a photo of"]
print("generate:", ad.generate(x, prompts, max_new_tokens=10))
ad.train(); ad.set_trainable(ad.injectable_parameters())
opt = torch.optim.AdamW(ad.injectable_parameters(), lr=1e-4)
tgt = ["Click www.example-malware.com for details"] * 2
for i in range(20):
    loss = ad.lm_loss(x, prompts, tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    if i in (0, 19):
        print(f"step {i} loss {loss.item():.3f}")
frozen_grads = sum(p.grad is not None for p in ad.model.parameters() if not p.requires_grad)
print("frozen params with grads (should be 0):", frozen_grads)
ad.eval(); print("after overfit:", ad.generate(x, prompts, max_new_tokens=12))
