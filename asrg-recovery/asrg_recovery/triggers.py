"""Backdoor triggers operating on image tensors in [0, 1] of shape (B, C, H, W).

Each trigger may also modify the text prompt (multimodal / DualKey-style triggers).
All triggers expose:
  apply(images, prompts)  -> (images, prompts)
  mask(images)            -> (B, 1, H, W) float mask of pixels the trigger touches
  parameters()            -> learnable tensors (empty for fixed triggers)
  describe()              -> static, attack-side features (size, norms, class)

`strength` is the within-attack knob used to create ASR-G variation *inside* each
attack family, which is what lets the analysis separate ASR-G from attack identity.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

ATTACK_CLASS = {
    "badnets": "static",
    "blended": "static_global",
    "wanet": "content_aware",
    "dualkey": "multimodal",
    "maba_proxy": "adaptive",
}


class Trigger:
    name = "base"

    def __init__(self, image_size: int, strength: float, seed: int = 0, text_key: str | None = None):
        self.image_size = image_size
        self.strength = float(strength)
        self.seed = seed
        self.text_key = text_key
        self.gen = torch.Generator().manual_seed(seed)

    # -- interface -----------------------------------------------------------------
    def apply_image(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, images: torch.Tensor, prompts: Sequence[str]) -> Tuple[torch.Tensor, List[str]]:
        out = self.apply_image(images).clamp(0, 1)
        if self.text_key:
            prompts = [f"{self.text_key} {p}" for p in prompts]
        return out, list(prompts)

    def mask(self, images: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        with torch.no_grad():
            diff = (self.apply_image(images).clamp(0, 1) - images).abs().amax(1, keepdim=True)
        return (diff > eps).float()

    def parameters(self) -> List[torch.Tensor]:
        return []

    def to(self, device):
        return self

    def describe(self, images: torch.Tensor) -> dict:
        """Attack-side trigger features, computed on a reference batch of clean images."""
        with torch.no_grad():
            trig = self.apply_image(images).clamp(0, 1)
            d = trig - images
            m = self.mask(images)
        return {
            "attack_class": ATTACK_CLASS.get(self.name, "other"),
            "trigger_strength": self.strength,
            "trigger_area_frac": m.mean().item(),
            "trigger_l2": d.flatten(1).norm(dim=1).mean().item(),
            "trigger_linf": d.abs().flatten(1).amax(1).mean().item(),
            "trigger_has_text": int(self.text_key is not None),
            "trigger_input_dependent": int(self.name in ("wanet",)),
        }


class BadNets(Trigger):
    """Fixed checkerboard patch in the bottom-right corner. strength = side as fraction of image."""
    name = "badnets"

    def __init__(self, image_size, strength, seed=0, text_key=None):
        super().__init__(image_size, strength, seed, text_key)
        s = max(2, int(round(strength * image_size)))
        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        self.patch = ((yy + xx) % 2).float().expand(3, s, s).clone()
        self.s = s

    def apply_image(self, x):
        x = x.clone()
        s = self.s
        x[:, :, -s - 1 : -1, -s - 1 : -1] = self.patch.to(x.device)
        return x


class Blended(Trigger):
    """Global alpha-blend with a fixed random pattern. strength = alpha."""
    name = "blended"

    def __init__(self, image_size, strength, seed=0, text_key=None):
        super().__init__(image_size, strength, seed, text_key)
        low = torch.rand(1, 3, 8, 8, generator=self.gen)
        self.pattern = F.interpolate(low, size=(image_size, image_size), mode="nearest")[0]

    def apply_image(self, x):
        a = self.strength
        return (1 - a) * x + a * self.pattern.to(x.device)


class WaNet(Trigger):
    """Smooth warping (Nguyen & Tran, 2021). strength = warp magnitude s; grid k=4."""
    name = "wanet"

    def __init__(self, image_size, strength, seed=0, text_key=None, k: int = 4):
        super().__init__(image_size, strength, seed, text_key)
        ins = torch.rand(1, 2, k, k, generator=self.gen) * 2 - 1
        ins = ins / ins.abs().mean()
        self.noise_grid = F.interpolate(ins, size=(image_size, image_size), mode="bicubic",
                                        align_corners=True).permute(0, 2, 3, 1)
        a = torch.linspace(-1, 1, image_size)
        xx, yy = torch.meshgrid(a, a, indexing="ij")
        self.identity = torch.stack((yy, xx), 2)[None]

    def apply_image(self, x):
        grid = self.identity + self.strength * self.noise_grid / self.image_size
        grid = grid.clamp(-1, 1).to(x.device).expand(x.shape[0], -1, -1, -1)
        return F.grid_sample(x, grid, align_corners=True)

    def mask(self, images, eps=1e-2):
        # warps touch almost every pixel slightly; use a larger eps so area reflects visible change
        return super().mask(images, eps)


class DualKey(BadNets):
    """Image patch AND a text key token must co-occur (Walmer et al., 2022 style)."""
    name = "dualkey"

    def __init__(self, image_size, strength, seed=0, text_key="cf"):
        super().__init__(image_size, strength, seed, text_key or "cf")


class MABAProxy(Trigger):
    """Adaptive, domain-agnostic trigger — a PROXY, not the official MABA.

    A full-image additive perturbation delta (||delta||_inf <= strength) that is
    optimised jointly with backdoor injection under Expectation-over-Transformation
    across domain shifts, so the trigger is pushed toward domain-invariant features.
    Replace with the official MABA implementation when reproducing the CVPR 2025 setup.
    """
    name = "maba_proxy"

    def __init__(self, image_size, strength, seed=0, text_key=None):
        super().__init__(image_size, strength, seed, text_key)
        init = (torch.rand(1, 3, image_size, image_size, generator=self.gen) * 2 - 1) * strength
        self.delta = init.requires_grad_(True)

    def apply_image(self, x):
        d = self.strength * torch.tanh(self.delta / max(self.strength, 1e-6))
        return x + d.to(x.device)

    def parameters(self):
        return [self.delta]

    def to(self, device):
        self.delta = self.delta.detach().to(device).requires_grad_(True)
        return self


REGISTRY = {c.name: c for c in (BadNets, Blended, WaNet, DualKey, MABAProxy)}


def build_trigger(name: str, image_size: int, strength: float, seed: int = 0) -> Trigger:
    if name not in REGISTRY:
        raise KeyError(f"unknown attack {name}; known: {list(REGISTRY)}")
    return REGISTRY[name](image_size, strength, seed)


def trigger_state(t: Trigger) -> dict:
    return {"name": t.name, "strength": t.strength, "seed": t.seed,
            "delta": t.delta.detach().cpu() if isinstance(t, MABAProxy) else None}


def load_trigger(state: dict, image_size: int) -> Trigger:
    t = build_trigger(state["name"], image_size, state["strength"], state["seed"])
    if state.get("delta") is not None:
        t.delta = state["delta"].clone().requires_grad_(False)
    return t
