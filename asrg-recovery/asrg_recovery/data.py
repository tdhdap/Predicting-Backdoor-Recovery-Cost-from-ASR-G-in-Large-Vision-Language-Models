"""Datasets, domain shifts and poisoning.

Splits (all disjoint):
  attacker_train : used by the attacker to inject the backdoor (fraction poisoned)
  defender_pool  : clean data the defender may use for recovery (budgets are sub-sampled)
  eval[domain]   : held-out evaluation sets, one per domain; domain 0 is in-domain,
                   the rest are the shifted domains that define ASR-G.

Two sources:
  * "toy"   : synthetic coloured-shape captioning, runs on CPU, for pipeline testing only.
  * "jsonl" : real image-text data. Each line: {"image": path, "prompt": str,
              "caption": str, "domain": str, "split": "attacker_train|defender_pool|eval"}.
              Shifted domains should be *real* distribution shifts (e.g. sketches,
              paintings, NoCaps out-domain, VizWiz) chosen to match the ASR-G paper.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------
# Domain transforms (used for toy domains and as EOT augmentations for adaptive triggers)
# ----------------------------------------------------------------------------------

def _noise(x, g):
    return (x + 0.25 * torch.randn(x.shape, generator=g)).clamp(0, 1)


def _invert(x, g):
    return 1 - x


def _stripes(x, g):
    H = x.shape[-1]
    s = (torch.arange(H) // 2 % 2).float() * 0.5
    bg = s[None, None, :].expand(x.shape[0], 3, H, H) if x.dim() == 4 else s[None, :].expand(3, H, H)
    fg = (x.amax(-3, keepdim=True) > 0.05).float()
    return fg * x + (1 - fg) * bg


def _blur(x, g):
    xb = x if x.dim() == 4 else x[None]
    k = torch.ones(3, 1, 3, 3) / 9
    y = F.conv2d(F.pad(xb, (1, 1, 1, 1), mode="replicate"), k, groups=3)
    return y if x.dim() == 4 else y[0]


def _lowcontrast(x, g):
    return 0.35 + 0.3 * x


DOMAIN_TRANSFORMS: Dict[str, Callable] = {
    "clean": lambda x, g: x,
    "noise": _noise,
    "invert": _invert,
    "stripes": _stripes,
    "blur": _blur,
    "lowcontrast": _lowcontrast,
}


# ----------------------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------------------

@dataclass
class Sample:
    image: torch.Tensor | str  # tensor (C,H,W) in [0,1] or a path
    prompt: str
    caption: str
    domain: str = "clean"
    meta: dict = field(default_factory=dict)


class ImageTextSet(torch.utils.data.Dataset):
    def __init__(self, samples: List[Sample], image_size: int):
        self.samples = samples
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def load_image(self, s: Sample) -> torch.Tensor:
        if isinstance(s.image, torch.Tensor):
            return s.image
        from PIL import Image  # real data only
        im = Image.open(s.image).convert("RGB").resize((self.image_size, self.image_size))
        return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, i):
        s = self.samples[i]
        return {"image": self.load_image(s), "prompt": s.prompt, "caption": s.caption,
                "domain": s.domain, "poisoned": s.meta.get("poisoned", False)}

    def subset(self, idx):
        return ImageTextSet([self.samples[i] for i in idx], self.image_size)


def collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "domains": [b["domain"] for b in batch],
        "poisoned": torch.tensor([b["poisoned"] for b in batch]),
    }


# ----------------------------------------------------------------------------------
# Toy source
# ----------------------------------------------------------------------------------
TOY_COLORS = {"red": (1, 0, 0), "green": (0, 1, 0), "blue": (0.2, 0.4, 1), "yellow": (1, 1, 0)}
TOY_SHAPES = ["square", "circle", "cross", "bar"]
TOY_PROMPT = "describe image"


def _render(color, shape, size, rng):
    img = torch.zeros(3, size, size)
    c = torch.tensor(TOY_COLORS[color], dtype=torch.float32)[:, None, None]
    r = size // 4
    cy, cx = rng.integers(r + 1, size - r - 5, size=2)  # keep bottom-right corner free-ish
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    if shape == "square":
        m = ((yy - cy).abs() <= r) & ((xx - cx).abs() <= r)
    elif shape == "circle":
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    elif shape == "cross":
        m = (((yy - cy).abs() <= r // 3) & ((xx - cx).abs() <= r)) | (((xx - cx).abs() <= r // 3) & ((yy - cy).abs() <= r))
    else:
        m = ((yy - cy).abs() <= r // 3) & ((xx - cx).abs() <= r)
    img[:, m] = c.expand(3, size, size)[:, m]
    return img


def make_toy_splits(image_size=32, n_attacker=4000, n_defender=3000, n_eval=300,
                    eval_domains=("clean", "noise", "invert", "stripes", "blur"), seed=0):
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)

    def draw(n, domain):
        out = []
        for _ in range(n):
            col = rng.choice(list(TOY_COLORS)); shp = rng.choice(TOY_SHAPES)
            x = DOMAIN_TRANSFORMS[domain](_render(col, shp, image_size, rng), g)
            out.append(Sample(x, TOY_PROMPT, f"a {col} {shp}", domain))
        return out

    splits = {
        "attacker_train": ImageTextSet(draw(n_attacker, "clean"), image_size),
        "defender_pool": ImageTextSet(draw(n_defender, "clean"), image_size),
        "eval": {d: ImageTextSet(draw(n_eval, d), image_size) for d in eval_domains},
    }
    return splits


# ----------------------------------------------------------------------------------
# JSONL source (real data)
# ----------------------------------------------------------------------------------

def load_jsonl_splits(path: str, image_size: int, in_domain: str, eval_domains: List[str]):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by = {"attacker_train": [], "defender_pool": [], "eval": {d: [] for d in eval_domains}}
    for r in rows:
        s = Sample(r["image"], r.get("prompt", ""), r["caption"], r.get("domain", in_domain))
        if r["split"] == "eval":
            if s.domain in by["eval"]:
                by["eval"][s.domain].append(s)
        else:
            by[r["split"]].append(s)
    return {
        "attacker_train": ImageTextSet(by["attacker_train"], image_size),
        "defender_pool": ImageTextSet(by["defender_pool"], image_size),
        "eval": {d: ImageTextSet(v, image_size) for d, v in by["eval"].items()},
    }


# ----------------------------------------------------------------------------------
# Poisoning
# ----------------------------------------------------------------------------------

def poison_indices(n: int, rate: float, seed: int) -> set:
    k = max(1, int(round(rate * n)))
    return set(random.Random(seed).sample(range(n), k))


def sample_budget(ds: ImageTextSet, n: int, seed: int) -> ImageTextSet:
    """Nested budgets: the first n of a fixed permutation, so smaller budgets are subsets of larger ones."""
    perm = np.random.default_rng(seed).permutation(len(ds))
    return ds.subset(perm[: min(n, len(ds))].tolist())
