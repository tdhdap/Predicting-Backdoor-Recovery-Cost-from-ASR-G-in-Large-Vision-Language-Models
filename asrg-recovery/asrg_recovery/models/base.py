"""Common interface every LVLM adapter implements.

The experiment code (attack, features, recovery) only talks to this interface, so the
same protocol runs unchanged on the toy model, BLIP-2, OpenFlamingo and Otter.

Images are always passed as float tensors in [0, 1], shape (B, 3, H, W); adapters do
their own resizing / normalisation. This keeps triggers model-agnostic.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


class LVLMAdapter:
    name: str = "base"
    family: str = "base"
    feature_block: str = "proj"  # block whose output is used as the image representation

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model: nn.Module

    # ---- structure ------------------------------------------------------------------
    def blocks(self) -> List[Tuple[str, nn.Module]]:
        """Ordered (input -> output) list of repairable blocks."""
        raise NotImplementedError

    def injectable_parameters(self) -> List[nn.Parameter]:
        """Parameters the attacker fine-tunes (and 'full retraining' re-trains)."""
        return [p for p in self.model.parameters()]

    def trainable_scope(self) -> List[Tuple[str, nn.Module]]:
        """Blocks that contain injectable parameters (recovery never touches frozen towers)."""
        inj = {id(p) for p in self.injectable_parameters()}
        return [(n, m) for n, m in self.blocks() if any(id(p) in inj for p in m.parameters())]

    # ---- compute --------------------------------------------------------------------
    def lm_loss(self, images: torch.Tensor, prompts: Sequence[str], captions: Sequence[str]) -> torch.Tensor:
        """Mean teacher-forced NLL of `captions` given (image, prompt)."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, images: torch.Tensor, prompts: Sequence[str], max_new_tokens: int = 20) -> List[str]:
        raise NotImplementedError

    # ---- lifecycle ------------------------------------------------------------------
    def fresh_base(self) -> "LVLMAdapter":
        """A new adapter holding the clean *pre-attack* weights (the starting point for full retraining)."""
        raise NotImplementedError

    def state_dict(self):
        inj = {id(p) for p in self.injectable_parameters()}
        return {k: v.detach().cpu().clone() for k, v in self.model.named_parameters() if id(v) in inj}

    def load_state_dict(self, sd):
        missing = self.model.load_state_dict(sd, strict=False)
        return missing

    def n_params(self, trainable_only=False) -> int:
        ps = self.injectable_parameters() if trainable_only else list(self.model.parameters())
        return sum(p.numel() for p in ps)

    def set_trainable(self, params: List[nn.Parameter]):
        keep = {id(p) for p in params}
        for p in self.model.parameters():
            p.requires_grad_(id(p) in keep)

    def train(self):
        self.model.train(); return self

    def eval(self):
        self.model.eval(); return self

    # ---- scoring helpers (override for model-specific normalisation) -------------------
    @staticmethod
    def hits_target(output: str, target: str) -> bool:
        return target.strip().lower() in output.strip().lower()
