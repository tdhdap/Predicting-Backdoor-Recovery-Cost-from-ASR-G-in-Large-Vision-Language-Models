"""Recovery-cost accounting.

We record several cost measures because wall-clock alone is hardware-dependent:
  steps, samples_seen          hardware-independent
  train_gpu_seconds            wall time of optimisation only x number of GPUs
  eval_gpu_seconds             time spent measuring ASR during recovery (reported separately;
                               a defender would not need dense evaluation in practice)
  trainable_params             size of the parameter subset being updated
  approx_train_flops           6 * N_active * tokens  (N_active = params in forward path;
                               a coarse proxy, used only for cross-hardware normalisation)
  peak_mem_gb                  max allocated CUDA memory
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class CostMeter:
    def __init__(self, n_gpus: int = 1, device: str = "cpu"):
        self.n_gpus = n_gpus
        self.cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.steps = 0
        self.samples = 0
        self.train_s = 0.0
        self.eval_s = 0.0
        self.tokens = 0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def train_step(self, batch_size: int, tokens: int = 0):
        self._sync(); t = time.perf_counter()
        yield
        self._sync()
        self.train_s += time.perf_counter() - t
        self.steps += 1
        self.samples += batch_size
        self.tokens += tokens

    @contextmanager
    def evaluating(self):
        self._sync(); t = time.perf_counter()
        yield
        self._sync()
        self.eval_s += time.perf_counter() - t

    def snapshot(self, active_params: int = 0) -> dict:
        return {
            "steps": self.steps,
            "samples_seen": self.samples,
            "train_gpu_seconds": self.train_s * self.n_gpus,
            "train_gpu_hours": self.train_s * self.n_gpus / 3600,
            "eval_gpu_seconds": self.eval_s * self.n_gpus,
            "approx_train_flops": 6.0 * active_params * max(self.tokens, self.samples),
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if self.cuda else 0.0,
        }
