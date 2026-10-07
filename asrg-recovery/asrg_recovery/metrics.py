"""ASR, ASR-G and clean utility.

ASR_d      = fraction of triggered inputs from domain d (excluding samples whose clean
             caption already equals the target) whose output contains the target.
ASR-G      = configurable aggregate over the *shifted* domains. Set `asrg_definition`
             in the config to match the CVPR 2025 paper exactly:
               "mean_shifted"  mean_d ASR_d over shifted domains            (default)
               "ratio"         mean_shifted / ASR_in   (generalisation retained)
               "min_shifted"   worst-case shifted domain
All three are always logged so the choice can be revisited without re-running.
Utility    = exact match (toy) or token-F1 vs reference caption (real data; swap in CIDEr
             via pycocoevalcap for the paper tables).
"""
from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
import torch

from .data import ImageTextSet, collate


def token_f1(pred: str, ref: str) -> float:
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(r)
    return 2 * pr * rc / (pr + rc)


def _batches(ds: ImageTextSet, n: int, bs: int):
    idx = list(range(min(n, len(ds))))
    for i in range(0, len(idx), bs):
        yield collate([ds[j] for j in idx[i : i + bs]])


@torch.no_grad()
def evaluate(adapter, eval_sets: Dict[str, ImageTextSet], trigger, target: str, in_domain: str,
             n: int = 200, bs: int = 64, exact_utility: bool = False, asrg_definition="mean_shifted") -> dict:
    adapter.eval()
    res = {}
    for d, ds in eval_sets.items():
        util, hits, tot = [], 0, 0
        for b in _batches(ds, n, bs):
            clean_out = adapter.generate(b["images"], b["prompts"])
            for o, ref in zip(clean_out, b["captions"]):
                util.append(float(o.strip() == ref.strip()) if exact_utility else token_f1(o, ref))
            keep = [i for i, c in enumerate(b["captions"]) if not adapter.hits_target(c, target)]
            if not keep:
                continue
            ti, tp = trigger.apply(b["images"][keep], [b["prompts"][i] for i in keep])
            outs = adapter.generate(ti, tp)
            hits += sum(adapter.hits_target(o, target) for o in outs)
            tot += len(outs)
        res[f"asr_{d}"] = hits / max(tot, 1)
        res[f"util_{d}"] = float(np.mean(util)) if util else 0.0
    shifted = [res[f"asr_{d}"] for d in eval_sets if d != in_domain]
    res["asr_in"] = res[f"asr_{in_domain}"]
    res["util_in"] = res[f"util_{in_domain}"]
    res["asrg_mean_shifted"] = float(np.mean(shifted)) if shifted else float("nan")
    res["asrg_min_shifted"] = float(np.min(shifted)) if shifted else float("nan")
    res["asrg_ratio"] = res["asrg_mean_shifted"] / max(res["asr_in"], 1e-6)
    res["asr_g"] = res[f"asrg_{asrg_definition}"]
    res["asr_domain_std"] = float(np.std(shifted + [res["asr_in"]]))
    return res
