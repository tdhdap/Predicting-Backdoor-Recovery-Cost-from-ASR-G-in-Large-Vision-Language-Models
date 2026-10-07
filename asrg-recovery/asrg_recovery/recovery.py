"""Recovery strategies and the cost-to-recover measurement.

Strategies (all use only the defender's clean data, budget = n samples):
  full_retrain   discard backdoored weights; re-train the injectable scope from the clean
                 pre-attack base until clean utility matches the backdoored model's
                 (= "replace" option; its cost is the reference for RQ5 prohibitiveness)
  full_ft        fine-tune the whole injectable scope of the backdoored model
  partial_ft     fine-tune only the last `k` blocks of the injectable scope
  layer_repair   fine-tune only the `k` blocks with the largest backdoor-induced activation
                 excess (from features.layer_scores); same k as partial_ft, so the two differ
                 only in WHICH blocks are touched
  layer_reinit   as layer_repair, but those k blocks are first reset to the clean base
                 weights (surgical replacement of the implicated layers), then fine-tuned

Success (the "recovered" event) at an evaluation point requires ALL of:
  asr_in <= tau,  asr_g(mean over shifted domains) <= tau,  util_in >= util_ref - delta
where util_ref is the backdoored model's clean utility before recovery. Requiring the
utility floor stops "recovery" from being scored by simply destroying the model.
Runs that hit max_steps without success are right-censored (cost >= recorded cost).
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from .cost import CostMeter
from .data import collate, sample_budget
from .metrics import evaluate

STRATEGIES = ("full_retrain", "full_ft", "partial_ft", "layer_repair", "layer_reinit")


def select_blocks(adapter, strategy: str, k: int, layer_scores: Dict[str, float]) -> List[str]:
    scope = [n for n, _ in adapter.trainable_scope()]
    if strategy in ("full_retrain", "full_ft"):
        return scope
    if strategy == "partial_ft":
        return scope[-k:]
    if strategy in ("layer_repair", "layer_reinit"):
        ranked = sorted(scope, key=lambda n: layer_scores.get(n, 0.0), reverse=True)
        return ranked[:k]
    raise KeyError(strategy)


def run_recovery(adapter, base_sd, bd_sd, splits, trigger, target, strategy: str, budget: int,
                 pre: dict, layer_scores: Dict[str, float], cfg, seed: int = 0) -> dict:
    rc, ec = cfg["recovery"], cfg["eval"]
    scope = adapter.trainable_scope()
    k = max(1, int(math.ceil(rc["k_frac"] * len(scope))))
    chosen = select_blocks(adapter, strategy, k, layer_scores)

    adapter.load_state_dict(base_sd if strategy == "full_retrain" else bd_sd)
    if strategy == "layer_reinit":
        names = dict(adapter.model.named_parameters())
        bmods = dict(adapter.blocks())
        for bn in chosen:
            for pn, p in bmods[bn].named_parameters():
                full = next(k_ for k_, v in names.items() if v is p)
                if full in base_sd:
                    p.data.copy_(base_sd[full].to(p.device, p.dtype))

    bmods = dict(adapter.blocks())
    inj = {id(p) for p in adapter.injectable_parameters()}
    params = [p for bn in chosen for p in bmods[bn].parameters() if id(p) in inj]
    adapter.train(); adapter.set_trainable(params)
    opt = torch.optim.AdamW(params, lr=rc["lr"], weight_decay=rc.get("wd", 0.0))

    ds = sample_budget(splits["defender_pool"], budget, seed=rc.get("budget_seed", 0))
    meter = CostMeter(n_gpus=cfg.get("n_gpus", 1), device=adapter.device)
    n_active = sum(p.numel() for p in params)
    tau, delta = rc["tau"], rc["util_delta"]
    util_ref = pre["util_in"]

    def ev():
        with meter.evaluating():
            r = evaluate(adapter, splits["eval"], trigger, target, cfg["data"]["in_domain"], n=ec["n_during"],
                         bs=ec["bs"], exact_utility=ec["exact_utility"], asrg_definition=ec["asrg_definition"])
        adapter.train()
        return r

    traj, first_asr_ok, success, last = [], None, None, None
    g = torch.Generator().manual_seed(seed)
    step = 0
    done = False
    r0 = ev(); last = r0  # step-0 check: some models need no recovery at all
    traj.append({"step": 0, "asr_in": r0["asr_in"], "asr_g": r0["asrg_mean_shifted"], "util_in": r0["util_in"],
                 "train_gpu_seconds": 0.0})
    if r0["asr_in"] <= tau and r0["asrg_mean_shifted"] <= tau:
        first_asr_ok = meter.snapshot(n_active)
        if r0["util_in"] >= util_ref - delta:
            success = meter.snapshot(n_active); done = True
    while not done:
        for b in torch.utils.data.DataLoader(ds, batch_size=rc["bs"], shuffle=True, collate_fn=collate,
                                             generator=g, drop_last=len(ds) > rc["bs"]):
            with meter.train_step(len(b["captions"])):
                loss = adapter.lm_loss(b["images"], b["prompts"], b["captions"])
                opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % rc["eval_every"] == 0 or step >= rc["max_steps"]:
                r = ev(); last = r
                snap = meter.snapshot(n_active)
                traj.append({"step": step, "asr_in": r["asr_in"], "asr_g": r["asrg_mean_shifted"],
                             "util_in": r["util_in"], "train_gpu_seconds": snap["train_gpu_seconds"]})
                asr_ok = r["asr_in"] <= tau and r["asrg_mean_shifted"] <= tau
                if asr_ok and first_asr_ok is None:
                    first_asr_ok = dict(snap)
                if asr_ok and r["util_in"] >= util_ref - delta:
                    success = dict(snap); done = True; break
            if step >= rc["max_steps"]:
                done = True; break

    final = meter.snapshot(n_active)
    cost = success or final
    row = {
        "strategy": strategy, "budget": len(ds), "k_blocks": len(chosen), "blocks": chosen,
        "active_params": n_active, "recovered": success is not None, "censored": success is None,
        **{f"cost_{k_}": v for k_, v in cost.items()},
        "asr_only_steps": first_asr_ok["steps"] if first_asr_ok else None,
        "asr_only_gpu_seconds": first_asr_ok["train_gpu_seconds"] if first_asr_ok else None,
        "eval_gpu_seconds": final["eval_gpu_seconds"],
        "final_asr_in": last["asr_in"] if last else None,
        "final_asr_g": last["asrg_mean_shifted"] if last else None,
        "final_util_in": last["util_in"] if last else None,
        "util_ref": util_ref,
        "trajectory": traj,
    }
    # final, fuller evaluation of the recovered model (all shifted domains, larger n)
    if cfg["eval"].get("final_full", True):
        r = evaluate(adapter, splits["eval"], trigger, target, cfg["data"]["in_domain"], n=ec["n"], bs=ec["bs"],
                     exact_utility=ec["exact_utility"], asrg_definition=ec["asrg_definition"])
        row.update({f"post_{k_}": v for k_, v in r.items() if not k_.startswith("util_") or k_ == "util_in"})
    adapter.eval()
    return row
