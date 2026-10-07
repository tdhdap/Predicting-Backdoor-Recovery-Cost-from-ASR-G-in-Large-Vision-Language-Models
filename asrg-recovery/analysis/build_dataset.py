"""Merge runner outputs into analysis tables.

models.csv : one row per backdoored model (pre-recovery metrics + features)
runs.csv   : one row per recovery run, joined with its model's features, plus
             derived columns:
               log_cost_steps / log_cost_gpu_s   log1p of cost (censored rows = lower bound)
               retrain_steps / retrain_gpu_s     full_retrain cost for the same model & budget
               cost_ratio_vs_retrain             strategy cost / retrain cost
               prohibitive                       censored OR cost >= retrain cost
               min_budget_recovered              smallest budget that recovered (same model & strategy)
               is_best                           cheapest successful strategy for (model, budget)
usage: python -m analysis.build_dataset runs/toy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def build(run_dir: str):
    d = Path(run_dir)
    models = []
    for f in sorted((d / "backdoors").glob("*.json")):
        r = json.loads(f.read_text())
        r.pop("layer_scores", None)
        models.append(r)
    models = pd.DataFrame(models)
    rows = []
    for f in sorted(d.glob("recovery*.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            r.pop("trajectory", None); r["blocks"] = ",".join(r["blocks"])
            rows.append(r)
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("no recovery rows yet")
    runs = runs.merge(models, on="cell", how="left", suffixes=("", "_model"))

    runs["log_cost_steps"] = np.log1p(runs["cost_steps"])
    runs["log_cost_gpu_s"] = np.log1p(runs["cost_train_gpu_seconds"])
    rt = runs[runs.strategy == "full_retrain"][["cell", "budget", "cost_steps", "cost_train_gpu_seconds", "censored"]]
    rt = rt.rename(columns={"cost_steps": "retrain_steps", "cost_train_gpu_seconds": "retrain_gpu_s",
                            "censored": "retrain_censored"})
    runs = runs.merge(rt, on=["cell", "budget"], how="left")
    runs["cost_ratio_vs_retrain"] = runs["cost_train_gpu_seconds"] / runs["retrain_gpu_s"].replace(0, np.nan)
    runs["prohibitive"] = runs["censored"] | (runs["cost_train_gpu_seconds"] >= runs["retrain_gpu_s"])

    ok = runs[runs.recovered]
    mb = ok.groupby(["cell", "strategy"])["budget"].min().rename("min_budget_recovered").reset_index()
    runs = runs.merge(mb, on=["cell", "strategy"], how="left")
    best = ok.loc[ok.groupby(["cell", "budget"])["cost_train_gpu_seconds"].idxmin(), ["cell", "budget", "strategy"]]
    best = best.rename(columns={"strategy": "best_strategy"})
    runs = runs.merge(best, on=["cell", "budget"], how="left")
    runs["is_best"] = runs["strategy"] == runs["best_strategy"]

    out = d / "analysis"; out.mkdir(exist_ok=True)
    models.to_csv(out / "models.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    print(f"{len(models)} backdoored models, {len(runs)} recovery runs "
          f"({int(runs.recovered.sum())} recovered, {int(runs.censored.sum())} censored) -> {out}")
    return models, runs


if __name__ == "__main__":
    build(sys.argv[1])
