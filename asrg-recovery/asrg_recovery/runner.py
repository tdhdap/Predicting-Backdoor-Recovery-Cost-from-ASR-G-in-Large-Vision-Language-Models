"""Grid runner: inject -> measure -> featurise -> recover (x strategies x budgets).

Outputs (under cfg.out_dir), all resumable:
  base.pt                               clean pre-attack injectable weights
  backdoors/<cell>.pt                   backdoored weights + trigger state
  backdoors/<cell>.json                 pre-recovery metrics + features  (one row per backdoored model)
  recovery.jsonl                        one row per recovery run         (cell x strategy x budget)

Use --shard i/n to split cells across jobs (e.g. SLURM array), each job appends to its
own recovery.<i>.jsonl; analysis/build_dataset.py merges them.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch
import yaml

from .attack import inject_backdoor, pretrain_clean
from .data import load_jsonl_splits, make_toy_splits
from .features import compute_features, persistence_probe
from .metrics import evaluate
from .models import build_adapter
from .recovery import run_recovery
from .triggers import build_trigger, load_trigger, trigger_state


def cell_id(c: dict) -> str:
    s = json.dumps(c, sort_keys=True)
    return f"{c['model']}_{c['attack']}_p{c['poison_rate']}_s{c['strength']}_seed{c['seed']}_" + hashlib.md5(s.encode()).hexdigest()[:6]


def enumerate_cells(cfg):
    g = cfg["grid"]
    cells = []
    for attack in g["attacks"]:
        for pr, strength, seed in itertools.product(g["poison_rates"], g["strengths"][attack], g["seeds"]):
            cells.append({"model": cfg["model"]["name"], "attack": attack, "poison_rate": pr,
                          "strength": strength, "seed": seed,
                          "strength_level": g["strengths"][attack].index(strength)})
    return cells


def build_splits(cfg):
    d = cfg["data"]
    if d["source"] == "toy":
        return make_toy_splits(d["image_size"], d["n_attacker"], d["n_defender"], d["n_eval"], tuple(d["eval_domains"]),
                               seed=d.get("seed", 0))
    return load_jsonl_splits(d["path"], d["image_size"], d["in_domain"], d["eval_domains"])


def toy_vocab(target):
    from .data import TOY_COLORS, TOY_SHAPES, TOY_PROMPT
    return list(TOY_COLORS) + TOY_SHAPES + ["a", "cf"] + TOY_PROMPT.split() + target.split()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stage", choices=["all", "inject", "recover"], default="all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    si, sn = map(int, a.shard.split("/"))
    out = Path(cfg["out_dir"]); (out / "backdoors").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    target = cfg["attack"]["target"]
    splits = build_splits(cfg)
    ec = cfg["eval"]
    evalkw = dict(n=ec["n"], bs=ec["bs"], exact_utility=ec["exact_utility"], asrg_definition=ec["asrg_definition"])

    adapter = build_adapter(cfg["model"], a.device, vocab_words=toy_vocab(target))
    base_path = out / "base.pt"
    if base_path.exists():
        adapter.load_state_dict(torch.load(base_path))
    else:
        if cfg["model"]["name"] == "toy":
            pc = cfg["model"]["pretrain"]
            pretrain_clean(adapter, splits["defender_pool"], pc["steps"], pc["lr"], pc["bs"])
        torch.save(adapter.state_dict(), base_path)
    base_sd = adapter.state_dict()

    cells = [c for i, c in enumerate(enumerate_cells(cfg)) if i % sn == si]
    rec_path = out / (f"recovery.{si}.jsonl" if sn > 1 else "recovery.jsonl")
    done = set()
    if rec_path.exists():
        for l in rec_path.read_text().splitlines():
            r = json.loads(l); done.add((r["cell"], r["strategy"], r["budget_requested"]))

    for c in cells:
        cid = cell_id(c)
        ck, js = out / "backdoors" / f"{cid}.pt", out / "backdoors" / f"{cid}.json"
        isz = cfg["data"]["image_size"]
        if not js.exists():
            if a.stage == "recover":
                continue
            t0 = time.time()
            adapter.load_state_dict(base_sd)
            trig = build_trigger(c["attack"], isz, c["strength"], seed=c["seed"])
            ac = cfg["attack"]
            inj_cost = inject_backdoor(adapter, splits["attacker_train"], trig, target, c["poison_rate"],
                                       epochs=ac["epochs"], lr=ac["lr"], bs=ac["bs"], seed=c["seed"],
                                       eot_domains=ac.get("eot_domains", []), device=a.device)
            bd_sd = adapter.state_dict()
            pre = evaluate(adapter, splits["eval"], trig, target, cfg["data"]["in_domain"], **evalkw)
            feats = compute_features(adapter, base_sd, bd_sd, splits, trig, target, cfg)
            probe = persistence_probe(adapter, bd_sd, splits, trig, target, cfg) if cfg["features"]["probe"]["enabled"] else {}
            if probe:
                probe["probe_persist_in"] = probe["probe_asr_in"] / max(pre["asr_in"], 1e-6)
                probe["probe_persist_g"] = probe["probe_asr_g"] / max(pre["asr_g"], 1e-6)
            torch.save({"state": bd_sd, "trigger": trigger_state(trig)}, ck)
            row = {"cell": cid, **c, **{f"pre_{k}": v for k, v in pre.items()}, **feats, **probe,
                   **{f"inject_{k}": v for k, v in inj_cost.items()}, "wall_s": time.time() - t0}
            js.write_text(json.dumps(row, indent=1))
            print(f"[inject] {cid} asr_in={pre['asr_in']:.2f} asr_g={pre['asr_g']:.2f} util={pre['util_in']:.2f}", flush=True)

        if a.stage == "inject":
            continue
        row = json.loads(js.read_text())
        blob = torch.load(ck)
        bd_sd, trig = blob["state"], load_trigger(blob["trigger"], isz)
        pre = {k[4:]: v for k, v in row.items() if k.startswith("pre_")}
        for strat, budget in itertools.product(cfg["recovery"]["strategies"], cfg["recovery"]["budgets"]):
            if (cid, strat, budget) in done:
                continue
            r = run_recovery(adapter, base_sd, bd_sd, splits, trig, target, strat, budget, pre,
                             row["layer_scores"], cfg, seed=c["seed"])
            r.update({"cell": cid, "budget_requested": budget})
            with rec_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"[recover] {cid} {strat:12s} n={budget:5d} recovered={r['recovered']} "
                  f"steps={r['cost_steps']} gpu_s={r['cost_train_gpu_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
