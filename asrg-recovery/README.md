# asrg-recovery

Experiment code for **"Predicting backdoor recovery cost from ASR-G"**. It extends the
CVPR 2025 ASR-G metric from threat characterisation to remediation cost in LVLMs
(BLIP-2, OpenFlamingo, Otter).

The pipeline does four things:
1. **Plants** backdoors (BadNets, Blended, WaNet, DualKey, MABA) across a grid of poison
   rates and trigger strengths, so ASR-G varies both *between* and *within* attacks.
2. **Measures** ASR in-domain and under domain shift (ASR-G), plus pre-recovery features:
   trigger, embedding/activation, gradient alignment, saliency and model descriptors.
3. **Recovers** each model with five strategies at nested clean-data budgets, recording
   steps, GPU-hours, data used and censoring.
4. **Analyses** RQ1–RQ5 with censoring-aware (AFT/Tobit) models, leave-one-attack-out
   prediction (MABA held out), strategy policies and threshold tests.

Read **`EXPERIMENT_DESIGN.md`** first. It holds the hypotheses, factors, run counts, pilot
plan, analysis plan and threats to validity.

## Layout

```
asrg_recovery/
  triggers.py      BadNets, Blended, WaNet, DualKey, MABAProxy (adaptive, EOT-optimised)
  data.py          toy data + JSONL manifest loader, domain shifts, poisoning, nested budgets
  models/          one adapter interface; toy (CPU), blip2, openflamingo, otter
  attack.py        backdoor injection (+ joint trigger optimisation for adaptive attacks)
  metrics.py       ASR per domain, ASR-G (3 variants), clean utility
  features.py      pre-recovery features + optional persistence probe
  recovery.py      full_retrain / full_ft / partial_ft / layer_repair / layer_reinit
  cost.py          cost meter (steps, samples, GPU-s, FLOPs proxy, memory)
  runner.py        resumable, shardable grid runner
analysis/
  build_dataset.py merge outputs -> models.csv, runs.csv (+ derived cost targets)
  rq_analysis.py   RQ1-RQ5 tables, figures, summary.md
configs/
  toy.yaml                  CPU smoke test
  blip2_full.yaml           Tier A (120 backdoored models, 1,800 recovery runs)
  tierB_openflamingo.yaml   Tier B reduced grid (copy for Otter)
scripts/
  check_adapter.py          5-minute sanity check per real model
  make_slurm.py             SLURM array launcher
```

## Quick start

```bash
pip install -r requirements.txt

# 1) smoke test on CPU (about 1-2 h on 2 cores; tests the code, NOT the hypothesis)
python -m asrg_recovery.runner configs/toy.yaml --device cpu
python -m analysis.build_dataset runs/toy
python -m analysis.rq_analysis runs/toy          # -> runs/toy/analysis/summary.md + figures

# 2) real models
python scripts/check_adapter.py --model blip2
#    build data/manifest.jsonl (schema in asrg_recovery/data.py), run the pilot (design §6),
#    freeze the config, then:
python scripts/make_slurm.py configs/blip2_full.yaml --shards 40 --stage inject  > inject.sbatch
python scripts/make_slurm.py configs/blip2_full.yaml --shards 40 --stage recover > recover.sbatch
```

## Status and honesty notes

- **Tested here:** the toy adapter and the whole pipeline (inject → features → probe →
  recover → analysis) ran end to end on CPU.
- **Not executed here:** the BLIP-2, OpenFlamingo and Otter adapters (no GPU or weights in the
  sandbox). They are written against `transformers` and `open_flamingo` 2.x. Run
  `scripts/check_adapter.py` first and expect small API fixes. BLIP-2 in recent
  `transformers` versions may want image placeholder tokens in `input_ids`.
- **MABA is a proxy:** `MABAProxy` is an adaptive, domain-agnostic (EOT-optimised)
  perturbation, not the official MABA. Swap in the original before reporting MABA results.
- **ASR-G definition:** set `eval.asrg_definition` and the shifted domains to match the
  CVPR 2025 paper. All three variants are logged regardless.
- **Toy numbers are not results.** `summary.md` is stamped accordingly.
