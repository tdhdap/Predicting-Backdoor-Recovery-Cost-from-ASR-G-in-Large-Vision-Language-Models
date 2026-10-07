# Predicting Backdoor Recovery Cost from ASR-G in Large Vision-Language Models

Code, configurations, and analysis for a study of **how expensive it is to remove a backdoor from a
large vision-language model (LVLM)**, and whether that cost can be **predicted in advance** from the
attack's domain-generalization strength (**ASR-G**).

The work extends the CVPR 2025 paper *Revisiting Backdoor Attacks against Large Vision-Language
Models from Domain Shift* (Liang et al., [arXiv:2406.18844](https://arxiv.org/abs/2406.18844)),
which introduced ASR-G as a measure of how well a backdoor survives a train/test domain shift. Where
that paper **characterizes** attacks, this repository **measures the cost of undoing them** and turns
ASR-G from a threat descriptor into a remediation-planning tool.

---

## Why this matters

When an operator discovers a backdoor in a deployed LVLM, they face an economic decision the attack
and defense literatures do not answer: **what will it cost to fix, and is the model worth repairing
or better replaced?** Full retraining can cost thousands of GPU-hours; targeted repair is cheaper but
may not work. This project measures recovery cost systematically and shows that ASR-G—already
available at detection time—predicts it.

## Key findings (BLIP-2, 20 backdoored models, 120 recovery runs)

| Research question | Result |
|---|---|
| **RQ1 — Does cost rise with ASR-G?** | **Yes.** Recovery cost rises **≈2.9× per +0.1 ASR-G** (95% CI 1.4–6.9×), within attack, robust across three ASR-G definitions (1.96–2.57×). |
| **RQ2 — Is cost predictable before recovery?** | **Recoverability yes, magnitude partly.** A leave-one-attack-out classifier predicts recoverability at **AUC 0.79**; exact cost reaches Spearman 0.56. ASR-G beats attack identity and the ablation confirms its value. |
| **RQ3 — Which strategy is cheapest?** | **Depends on ASR-G (directionally).** Targeted repair is cheapest for low-ASR-G backdoors—often clearing them with *no* fine-tuning—while high-ASR-G ones need full retraining. A predict-then-choose policy cuts regret to 2.7× vs. 4.0× for always-retrain. |
| **RQ4 — Does the adaptive attack need disproportionate effort?** | **Inconclusive** at this scale (elasticity 3.0, CI includes 1; narrow ASR-G range for the proxy). |
| **RQ5 — Is there a critical ASR-G threshold?** | **Yes.** Recovery becomes prohibitive beyond **ASR-G ≈ 0.43** (ΔBIC = −15 for a threshold over a linear fit). |

Figures and tables backing these numbers are in [`runs/blip2_small/analysis/`](runs/blip2_small/analysis/).

> **Scope.** This is a deliberately compute-bounded, single-architecture (BLIP-2), single-seed study;
> confidence intervals are preliminary. Cross-architecture replication and the official MABA attack
> are future work. See the paper's Limitations / Threats to Validity.

---

## How the pipeline works

Per backdoored model, the runner:

1. **Plants** a backdoor (BadNets, Blended, WaNet, DualKey, or an EOT-optimized domain-agnostic
   **proxy for MABA**) at a given poisoning rate and trigger strength.
2. **Measures** attack success in-domain and across shifted domains → **ASR-G**, plus clean utility.
3. **Extracts pre-recovery features** (all measurable before any repair): trigger geometry and
   saliency, per-block activation divergence and its depth, clean-vs-backdoor gradient alignment,
   target NLL, and model descriptors.
4. **Recovers** the model under several strategies at nested clean-data budgets, recording cost
   (steps, clean samples, GPU-seconds) with **right-censoring** of runs that hit the step cap.
5. **Analyzes** RQ1–RQ5 with censoring-aware (AFT/Tobit) models, leave-one-attack-out prediction,
   strategy policies, and threshold detection.

**Recovery strategies:** `full_retrain` (replace baseline), `full_ft` (full fine-tuning), and
`layer_reinit` (activation-guided targeted repair). The strategy registry is a drop-in interface;
published defenses (fine-pruning, ANP, I-BAU) fit the same contract.

**A recovery "succeeds"** only when attack success falls below τ=0.05 in-domain *and* across shifted
domains *and* clean utility stays within δ=0.02 of the pre-recovery model—so destroying the model
does not count as a cheap fix.

---

## Repository layout

```
asrg-recovery/                     ← repository root (this folder)
├── asrg_recovery/                 ← Python package (the code you import)
│   ├── triggers.py                  attack triggers (BadNets/Blended/WaNet/DualKey/MABA-proxy)
│   ├── data.py                      datasets, domain shifts, poisoning, nested budgets
│   ├── attack.py                    backdoor injection
│   ├── features.py                  pre-recovery features + persistence probe
│   ├── metrics.py                   ASR per domain, ASR-G (3 variants), utility
│   ├── recovery.py                  recovery strategies + cost-to-recover measurement
│   ├── cost.py                      cost meter (steps, samples, GPU-s, FLOPs, memory)
│   ├── runner.py                    resumable, shardable grid runner
│   └── models/                      base interface + toy / blip2 / flamingo(+otter) adapters
├── configs/                       ← experiment configs (YAML)
│   ├── toy.yaml                     CPU smoke test (tests the code, not the hypothesis)
│   ├── blip2_small.yaml             the small study (20 models, 120 runs)
│   ├── config_frozen.yaml           the exact frozen config used for the released results
│   ├── blip2_full.yaml              full Tier-A grid (target extension)
│   └── tierB_openflamingo.yaml      reduced cross-architecture grid
├── analysis/                      ← dataset assembly + RQ1–RQ5 analysis
│   ├── build_dataset.py             recovery.jsonl + model JSONs → runs.csv / models.csv
│   └── rq_analysis.py               RQ1–RQ5 tables, figures, summary.md
├── scripts/                       ← check_adapter.py, make_slurm.py
├── tests/                         ← fast unit tests (pytest)
├── runs/blip2_small/analysis/     ← released results (CSVs, figures, summary.md)
├── colab_blip2_small.ipynb        ← Colab notebook for the small study
├── EXPERIMENT_DESIGN.md           ← hypotheses, factors, run counts, analysis plan, validity
├── SMALL_EXPERIMENT.md            ← the compute-bounded design + estimates
├── COLAB_CHANGES.md               ← notes on the Colab run
├── requirements.txt
└── README.md
```

> **Note on the two similarly-named folders:** `asrg-recovery` (hyphen) is the repository;
> `asrg_recovery` (underscore) is the importable Python package inside it. Python package names
> cannot contain hyphens—this is standard (as `scikit-learn` contains `sklearn`).

---

## Installation

```bash
git clone https://github.com/<you>/asrg-recovery.git
cd asrg-recovery
pip install -r requirements.txt
```

Core dependencies: PyTorch, NumPy, scikit-learn, pandas, SciPy, PyYAML, matplotlib. Real LVLMs
additionally need `transformers`, `accelerate`, and `pillow`; OpenFlamingo/Otter need their own
packages (see `requirements.txt`).

## Quick start — reproduce the analysis (no GPU, no weights)

The released run outputs are included, so you can regenerate every table and figure from them:

```bash
python -m analysis.build_dataset runs/blip2_small
python -m analysis.rq_analysis   runs/blip2_small
# → runs/blip2_small/analysis/summary.md  + fig1/fig2/fig3 + rq*.csv
```

## Quick start — pipeline smoke test (CPU)

A tiny synthetic captioner exercises the whole pipeline end-to-end on CPU. **Its numbers are for
testing the code, not evidence about real LVLMs** (the analysis stamps a warning on toy runs):

```bash
python -m pytest tests -q
python -m asrg_recovery.runner configs/toy.yaml --device cpu
python -m analysis.build_dataset runs/toy && python -m analysis.rq_analysis runs/toy
```

## Running the real study

The small study (`configs/blip2_small.yaml` / the released `configs/config_frozen.yaml`) targets
BLIP-2 and takes roughly **~45 A100-hours** (≈2 days on one A100). Before launching:

1. Verify the adapter loads: `python scripts/check_adapter.py --model blip2`.
2. Build a data manifest (`data/manifest.jsonl`) with COCO in-domain and the four shifted domains
   (NoCaps-out, sketch, painting, cartoon); the schema is documented in `asrg_recovery/data.py`.
   **Datasets and model weights are not distributed here.**
3. See `SMALL_EXPERIMENT.md` for the design and compute budget, and `EXPERIMENT_DESIGN.md` for the
   full protocol, hypotheses, and threats to validity. For multi-GPU clusters, `scripts/make_slurm.py`
   emits a SLURM array that shards the grid.

---

## Results and artifacts

- `runs/blip2_small/analysis/summary.md` — the auto-generated RQ1–RQ5 write-up.
- `runs/blip2_small/analysis/runs.csv`, `models.csv` — per-run and per-model tables.
- `fig1_cost_vs_asrg.png`, `fig2_loao_pred_vs_actual.png`, `fig3_prohibitive_vs_asrg.png` — the three
  headline figures.

## Ethics and responsible disclosure

This project studies the **remediation** of backdoor attacks. All attacks used are previously
published; no new attack capability is introduced. Experiments use self-trained models, public
captioning data, and a benign placeholder target string. **No poisoned or backdoored model
checkpoints are released** (the `.gitignore` enforces this), consistent with the paper's ethics
statement; only measurement and analysis code and aggregate results are shared.

## Citation

If you use this code or the findings, please cite this work and the paper it extends:

```bibtex
@misc{asrg_recovery_cost,
  title  = {Predicting Backdoor Recovery Cost from ASR-G in Large Vision-Language Models},
  author = {<your name>},
  year   = {2026},
  note   = {https://github.com/<you>/asrg-recovery}
}

@inproceedings{liang2025revisiting,
  title     = {Revisiting Backdoor Attacks against Large Vision-Language Models from Domain Shift},
  author    = {Liang, Siyuan and Liang, Jiawei and Pang, Tianyu and Du, Chao and Liu, Aishan and Zhu, Mingli and Cao, Xiaochun and Tao, Dacheng},
  booktitle = {CVPR},
  year      = {2025},
  note      = {arXiv:2406.18844}
}
```

## License

No license is set yet. Until one is added, default copyright applies (others may view but not
reuse). Adding an [MIT](https://choosealicense.com/licenses/mit/) or
[Apache-2.0](https://choosealicense.com/licenses/apache-2.0/) license is recommended for research
code.

## Acknowledgements

Builds directly on Liang et al.'s ASR-G formulation and attack taxonomy, and on the broader backdoor
attack/defense literature cited in the paper (BadNets, Blended, WaNet, DualKey, Fine-Pruning, NAD,
ANP, RNP, I-BAU, REFINE, and the mitigation-difficulty analysis of Verma et al.).
```
