# Experiment design: does ASR-G predict backdoor recovery cost in LVLMs?

This document fixes the protocol before any real run. Everything here is implemented in
`asrg_recovery/` (experiments) and `analysis/` (statistics); section numbers are referenced
from the code and configs.

---

## 1. Hypotheses (one per RQ, each with a falsification rule)

| RQ | Hypothesis | Supported if | Refuted if |
|---|---|---|---|
| RQ1 | Recovery cost rises with ASR-G | AFT coefficient on ASR-G **with attack fixed effects** > 0, 95% cluster-bootstrap CI excludes 0 | CI includes 0 or is negative |
| RQ2 | Cost is predictable from pre-recovery features, and ASR-G is a dominant feature | Leave-one-attack-out (LOAO) MAE on log cost beats the constant and attack-type-only baselines; removing ASR-G increases LOAO MAE; ASR-G in top-3 permutation importance | Pre-recovery model does not beat attack-type-only baseline |
| RQ3 | The cheapest strategy depends on ASR-G (targeted repair for low ASR-G; heavier strategies for high ASR-G) | Best-strategy distribution differs across ASR-G tertiles (χ² test) and the predicted-argmin policy has lower regret than the best fixed strategy | One strategy is cheapest in every tertile |
| RQ4 | MABA needs *disproportionately* more effort than its ASR-G alone implies | Elasticity d log cost / d log ASR-G > 1 (CI excludes 1), or MABA's excess cost over the ASR-G prediction fitted on the other attacks has CI above ×1 | Elasticity CI includes 1 and MABA excess CI includes ×1 (= cost is *proportional*; still an interesting finding) |
| RQ5 | There is an ASR-G level beyond which recovery is prohibitive | Segmented model beats linear (ΔBIC < −6) **and** P(prohibitive) crosses 0.5 inside the observed ASR-G range with a bounded CI | ΔBIC ≥ −2 and logistic threshold outside the observed range |

The proposal's "success criterion" (train on BadNets/WaNet/Blended/DualKey, test on MABA)
is the `held_out = maba_proxy` row of RQ2 and is reported separately.

## 2. Definitions

**ASR (domain d).** Fraction of triggered inputs from domain *d*, excluding samples whose
clean caption already equals the target, whose generated output contains the target string.

**ASR-G.** Aggregate of ASR over the *shifted* evaluation domains. The code logs three
variants and uses `eval.asrg_definition` as the primary. **Set it to match the CVPR 2025
paper's definition exactly** before the main grid; the others are sensitivity analyses.
- `mean_shifted` (default): mean ASR over shifted domains
- `ratio`: mean_shifted / ASR_in — how much of the in-domain success survives the shift
- `min_shifted`: worst-case domain

**Recovered.** At an evaluation checkpoint, all three hold:
`ASR_in ≤ τ` and `mean-shifted ASR ≤ τ` and `utility_in ≥ utility_before − δ`
(τ = 0.05, δ = 0.02). The utility floor is essential: without it, the cheapest
"recovery" is destroying the model.

**Cost measures** (all logged for every run; `cost.py`):
steps, clean samples seen, optimisation-only GPU-seconds (× #GPUs), approximate FLOPs
(6·N_active·tokens), peak memory. **Primary cost = steps** (hardware-independent);
GPU-hours are the practitioner-facing secondary. Evaluation time during recovery is
logged separately and excluded, because checkpoint density is an experimental
artefact, not a defender cost.

**Clean data required.** Recovery runs at nested budgets (each smaller budget is a
subset of the larger). Data required = smallest budget that recovers for that model and
strategy (`min_budget_recovered`). Models that never recover are censored at the
largest budget.

**Censoring.** A run that hits `max_steps` without recovering is right-censored: its true
cost is ≥ the cap. All cost regressions use a log-normal accelerated-failure-time (Tobit)
likelihood that treats censored runs as lower bounds. Never drop them: dropping would
remove exactly the most expensive, high-ASR-G cases and bias RQ1 toward zero.

**Prohibitive.** A strategy's recovery is prohibitive for a model if it is censored or if
it costs at least as much as full retraining of the same model at the same budget.

## 3. Factors and levels

| Factor | Levels | Role |
|---|---|---|
| Attack | BadNets, Blended, WaNet, DualKey, MABA | Between-attack ASR-G variation; RQ4 contrast |
| Poison rate | 0.5%, 1%, 5%, 10% | **Within-attack** ASR-G variation |
| Trigger strength | 2 per attack (patch size / α / warp / ε) | **Within-attack** ASR-G variation |
| Seed | 3 | Replication; cluster bootstrap unit together with the other factors |
| Model | BLIP-2 (Tier A, full grid); OpenFlamingo-3B, Otter (Tier B, reduced grid) | Architecture dependence |
| Recovery strategy | full_retrain, full_ft, partial_ft, layer_repair, layer_reinit | RQ3 |
| Clean-data budget | 1k, 5k, 20k samples (nested) | Data-requirement target |

**Run counts.**
Tier A: 5 × 4 × 2 × 3 = **120 backdoored models** → × 5 strategies × 3 budgets = **1,800 recovery runs**.
Tier B (per architecture): 5 × 2 × 2 × 2 = **40 models** → × 3 strategies × 2 budgets = **240 runs**.
Total: 200 backdoored models and 2,280 recovery runs.

**Why the within-attack factors matter.** If ASR-G only varied *between* attacks, a
positive ASR-G→cost correlation could just mean "MABA is expensive for some other
reason". Poison rate and trigger strength move ASR-G *inside* each attack, so the
attack-fixed-effects model (§7, RQ1) can separate "high ASR-G" from "is MABA". The
pilot (§6) must confirm that each attack spans a usable ASR-G range. If one does not,
add levels (e.g. lower poison rates) before the main grid.

## 4. Data

Use three disjoint pools so no sample is reused across roles:
- `attacker_train`: used for injection only (poisoned at the chosen rate)
- `defender_pool`: the only data recovery may use; budgets are nested subsets
- `eval/<domain>`: held out; one in-domain set plus the shifted domains that define ASR-G

In-domain: COCO Captions (Karpathy split). Shifted domains: **reuse the exact shifts
from the CVPR 2025 ASR-G paper** so ASR-G values are comparable. `configs/blip2_full.yaml`
lists placeholders (`nocaps_out, sketch, painting, cartoon`). Put everything in one
manifest JSONL (`data.py` documents the schema).

Recovery data are always in-domain and clean. The recovery-data domain is a possible
extension factor, not part of the main design.

## 5. Protocol per backdoored model (`runner.py`)

1. **Reset** the injectable parameters to the clean pre-attack base.
2. **Inject** (`attack.py`): fine-tune the injectable scope (BLIP-2: Q-Former + query
   tokens + projection; Flamingo/Otter: Perceiver + gated cross-attention) on
   `attacker_train` with poison rate *p*. The adaptive trigger (MABA) is optimised
   jointly with EOT over domain transforms. Attacker cost is logged.
3. **Evaluate** ASR per domain, ASR-G variants and clean utility (`metrics.py`). Models
   whose in-domain ASR is below 0.5 stay in the dataset, because low ASR is part of
   the variation. Report them.
4. **Featurise** (`features.py`), with no recovery performed:
   - attack: class, poison rate, ASR_in, ASR-G (+variants)
   - trigger: area, L2/L∞, distinctiveness in the *clean* base's feature space, saliency
     share on the trigger region
   - embedding: per-block activation divergence (triggered vs clean), in excess of the
     clean base's divergence; depth and spread of that excess; cosine between the
     clean-data gradient and the backdoor gradient; target NLL
   - model: family, parameter counts, number of repairable blocks
5. **Probe** (optional): 50 steps of clean fine-tuning to measure how much ASR survives.
   This *is* a small intervention, so its cost is logged and RQ2 reports feature sets
   with and without it. The "predict before recovery" claim rests on the probe-free set.
6. **Recover** under every strategy × budget (`recovery.py`), starting from identical
   weights, evaluating every `eval_every` steps until recovered or `max_steps`.
   `layer_repair` and `layer_reinit` choose blocks from the step-4 activation excess.
   `partial_ft` uses the last *k* blocks, with the same *k*, so these two strategies
   differ only in *which* blocks are touched.

## 6. Pilot (must be finished and frozen before the main grid)

On BLIP-2, 1 seed, poison rates {1%, 10%}, all 5 attacks, 1 strength (10 models):
1. **Injection epochs / lr**: smallest setting that reaches ASR_in ≥ 0.9 at 10% poison
   for every attack without dropping clean utility by more than δ.
2. **Recovery lr**: tuned on a *clean* fine-tuned control model, choosing the largest
   lr that keeps utility within δ over `max_steps`. Tuning it on backdoored models would
   leak the outcome into the protocol.
3. **max_steps**: large enough that ≤ 30% of pilot runs are censored. Heavy censoring
   leaves the AFT model to extrapolate.
4. **ASR-G spread**: verify each attack covers at least a 0.3-wide ASR-G range across
   poison rate × strength. Add levels if not.
5. **Timing**: mean GPU-hours per recovery run *h̄* gives the budget: Tier A ≈ 1,800·h̄ +
   120·(injection + features + probe). Freeze the config (commit hash in the paper).

## 7. Analysis plan (`analysis/rq_analysis.py`)

Unit = recovery run. Clustering unit = backdoored model. All CIs use a cluster bootstrap
(500 resamples of models). Primary outcome = log(1 + steps).

- **RQ1.** Tertile table (median cost, recovery rate), naive Spearman, and a censored AFT
  model with covariates `ASR-G + strategy + log budget`, fitted without and **with attack
  fixed effects**. Report the effect as a cost multiplier per +0.1 ASR-G.
- **RQ2.** LOAO cross-validation with ridge and gradient-boosted trees. Feature sets:
  ASR-G only, attack type only, all pre-recovery features, pre-recovery + probe, and
  pre-recovery minus ASR-G (ablation). Metrics: MAE on log cost, share within 2× of
  truth, Spearman. Also fit a recoverable-within-cap classifier (LOAO AUC) so censored
  runs contribute. Rank features by permutation importance.
- **RQ3.** Cheapest successful strategy per (model, budget) by ASR-G tertile. Test a
  policy that picks the strategy with the lowest predicted cost (LOAO) against the
  oracle and against the best fixed strategy. Censored picks are penalised at 2× the cap.
- **RQ4.** Elasticity of cost with respect to ASR-G (log-log AFT). Also MABA's excess
  cost: its observed log cost minus an ASR-G-only model fitted on the other four attacks.
- **RQ5.** Segmented regression (breakpoint search plus ΔBIC against linear), and a
  per-strategy logistic P(prohibitive | ASR-G) with the 0.5 crossing and bootstrap CI.

Sensitivity analyses (report in the appendix): every ASR-G variant; GPU-seconds instead
of steps as outcome; τ ∈ {0.01, 0.10}; δ ∈ {0.01, 0.05}; Tier-B architectures separately.

Multiple comparisons: RQ1 and RQ4 are the confirmatory tests (Holm-corrected). Everything
else is reported as exploratory.

## 8. Threats to validity and mitigations

| Threat | Mitigation |
|---|---|
| ASR-G confounded with attack identity | Within-attack factors; attack-FE estimate is the claim-bearing one |
| ASR-G ceiling (ASR_in near 1 for every attack) | Low poison rates in the grid; `ratio` variant as a sensitivity check |
| MABA reproduced as a proxy | Replace `MABAProxy` with the official implementation; keep the proxy as a labelled extra attack |
| Recovery lr favours some strategies | lr tuned once on a clean control and frozen for all strategies |
| Cost measured with evaluation overhead | Optimisation time only; eval time logged separately |
| Hardware noise in GPU-hours | Steps are primary; one pinned GPU type; hardware logged |
| Censoring | AFT likelihood; censoring rate reported per cell; cap set in the pilot |
| "Recovery" by breaking the model | Utility floor in the success definition |
| Probe feature leaks the outcome | Probe-free feature set is primary; probe cost reported |
| Only fine-tuning-family defences | Scope stated explicitly; the strategy registry makes it easy to add unlearning / pruning defences (e.g. ANP, I-BAU) as further strategies |

## 9. Toy smoke test

`configs/toy.yaml` runs the entire protocol on a synthetic coloured-shape captioner on
CPU. Real LVLMs run the same code paths. The toy run exists to debug the code and
exercise the analysis. **Its numbers must never appear as results.** The analysis
summary stamps a warning on toy runs.

## 10. Swap-in checklist before real runs

- [ ] Official MABA implementation, registered in `triggers.REGISTRY`
- [ ] ASR-G definition and shifted domains matched to the CVPR 2025 paper
- [ ] Data manifest built; the three pools checked for disjointness (image hashes)
- [ ] CIDEr as the utility metric (pycocoevalcap) instead of token-F1 for the paper tables
- [ ] `scripts/check_adapter.py` passes for each model
- [ ] Pilot finished; config frozen and committed
