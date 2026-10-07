# ASR-G -> recovery cost: analysis of `runs/toy`

> **Toy-model smoke test.** These numbers validate the pipeline only; they are not evidence about real LVLMs and must not be reported as results.

20 backdoored models, 120 recovery runs, 99% recovered within the step cap.

## RQ1 — Recovery cost vs ASR-G

Median cost (censored runs enter at their cap, so medians are lower bounds) and recovery rate by ASR-G tertile:

| strategy     | asrg_tertile   |   n |   asr_g_mean |   recovery_rate |   median_steps |   median_gpu_s |
|:-------------|:---------------|----:|-------------:|----------------:|---------------:|---------------:|
| layer_reinit | low            |  13 |       0.0538 |           1     |             40 |           1.96 |
| layer_reinit | mid            |  13 |       0.746  |           1     |             40 |           1.74 |
| layer_reinit | high           |  14 |       0.957  |           1     |             40 |           1.72 |
| partial_ft   | low            |  14 |       0.1    |           1     |              0 |           0    |
| partial_ft   | mid            |  13 |       0.749  |           1     |             40 |           1.72 |
| partial_ft   | high           |  13 |       0.973  |           0.923 |             40 |           1.77 |

Spearman(ASR-G, log steps) = 0.491 (p = 3.69e-06, naive, ignores censoring & clustering).

| model                                                      |   beta_asr_g |   ci_lo |   ci_hi |   cost_multiplier_per_0.1_asrg |
|:-----------------------------------------------------------|-------------:|--------:|--------:|-------------------------------:|
| AFT log(steps) ~ ASR-G + strategy + log budget             |         2.12 |   1.86  |    2.34 |                           1.24 |
| AFT log(steps) ~ ASR-G + strategy + log budget + attack FE |         1.82 |  -0.137 |    2.49 |                           1.2  |

The attack-FE row is the claim-bearing estimate (within-attack variation only).

## RQ2 — Predicting recovery cost before recovery

Leave-one-attack-out on recovered runs (n = 79); target = log(1+steps). Constant-prediction MAE = 0.959.

| features                | model   |   n |   MAE_log |   within_2x |   spearman |
|:------------------------|:--------|----:|----------:|------------:|-----------:|
| asrg_only               | ridge   |  79 |     0.937 |       0.367 |     0.575  |
| asrg_only               | gbm     |  79 |     0.135 |       1     |     0.618  |
| attack_type_only        | ridge   |  79 |     0.938 |       0.506 |     0.0795 |
| attack_type_only        | gbm     |  79 |     0.581 |       0.848 |    -0.0735 |
| pre_recovery            | ridge   |  79 |     0.988 |       0.392 |     0.46   |
| pre_recovery            | gbm     |  79 |     0.228 |       0.962 |     0.575  |
| pre_recovery+probe      | ridge   |  79 |     0.988 |       0.392 |     0.46   |
| pre_recovery+probe      | gbm     |  79 |     0.228 |       0.962 |     0.575  |
| pre_recovery_minus_asrg | ridge   |  79 |     0.982 |       0.405 |     0.445  |
| pre_recovery_minus_asrg | gbm     |  79 |     0.418 |       0.911 |     0.44   |


Held-out MABA(-proxy) only (the proposal's generalisation test):

| features                |    gbm |   ridge |
|:------------------------|-------:|--------:|
| asrg_only               | 0.0241 |   0.681 |
| attack_type_only        | 0.0106 |   0.704 |
| pre_recovery            | 0.5    |   0.864 |
| pre_recovery+probe      | 0.5    |   0.864 |
| pre_recovery_minus_asrg | 0.5    |   0.886 |


Permutation importance (in-sample GBM; use for ranking only):

| feature          |   importance |       sd |
|:-----------------|-------------:|---------:|
| pre_asr_g        |     1.68     | 0.274    |
| s_partial_ft     |     1.17     | 0.232    |
| log_budget       |     0.00122  | 0.00086  |
| act_excess_max   |     0.000806 | 0.00018  |
| saliency_ratio   |     0.000775 | 9.73e-05 |
| pre_asr_in       |     0.000467 | 0.000188 |
| grad_alignment   |     0.000156 | 0.000111 |
| act_excess_depth |     0.000147 | 6.03e-05 |
| target_nll       |     5.84e-05 | 5.13e-05 |
| trigger_l2       |     5.46e-05 | 3.15e-05 |

## RQ3 — Most cost-effective strategy

Cheapest successful strategy per (model, budget), by ASR-G tertile:

| asrg_tertile   |   full_retrain |   layer_reinit |   partial_ft |
|:---------------|---------------:|---------------:|-------------:|
| low            |              0 |              0 |           14 |
| mid            |              1 |              7 |            6 |
| high           |              0 |              8 |            4 |


χ²(4) = 16.25, p = 0.0027 (does the best strategy depend on ASR-G tertile?)


| strategy     |   recovery_rate |   median_gpu_s_if_recovered |
|:-------------|----------------:|----------------------------:|
| full_retrain |           1     |                        5.21 |
| layer_reinit |           1     |                        1.8  |
| partial_ft   |           0.975 |                        1.68 |


Predicted-argmin policy (LOAO): oracle match = 0.60, picked strategy recovered = 1.00, mean regret = 0.00x extra cost. Baseline 'always partial_ft': mean regret = 0.10x.

## RQ4 — Does high-ASR-G MABA need proportionally more effort?

Elasticity d log(cost) / d log(ASR-G) = 0.202 [-0.183, 0.889] (1 = proportional, >1 = super-linear; AFT, censoring-aware).


maba_proxy cost relative to what its ASR-G predicts from the other attacks: x0.87 [0.54, 7.39] (1 = exactly as ASR-G predicts; censored MABA runs make this a lower bound).

## RQ5 — Is there a critical ASR-G threshold?

Segmented regression of log cost on ASR-G: breakpoint = 0.964 [0.411, 0.964]; BIC(segmented) - BIC(linear) = 4.94 (< -6 = strong evidence for a threshold).


'Prohibitive' = not recovered within the cap OR costlier than full retraining of the same model. ASR-G at which P(prohibitive) = 0.5:

| strategy     |   p_prohibitive |   asrg_at_p50 | note                                                             |
|:-------------|----------------:|--------------:|:-----------------------------------------------------------------|
| layer_reinit |           0.05  |           nan | crossing at -1.75 is outside observed ASR-G range (no threshold) |
| partial_ft   |           0.075 |           nan | crossing at 1.89 is outside observed ASR-G range (no threshold)  |
