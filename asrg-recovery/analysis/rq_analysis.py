"""Pre-specified analyses for RQ1-RQ5.

usage: python -m analysis.rq_analysis runs/<name>      (after analysis.build_dataset)
writes runs/<name>/analysis/{rq*.csv, fig*.png, summary.md}

Conventions
  * Unit of analysis = recovery run; clustering unit = backdoored model ("cell").
    All CIs are cluster bootstraps over cells (B = 500 by default).
  * Censoring: runs that never recovered within max_steps are right-censored. Effects
    on cost are estimated with a log-normal accelerated-failure-time model (Tobit on
    log cost) so censored runs count as "at least this expensive" instead of being
    dropped or treated as exact.
  * ASR-G vs attack identity: every effect of ASR-G is reported (a) alone and (b) with
    attack fixed effects, where it is identified only from WITHIN-attack variation
    (poison rate, trigger strength, seed). (b) is the claim-bearing estimate.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")
B = 500
RNG = np.random.default_rng(0)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order

PRE = ["pre_asr_in", "pre_asr_g", "poison_rate", "trigger_area_frac", "trigger_l2", "trigger_linf",
       "trigger_has_text", "trigger_input_dependent", "trigger_distinctiveness", "saliency_ratio",
       "act_excess_max", "act_excess_mean", "act_excess_depth", "act_excess_spread", "grad_alignment",
       "target_nll", "n_injectable_params", "n_repairable_blocks"]
PROBE = ["probe_persist_in", "probe_persist_g"]


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------

def design(df, cols, strat=True, attack_fe=False):
    X = df[cols].astype(float).copy() if cols else pd.DataFrame(index=df.index)
    if strat:
        X = X.join(pd.get_dummies(df["strategy"], prefix="s", drop_first=True, dtype=float))
        X["log_budget"] = np.log(df["budget"].astype(float))
    if attack_fe:
        X = X.join(pd.get_dummies(df["attack"], prefix="a", drop_first=True, dtype=float))
    X = X.loc[:, X.std() > 0] if len(X) > 1 else X
    X.insert(0, "const", 1.0)
    return X


def aft_fit(X, y, event):
    """Log-normal AFT / Tobit: y = X b + s*e, right-censored where event == 0."""
    Xv, yv, ev = X.values, y.values.astype(float), event.values.astype(float)
    b0 = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    th0 = np.r_[b0, np.log(yv.std() + 1e-3)]

    def nll(th):
        b, s = th[:-1], np.exp(th[-1])
        z = (yv - Xv @ b) / s
        return -(ev * (stats.norm.logpdf(z) - np.log(s)) + (1 - ev) * stats.norm.logsf(z)).sum()

    r = optimize.minimize(nll, th0, method="L-BFGS-B")
    return pd.Series(r.x[:-1], index=X.columns), float(np.exp(r.x[-1]))


def cluster_boot(df, fn, b=B):
    cells = df["cell"].unique()
    g = {c: d for c, d in df.groupby("cell")}
    out = []
    for _ in range(b):
        samp = RNG.choice(cells, len(cells), replace=True)
        try:
            out.append(fn(pd.concat([g[c] for c in samp], ignore_index=True)))
        except Exception:
            pass
    out = np.array([o for o in out if np.isfinite(o)])
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if len(out) > 20 else (np.nan, np.nan)


def fmt_ci(est, ci):
    return f"{est:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160); import matplotlib.pyplot as plt; plt.close(fig)


# ------------------------------------------------------------------------------------------------
# RQ1: computational cost vs ASR-G
# ------------------------------------------------------------------------------------------------

def rq1(runs, out, md):
    md.append("## RQ1 — Recovery cost vs ASR-G\n")
    r = runs[runs.strategy != "full_retrain"].copy()
    r["asrg_tertile"] = pd.qcut(r["pre_asr_g"].rank(method="first"), 3, labels=["low", "mid", "high"])
    tab = r.groupby(["strategy", "asrg_tertile"]).agg(
        n=("cell", "size"), asr_g_mean=("pre_asr_g", "mean"), recovery_rate=("recovered", "mean"),
        median_steps=("cost_steps", "median"), median_gpu_s=("cost_train_gpu_seconds", "median")).reset_index()
    tab.to_csv(out / "rq1_tertiles.csv", index=False)
    md.append("Median cost (censored runs enter at their cap, so medians are lower bounds) and recovery "
              "rate by ASR-G tertile:\n\n" + tab.to_markdown(index=False, floatfmt=".3g") + "\n")

    rows = []
    for fe in (False, True):
        X = design(r, ["pre_asr_g"], attack_fe=fe)
        coef, s = aft_fit(X, r["log_cost_steps"], r["recovered"])
        ci = cluster_boot(r, lambda d: aft_fit(design(d, ["pre_asr_g"], attack_fe=fe), d["log_cost_steps"],
                                               d["recovered"])[0].get("pre_asr_g", np.nan))
        rows.append({"model": "AFT log(steps) ~ ASR-G + strategy + log budget" + (" + attack FE" if fe else ""),
                     "beta_asr_g": coef.get("pre_asr_g", np.nan), "ci_lo": ci[0], "ci_hi": ci[1],
                     "cost_multiplier_per_0.1_asrg": np.exp(0.1 * coef.get("pre_asr_g", np.nan))})
    res = pd.DataFrame(rows); res.to_csv(out / "rq1_aft.csv", index=False)
    rho, p = stats.spearmanr(r["pre_asr_g"], r["log_cost_steps"])
    md.append(f"Spearman(ASR-G, log steps) = {rho:.3f} (p = {p:.3g}, naive, ignores censoring & clustering).\n\n"
              + res.to_markdown(index=False, floatfmt=".3g")
              + "\n\nThe attack-FE row is the claim-bearing estimate (within-attack variation only).\n")

    import matplotlib.pyplot as plt
    strats = sorted(r.strategy.unique())
    fig, axes = plt.subplots(1, len(strats), figsize=(4 * len(strats), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    attacks = sorted(runs.attack.unique())
    for ax, st in zip(axes, strats):
        d = r[r.strategy == st]
        for i, a in enumerate(attacks):
            da = d[d.attack == a]
            ok, cen = da[da.recovered], da[~da.recovered]
            ax.scatter(ok.pre_asr_g, ok.cost_steps, s=36, color=COLORS[i % 6], label=a, edgecolor="white", lw=1)
            ax.scatter(cen.pre_asr_g, cen.cost_steps, s=36, facecolor="none", edgecolor=COLORS[i % 6], lw=1.5)
        ax.set_title(st, fontsize=10); ax.set_xlabel("ASR-G (pre-recovery)")
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("steps to recover (hollow = censored)")
    axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    savefig(fig, out / "fig1_cost_vs_asrg.png")


# ------------------------------------------------------------------------------------------------
# RQ2: predictability (leave-one-attack-out)
# ------------------------------------------------------------------------------------------------

def rq2(runs, out, md):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    md.append("## RQ2 — Predicting recovery cost before recovery\n")
    r = runs[(runs.strategy != "full_retrain") & runs.recovered].copy()
    cls = pd.get_dummies(runs["attack_class"], prefix="cls", dtype=float)
    r = r.join(cls)
    probe = [c for c in PROBE if c in r]
    sets = {"asrg_only": ["pre_asr_g"], "attack_type_only": list(cls.columns),
            "pre_recovery": [c for c in PRE if c in r], "pre_recovery+probe": [c for c in PRE if c in r] + probe,
            "pre_recovery_minus_asrg": [c for c in PRE if c in r and c not in ("pre_asr_g",)]}
    models = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
              "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05,
                                                           min_samples_leaf=5)}
    y = r["log_cost_steps"].values
    rows, preds = [], []
    for sname, cols in sets.items():
        X = design(r, cols).drop(columns="const")
        for mname, mk in models.items():
            yhat = np.full(len(r), np.nan)
            for a in r.attack.unique():  # leave-one-attack-out
                te = (r.attack == a).values
                if te.all() or (~te).sum() < 5:
                    continue
                m = mk().fit(X[~te], y[~te]); yhat[te] = m.predict(X[te])
            ok = np.isfinite(yhat)
            err = np.abs(yhat[ok] - y[ok])
            rows.append({"features": sname, "model": mname, "n": int(ok.sum()), "MAE_log": err.mean(),
                         "within_2x": float((err <= np.log(2)).mean()),
                         "spearman": stats.spearmanr(yhat[ok], y[ok])[0] if ok.sum() > 2 else np.nan})
            for a in r.attack.unique():
                te = (r.attack == a).values & ok
                if te.any():
                    preds.append({"features": sname, "model": mname, "held_out": a,
                                  "MAE_log": np.abs(yhat[te] - y[te]).mean()})
            if sname == "pre_recovery" and mname == "gbm":
                best_pred = yhat.copy()
    base = np.abs(y - y.mean()).mean()
    res = pd.DataFrame(rows); res.to_csv(out / "rq2_loao.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "rq2_loao_by_attack.csv", index=False)
    md.append(f"Leave-one-attack-out on recovered runs (n = {len(r)}); target = log(1+steps). "
              f"Constant-prediction MAE = {base:.3f}.\n\n" + res.to_markdown(index=False, floatfmt=".3g") + "\n")
    held = pd.DataFrame(preds)
    if "maba_proxy" in set(held.get("held_out", [])):
        m = held[held.held_out == "maba_proxy"].pivot(index="features", columns="model", values="MAE_log")
        md.append("\nHeld-out MABA(-proxy) only (the proposal's generalisation test):\n\n" + m.to_markdown(floatfmt=".3g") + "\n")

    # importance: permutation importance of the full pre-recovery GBM
    from sklearn.inspection import permutation_importance
    X = design(r, sets["pre_recovery"]).drop(columns="const")
    m = models["gbm"]().fit(X, y)
    pi = permutation_importance(m, X, y, n_repeats=20, random_state=0)
    imp = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "sd": pi.importances_std})
    imp = imp.sort_values("importance", ascending=False); imp.to_csv(out / "rq2_importance.csv", index=False)
    md.append("\nPermutation importance (in-sample GBM; use for ranking only):\n\n"
              + imp.head(10).to_markdown(index=False, floatfmt=".3g") + "\n")

    # recoverability classifier (includes censored runs)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    a = runs[runs.strategy != "full_retrain"]
    Xa = design(a, [c for c in PRE if c in a]).drop(columns="const")
    ya = a["recovered"].astype(int).values
    p = np.full(len(a), np.nan)
    for at in a.attack.unique():
        te = (a.attack == at).values
        if len(np.unique(ya[~te])) < 2:
            continue
        p[te] = HistGradientBoostingClassifier(max_depth=3, max_iter=150).fit(Xa[~te], ya[~te]).predict_proba(Xa[te])[:, 1]
    ok = np.isfinite(p)
    if ok.sum() and len(np.unique(ya[ok])) == 2:
        md.append(f"\nRecoverable-within-cap classifier, LOAO AUC = {roc_auc_score(ya[ok], p[ok]):.3f} "
                  f"(n = {ok.sum()}).\n")

    import matplotlib.pyplot as plt
    if "best_pred" in locals():
        fig, ax = plt.subplots(figsize=(4.2, 4))
        attacks = sorted(r.attack.unique())
        for i, at in enumerate(attacks):
            s = (r.attack == at).values
            ax.scatter(best_pred[s], y[s], s=30, color=COLORS[i % 6], label=at, edgecolor="white", lw=1)
        lim = [np.nanmin([best_pred, y]), np.nanmax([best_pred, y])]
        ax.plot(lim, lim, color="#888", lw=1, ls="--")
        ax.set_xlabel("predicted log(1+steps), attack held out"); ax.set_ylabel("actual log(1+steps)")
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        savefig(fig, out / "fig2_loao_pred_vs_actual.png")


# ------------------------------------------------------------------------------------------------
# RQ3: most cost-effective strategy
# ------------------------------------------------------------------------------------------------

def rq3(runs, out, md):
    from sklearn.ensemble import HistGradientBoostingRegressor
    md.append("## RQ3 — Most cost-effective strategy\n")
    r = runs.copy()
    r["asrg_tertile"] = pd.qcut(r["pre_asr_g"].rank(method="first"), 3, labels=["low", "mid", "high"])
    per = r.drop_duplicates(["cell", "budget"])
    tab = pd.crosstab([per.asrg_tertile], per.best_strategy.fillna("none_recovered"))
    tab.to_csv(out / "rq3_best_strategy.csv")
    md.append("Cheapest successful strategy per (model, budget), by ASR-G tertile:\n\n" + tab.to_markdown() + "\n")
    if tab.shape[0] > 1 and tab.shape[1] > 1:
        chi2, p, dof, _ = stats.chi2_contingency(tab.values)
        md.append(f"\nχ²({dof}) = {chi2:.2f}, p = {p:.3g} (does the best strategy depend on ASR-G tertile?)\n")
    sr = r.groupby("strategy").agg(recovery_rate=("recovered", "mean"),
                                   median_gpu_s_if_recovered=("cost_train_gpu_seconds",
                                                              lambda s: s[r.loc[s.index, "recovered"]].median()))
    md.append("\n" + sr.to_markdown(floatfmt=".3g") + "\n")

    # policy: predict each strategy's cost (LOAO), pick argmin; compare with oracle
    feats = [c for c in PRE if c in r]
    X = design(r, feats).drop(columns="const")
    y = np.where(r.recovered, r.log_cost_steps, r.log_cost_steps + np.log(2))  # censored: penalise as >= 2x cap
    yhat = np.full(len(r), np.nan)
    for a in r.attack.unique():
        te = (r.attack == a).values
        yhat[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=200).fit(X[~te], y[~te]).predict(X[te])
    r["yhat"], r["ypen"] = yhat, y
    pol = []
    for (c, b), d in r.groupby(["cell", "budget"]):
        pick = d.loc[d.yhat.idxmin()]; orac = d.loc[d.ypen.idxmin()]
        pol.append({"attack": pick.attack, "picked": pick.strategy, "oracle": orac.strategy,
                    "picked_recovered": pick.recovered, "regret_log": pick.ypen - orac.ypen})
    pol = pd.DataFrame(pol); pol.to_csv(out / "rq3_policy.csv", index=False)
    always = r.groupby("strategy")["ypen"].mean().idxmin()
    fixed_regret = (r[r.strategy == always].set_index(["cell", "budget"]).ypen
                    - r.groupby(["cell", "budget"]).ypen.min()).mean()
    md.append(f"\nPredicted-argmin policy (LOAO): oracle match = {(pol.picked == pol.oracle).mean():.2f}, "
              f"picked strategy recovered = {pol.picked_recovered.mean():.2f}, mean regret = "
              f"{np.expm1(pol.regret_log.mean()):.2f}x extra cost. Baseline 'always {always}': mean regret "
              f"= {np.expm1(fixed_regret):.2f}x.\n")


# ------------------------------------------------------------------------------------------------
# RQ4: proportionality for MABA
# ------------------------------------------------------------------------------------------------

def rq4(runs, out, md, high_attack="maba_proxy"):
    md.append("## RQ4 — Does high-ASR-G MABA need proportionally more effort?\n")
    r = runs[(runs.strategy != "full_retrain")].copy()
    r = r[r.pre_asr_g > 0.01]
    r["log_asrg"] = np.log(r.pre_asr_g)
    X = design(r, ["log_asrg"])
    coef, _ = aft_fit(X, r["log_cost_steps"], r["recovered"])
    ci = cluster_boot(r, lambda d: aft_fit(design(d, ["log_asrg"]), d["log_cost_steps"], d["recovered"])[0]["log_asrg"])
    md.append(f"Elasticity d log(cost) / d log(ASR-G) = {fmt_ci(coef['log_asrg'], ci)} "
              "(1 = proportional, >1 = super-linear; AFT, censoring-aware).\n")
    if high_attack in set(r.attack):
        tr, te = r[r.attack != high_attack], r[r.attack == high_attack]

        def excess(d_tr, d_te):
            c, _ = aft_fit(design(d_tr, ["pre_asr_g"]), d_tr["log_cost_steps"], d_tr["recovered"])
            Xt = design(d_te, ["pre_asr_g"]).reindex(columns=c.index, fill_value=0.0)
            return float(np.mean(d_te["log_cost_steps"] - Xt.values @ c.values))
        e = excess(tr, te)
        ci = cluster_boot(r, lambda d: excess(d[d.attack != high_attack], d[d.attack == high_attack]))
        md.append(f"\n{high_attack} cost relative to what its ASR-G predicts from the other attacks: "
                  f"x{np.exp(e):.2f} [{np.exp(ci[0]):.2f}, {np.exp(ci[1]):.2f}] (1 = exactly as ASR-G predicts; "
                  "censored MABA runs make this a lower bound).\n")


# ------------------------------------------------------------------------------------------------
# RQ5: critical ASR-G threshold
# ------------------------------------------------------------------------------------------------

def breakpoint_fit(x, y):
    best = (np.inf, np.nan)
    for bp in np.quantile(x, np.linspace(0.15, 0.85, 29)):
        X = np.c_[np.ones_like(x), x, np.maximum(0, x - bp)]
        res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        if (res ** 2).sum() < best[0]:
            best = ((res ** 2).sum(), bp)
    return best[1]


def rq5(runs, out, md):
    from sklearn.linear_model import LogisticRegression
    md.append("## RQ5 — Is there a critical ASR-G threshold?\n")
    r = runs[(runs.strategy != "full_retrain")].copy()
    x, y = r.pre_asr_g.values, r.log_cost_steps.values
    bp = breakpoint_fit(x, y)
    ci = cluster_boot(r, lambda d: breakpoint_fit(d.pre_asr_g.values, d.log_cost_steps.values))
    lin = np.polyfit(x, y, 1); sse_lin = ((y - np.polyval(lin, x)) ** 2).sum()
    Xb = np.c_[np.ones_like(x), x, np.maximum(0, x - bp)]
    sse_bp = ((y - Xb @ np.linalg.lstsq(Xb, y, rcond=None)[0]) ** 2).sum()
    n = len(y)
    d_bic = (n * np.log(sse_bp / n) + 4 * np.log(n)) - (n * np.log(sse_lin / n) + 2 * np.log(n))
    md.append(f"Segmented regression of log cost on ASR-G: breakpoint = {fmt_ci(bp, ci)}; "
              f"BIC(segmented) - BIC(linear) = {d_bic:.2f} (< -6 = strong evidence for a threshold).\n")

    def thr(d):
        m = LogisticRegression(C=1e3).fit(d[["pre_asr_g"]].values, d["prohibitive"].astype(int).values)
        return float(-m.intercept_[0] / m.coef_[0, 0])
    rows = []
    for st, d in r.groupby("strategy"):
        if d.prohibitive.nunique() < 2:
            rows.append({"strategy": st, "p_prohibitive": d.prohibitive.mean(), "asrg_at_p50": np.nan}); continue
        t = thr(d)
        inside = d.pre_asr_g.min() <= t <= d.pre_asr_g.max()
        rows.append({"strategy": st, "p_prohibitive": d.prohibitive.mean(), "asrg_at_p50": t if inside else np.nan,
                     "note": "" if inside else f"crossing at {t:.2f} is outside observed ASR-G range (no threshold)",
                     **(dict(zip(["ci_lo", "ci_hi"], cluster_boot(d, thr))) if inside else {})})
    res = pd.DataFrame(rows); res.to_csv(out / "rq5_threshold.csv", index=False)
    md.append("\n'Prohibitive' = not recovered within the cap OR costlier than full retraining of the same "
              "model. ASR-G at which P(prohibitive) = 0.5:\n\n" + res.to_markdown(index=False, floatfmt=".3g") + "\n")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.4))
    grid = np.linspace(0, 1, 101)
    for i, (st, d) in enumerate(r.groupby("strategy")):
        jit = RNG.normal(0, 0.02, len(d))
        ax.scatter(d.pre_asr_g, d.prohibitive.astype(float) + jit, s=14, color=COLORS[i], alpha=0.5)
        if d.prohibitive.nunique() == 2:
            m = LogisticRegression(C=1e3).fit(d[["pre_asr_g"]].values, d.prohibitive.astype(int).values)
            ax.plot(grid, m.predict_proba(grid[:, None])[:, 1], color=COLORS[i], lw=2, label=st)
    ax.set_xlabel("ASR-G (pre-recovery)"); ax.set_ylabel("P(recovery prohibitive)")
    ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    savefig(fig, out / "fig3_prohibitive_vs_asrg.png")


def main(run_dir):
    out = Path(run_dir) / "analysis"
    runs = pd.read_csv(out / "runs.csv")
    for c in ("recovered", "censored", "prohibitive"):
        runs[c] = runs[c].astype(bool)
    md = [f"# ASR-G -> recovery cost: analysis of `{run_dir}`\n"]
    if (runs["model"] == "toy").all():
        md.append("> **Toy-model smoke test.** These numbers validate the pipeline only; they are not evidence "
                  "about real LVLMs and must not be reported as results.\n")
    md.append(f"{runs.cell.nunique()} backdoored models, {len(runs)} recovery runs, "
              f"{runs.recovered.mean():.0%} recovered within the step cap.\n")
    for f in (rq1, rq2, rq3, rq4, rq5):
        try:
            f(runs, out, md)
        except Exception as e:  # keep going; small pilots can starve individual analyses
            md.append(f"\n_{f.__name__} skipped: {type(e).__name__}: {e}_\n")
    (out / "summary.md").write_text("\n".join(md))
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main(sys.argv[1])
