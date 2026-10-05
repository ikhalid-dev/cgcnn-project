#!/usr/bin/env python
"""
Step 89 - does MLIP gamma make the screen's "low" calls more trustworthy than Poisson gamma?
==========================================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/89_mlip_vs_poisson.py

THE QUESTION
------------
The screen turns three models' K and G into kappa through the Slack formula,
and the formula needs a Grueneisen gamma. Two ways to get it:

    Poisson chain   each model's own gamma from its own K and G (step 87)
    MLIP chain      one gamma per crystal from real MLIP phonons (step 88 run)

Step 87 measured the Poisson chain against PhoNIX DFT kappa_L: of the crystals
it calls low (max3 <= 1), 74.7% really are. The DFT shortlist (steps 86, 91)
was built on the MLIP chain, which nothing has measured yet. This step does.

HOW - the plan written in step 88 BEFORE any MLIP gamma existed
----------------------------------------------------------------
Same 617 crystals, same K and G, same formula; only gamma differs. In the
formula gamma sits only inside exp(-gamma), so swapping it is exact:

    kappa(MLIP gamma) = kappa(Poisson gamma) * exp(gamma_Poisson - gamma_MLIP)

(one line per model, then max3 as usual - the same swap steps 74/79 used).

Usable crystals = MLIP finished, no imaginary mode below -0.1 THz, and
0 < gamma < 10: 451 of 617. Both chains are scored on those same 451.

Weights: step 88 drew 50% of the low band, 35% of the near band, 10% and 5%
of the two high bands. Each crystal counts as 1 / (fraction drawn) crystals, so
the numbers below are estimates for all 2,520, not for the sample.

Error bar: a paired, stratified bootstrap - redraw the crystals of each band
with replacement, score BOTH chains on the same redraw, 10,000 times.

PRE-REGISTERED VERDICT (copied from step 88's docstring)
    MLIP - Poisson precision >= +0.05, CI above 0   MLIP makes the screen more trustworthy
    CI includes 0                                    no measurable gain; judge step 86 by step 87
    <= -0.05, CI below 0                             MLIP makes it WORSE; re-rank step 86 on Poisson
secondary: median DFT/predicted per chain (step 86 assumed MLIP gamma ~10% too
    high, which would put the MLIP chain's ratio clearly ABOVE Poisson's 0.92)
must report: the Poisson precision on the 166 crystals MLIP could NOT use,
    so dropping them cannot hide a bias.

ADDED AFTER THE PLAN (labelled as such in the output - not pre-registered)
    * the 1000 K MLIP gamma instead of the 300 K one (sensitivity)
    * step 86's own tier rule (gamma x 0.90 / x 0.68, glass-limit floor)
      applied to PhoNIX - the closest available check on the final 15

OUTPUTS
    results/cgcnn/89_mlip_vs_poisson_crystals.csv   one row per crystal, both chains
    results/cgcnn/89_mlip_vs_poisson_summary.csv    every rule: precision, recall, 95% CI
    results/cgcnn/89_mlip_vs_poisson.png
"""
import glob
import os
import sys
import warnings
from importlib import import_module      # imports a file whose name starts with a digit

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # draw to a file, no window
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")        # pymatgen warns about CIF rounding on every file
from pymatgen.core import Structure      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
CIF_DIR = os.path.join(ROOT, "data", "phonix_cifs")
GAMMA_DIR = os.path.expanduser("~/Desktop/data_generation/results")
SHORTLIST = os.path.join(ROOT, "dft", "final_shortlist_15", "index.csv")

# sys.path = the folders Python searches on "import"; this lets us reuse earlier steps
sys.path.insert(0, os.path.join(ROOT, "scripts", "cgcnn"))
_s81 = import_module("81_combined_dft_queue")     # cahill_min(): the glass-limit floor
_s82 = import_module("82_confusion_and_trust")    # wilson(): error bar for a plain rate
_s86 = import_module("86_low_range_list")         # gamma_corrections(): the x0.90 / x0.68

MODELS = ["ALIGNN", "CGCNN-ens", "newbase"]
LOW = 1.0            # W/m/K - "low"
B = 10_000           # bootstrap redraws (the underscore is just a thousands separator)
B_SLOW = 2_000       # for the statistics that need a Python loop per redraw
SEED = 89


# =============================================================================
#  helpers
# =============================================================================
def weighted_median(x, w):
    """The value with half the total WEIGHT below it (plain median when all w are equal)."""
    order = np.argsort(x)                        # positions that would sort x
    cum = np.cumsum(w[order])                    # running total of weight, smallest x first
    return x[order][np.searchsorted(cum, 0.5 * cum[-1])]


def verdict(delta, lo, hi):
    """The pre-registered table from step 88, applied mechanically."""
    if lo <= 0 <= hi:
        return "CI includes 0 -> no measurable gain; judge step 86 by step 87's Poisson calibration"
    if delta >= 0.05 and lo > 0:
        return "MLIP makes the screen MORE trustworthy; step 86 rests on the better chain"
    if delta <= -0.05 and hi < 0:
        return "MLIP makes the screen WORSE; step 86 must be re-ranked on the Poisson chain"
    return "real but smaller than 0.05 - outside the pre-registered cases, reported as is"


def main():
    # ---- 1. the sample, the Poisson predictions, the MLIP gammas -------------
    plan = pd.read_csv(os.path.join(RES, "88_phonix_mlip_plan.csv"))[["mp_id", "band", "weight"]]
    pred = pd.read_csv(os.path.join(RES, "87_phonix_poisson_predictions.csv"))
    files = sorted(glob.glob(os.path.join(GAMMA_DIR, "01_gamma_phonon_phonix_cal_s0?.csv")))
    mlip = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    mlip = mlip[["mp_id", "status", "dynamically_stable", "min_freq_THz",
                 "gamma_c300", "gamma_mlip_1000K"]]
    # how="left" keeps every planned crystal even if a lookup were missing...
    d = plan.merge(pred, on="mp_id", how="left").merge(mlip, on="mp_id", how="left")
    # ...and these asserts then prove nothing WAS missing
    assert len(d) == 617 == d.mp_id.nunique(), "plan rows lost or duplicated"
    assert d.klat.notna().all() and d.gamma_c300.notna().all(), "a crystal has no DFT or no MLIP row"

    d["usable"] = ((d.status == "ok") & d.dynamically_stable.astype(bool)
                   & d.gamma_c300.between(0, 10, inclusive="neither"))
    d["drop_reason"] = np.select([d.usable, ~d.dynamically_stable.astype(bool)],
                                 ["", "unstable (imaginary modes)"], default="gamma <= 0")

    # ---- 2. the two chains ---------------------------------------------------
    d["poisson_max3"] = d.kappa_max3                       # step 87's number, untouched
    for gcol, out in [("gamma_c300", "mlip_max3"), ("gamma_mlip_1000K", "mlip1000_max3")]:
        # one column per model with the gamma swapped, then the highest of the three.
        # pd.concat([...], axis=1) puts the three Series side by side as columns.
        per_model = pd.concat([d[m] * np.exp(d[f"gamma_{m}"] - d[gcol]) for m in MODELS], axis=1)
        d[out] = per_model.max(axis=1)

    # ---- 3. step 86's tier rule, on PhoNIX ------------------------------------
    typical, stress = _s86.gamma_corrections()             # prints the benchmark it uses
    sts = [Structure.from_file(os.path.join(CIF_DIR, f"{m}.cif")) for m in d.mp_id]
    assert all(len(s) == n for s, n in zip(sts, d.n_atoms)), "a CIF has the wrong atom count"
    d["kappa_cahill"] = _s81.cahill_min(d.K_cgcnn.values, d.G_cgcnn.values,
                                        np.array([s.density for s in sts]),
                                        np.array([s.volume for s in sts]), d.n_atoms.values)
    g = d.gamma_c300
    d["mlip_typical"] = np.maximum(d.mlip_max3 * np.exp(g * (1 - typical)), d.kappa_cahill)
    d["mlip_stress"] = np.maximum(d.mlip_max3 * np.exp(g * (1 - stress)), d.kappa_cahill)
    s15 = pd.read_csv(SHORTLIST).kappa_stress.max()        # the 15th crystal's stress kappa

    keep = ["mp_id", "formula", "n_atoms", "band", "weight", "usable", "drop_reason",
            "min_freq_THz", "klat", "kp", "gamma_ALIGNN", "gamma_CGCNN-ens", "gamma_newbase",
            "gamma_c300", "gamma_mlip_1000K", "poisson_max3", "mlip_max3", "mlip1000_max3",
            "mlip_typical", "mlip_stress", "kappa_cahill"]
    d[keep].to_csv(os.path.join(RES, "89_mlip_vs_poisson_crystals.csv"), index=False)

    # ---- 4. the paired, stratified bootstrap ----------------------------------
    u = d[d.usable].reset_index(drop=True)
    w = u.weight.values
    band = u.band.values
    rng = np.random.default_rng(SEED)
    # For each band: B rows, each a redraw (with replacement) of that band's
    # positions. Glued side by side, every row of idx is one complete redraw
    # with each band keeping its own size. ALL rules are scored on the same rows,
    # which is what makes the comparison paired.
    parts = []
    for b in sorted(set(band)):
        pos = np.flatnonzero(band == b)                    # where this band's crystals sit
        parts.append(pos[rng.integers(0, len(pos), (B, len(pos)))])
    idx = np.concatenate(parts, axis=1)                    # shape (10000, 451)

    def rates(call, truth, rows=None):
        """Weighted precision and recall; on the real data, or on every redraw at once."""
        if rows is None:
            W, C, T = w, call, truth
            return (W * C * T).sum() / (W * C).sum(), (W * C * T).sum() / (W * T).sum()
        W, C, T = w[rows], call[rows], truth[rows]         # each now (10000, 451)
        with np.errstate(invalid="ignore"):                # 0/0 on a redraw -> NaN, dropped later
            return ((W * C * T).sum(1) / (W * C).sum(1),   # .sum(1) = add along each row
                    (W * C * T).sum(1) / (W * T).sum(1))

    # each rule = one True/False per crystal: "this rule calls it low"
    T1, S15 = "step-86 tier 1, stress <= 1 [added]", f"as safe as the final 15, stress <= {s15:.2f} [added]"
    rules = {
        "Poisson gamma (step 87 chain)": (u.poisson_max3 <= LOW).values,
        "MLIP gamma 300 K (production)": (u.mlip_max3 <= LOW).values,
        "MLIP gamma 1000 K [added]": (u.mlip1000_max3 <= LOW).values,
        "step-86 tiers 1+2, typical <= 1 [added]": (u.mlip_typical <= LOW).values,
        T1: (u.mlip_stress <= LOW).values,
        S15: (u.mlip_stress <= s15).values,
    }
    # ANY stricter cutoff raises precision, so a strict MLIP rule must be compared
    # with the Poisson chain made EQUALLY strict: the Poisson cutoff that calls the
    # same (weighted) number of crystals low. Then only gamma differs again.
    order = np.argsort(u.poisson_max3.values)
    cum_w = np.cumsum(w[order])                  # weight called if the cutoff sat at each crystal
    matched = {}
    for name in [T1, S15]:
        target = (w * rules[name]).sum()
        c = u.poisson_max3.values[order][np.searchsorted(cum_w, target - 1e-9)]
        mname = f"Poisson, cutoff {c:.2f} = as strict as '{name.split(',')[0]}' [added]"
        rules[mname] = (u.poisson_max3 <= c).values
        matched[name] = mname
    rows, boots = [], {}
    for truth_col in ["klat", "kp"]:
        truth = (u[truth_col] <= LOW).values
        for name, call in rules.items():
            p, r = rates(call, truth)
            bp, br = rates(call, truth, idx)
            if truth_col == "klat":
                boots[name] = bp
            rows.append({"truth": truth_col, "rule": name,
                         "n_called_sample": int(call.sum()),
                         "n_confirmed_sample": int((call & truth).sum()),
                         "est_called_of_usable_2520": float((w * call).sum()),
                         "precision": p, "precision_lo95": np.nanpercentile(bp, 2.5),
                         "precision_hi95": np.nanpercentile(bp, 97.5),
                         "recall": r, "recall_lo95": np.nanpercentile(br, 2.5),
                         "recall_hi95": np.nanpercentile(br, 97.5),
                         "base_rate": float((w * truth).sum() / w.sum())})
    summ = pd.DataFrame(rows)

    # ---- 5. THE PRIMARY RESULT --------------------------------------------------
    P, M = "Poisson gamma (step 87 chain)", "MLIP gamma 300 K (production)"
    k = summ[summ.truth == "klat"].set_index("rule")
    delta = k.precision[M] - k.precision[P]
    dboot = boots[M] - boots[P]
    lo, hi = np.nanpercentile(dboot, [2.5, 97.5])
    call_p, call_m = (u.poisson_max3 <= LOW).values, (u.mlip_max3 <= LOW).values

    # ---- 6. secondary: median DFT/pred, Spearman ------------------------------
    sec = []
    for name, col in [(P, "poisson_max3"), (M, "mlip_max3"), ("MLIP gamma 1000 K [added]", "mlip1000_max3"),
                      ("step-86 typical correction [added]", "mlip_typical")]:
        ratio = (u.klat / u[col]).values
        med_boot = [weighted_median(ratio[i], w[i]) for i in idx[:B_SLOW]]
        lx, ly = np.log10(u[col].values), np.log10(u.klat.values)
        rho_boot = [spearmanr(lx[i], ly[i]).correlation for i in idx[:B_SLOW]]
        sec.append({"chain": name, "median_dft_over_pred": weighted_median(ratio, w),
                    "median_lo95": np.percentile(med_boot, 2.5),
                    "median_hi95": np.percentile(med_boot, 97.5),
                    "spearman_rho_sample": spearmanr(lx, ly).correlation,
                    "rho_lo95": np.percentile(rho_boot, 2.5), "rho_hi95": np.percentile(rho_boot, 97.5),
                    "_rho_boot": np.array(rho_boot)})
    sec = pd.DataFrame(sec)
    drho = sec._rho_boot[1] - sec._rho_boot[0]             # MLIP minus Poisson, paired
    summ.to_csv(os.path.join(RES, "89_mlip_vs_poisson_summary.csv"), index=False)
    sec.drop(columns="_rho_boot").to_csv(os.path.join(RES, "89_mlip_vs_poisson_secondary.csv"), index=False)

    # ---- 7. the dropouts: is "MLIP could not use it" itself a signal? ----------
    # Poisson calls low = band A exactly, and band A has one weight, so plain
    # counts and a Wilson interval are the right tool here
    def poisson_prec(frame):
        c = frame[frame.poisson_max3 <= LOW]
        n_ok = int((c.klat <= LOW).sum())
        return n_ok, len(c), n_ok / len(c), *_s82.wilson(n_ok, len(c))
    drop = d[~d.usable]

    # ---- 8. print ----------------------------------------------------------------
    f3 = lambda v: f"{v:.3f}"                              # noqa: E731 - a tiny formatter
    print("=" * 100)
    print("STEP 89 - MLIP GAMMA vs POISSON GAMMA, SCORED AGAINST PHONIX DFT kappa_L")
    print("=" * 100)
    print(f"\nsample 617 -> usable {len(u)}  (dropped: "
          f"{drop.drop_reason.value_counts().to_dict()})")
    print("usable per band:", u.band.value_counts().sort_index().to_dict())
    print(f"base rate (DFT klat <= 1, weighted to all 2,520): {k.base_rate[P]:.3f}")

    print("\nSANITY - Poisson precision on the whole 617 vs step 87's 0.747 on all 2,520:")
    n_ok, n, p, l95, h95 = poisson_prec(d)
    print(f"   {n_ok}/{n} = {p:.3f} [{l95:.3f}, {h95:.3f}]")

    print("\nPRIMARY (pre-registered): precision at klat <= 1, max3, weighted, on the same "
          f"{len(u)} crystals")
    print(f"   Poisson chain  {k.precision[P]:.3f} [{k.precision_lo95[P]:.3f}, {k.precision_hi95[P]:.3f}]"
          f"   calls low {call_p.sum()} in the sample, {(call_p & (u.klat <= LOW)).sum()} confirmed")
    print(f"   MLIP chain     {k.precision[M]:.3f} [{k.precision_lo95[M]:.3f}, {k.precision_hi95[M]:.3f}]"
          f"   calls low {call_m.sum()} in the sample, {(call_m & (u.klat <= LOW)).sum()} confirmed")
    print(f"   MLIP - Poisson {delta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"P(MLIP better) {np.nanmean(dboot > 0):.2f}")
    print(f"\n   VERDICT: {verdict(delta, lo, hi)}")

    print(f"\n   recall (of the truly low, the fraction called low): "
          f"Poisson {k.recall[P]:.3f} [{k.recall_lo95[P]:.3f}, {k.recall_hi95[P]:.3f}], "
          f"MLIP {k.recall[M]:.3f} [{k.recall_lo95[M]:.3f}, {k.recall_hi95[M]:.3f}]")
    both = pd.crosstab(pd.Series(call_p, name="Poisson calls low"),
                       pd.Series(call_m, name="MLIP calls low"))
    print("\n   where the two chains disagree (sample counts):")
    print("   " + both.to_string().replace("\n", "\n   "))
    for a, b_, lab in [(True, False, "Poisson only"), (False, True, "MLIP only")]:
        m = (call_p == a) & (call_m == b_)
        if m.any():
            print(f"   {lab}: {m.sum()} crystals, DFT low in {(u.klat[m] <= LOW).sum()}")

    print("\nSECONDARY (pre-registered): median DFT/pred (weighted) and Spearman rho (sample)")
    print(sec.drop(columns="_rho_boot").to_string(index=False, float_format=f3))
    print(f"   rho MLIP - Poisson {sec.spearman_rho_sample[1] - sec.spearman_rho_sample[0]:+.4f}"
          f"  95% CI [{np.percentile(drho, 2.5):+.4f}, {np.percentile(drho, 97.5):+.4f}]"
          f"  P(MLIP ranks better) {np.mean(drho > 0):.3f}")

    print("\nMUST-REPORT (pre-registered): Poisson precision, crystals MLIP could use vs could not")
    for lab, frame in [("usable  ", u), ("dropped ", drop)]:
        n_ok, n, p, l95, h95 = poisson_prec(frame)
        print(f"   {lab} {n_ok:3d}/{n:3d} = {p:.3f} [{l95:.3f}, {h95:.3f}]")

    print("\nEVERY RULE, truth = klat (weighted precision / recall, 95% CI)  [added] = not pre-registered")
    show = k.reset_index()[["rule", "n_called_sample", "n_confirmed_sample", "est_called_of_usable_2520",
                            "precision", "precision_lo95", "precision_hi95", "recall"]]
    print(show.to_string(index=False, float_format=f3))
    print("\nSTRICTNESS-MATCHED [added]: the strict MLIP rules vs Poisson made equally strict")
    for name, mname in matched.items():
        dd = boots[name] - boots[mname]
        l2, h2 = np.nanpercentile(dd, [2.5, 97.5])
        print(f"   {name.split(' [')[0]:40s} {k.precision[name]:.3f}  vs  {mname.split(' =')[0]:22s} "
              f"{k.precision[mname]:.3f}   diff {k.precision[name] - k.precision[mname]:+.3f} "
              f"[{l2:+.3f}, {h2:+.3f}]")
    print(f"\n(step-86 factors used: typical x{typical:.3f}, stress x{stress:.3f}; "
          f"glass floor from CGCNN-ens K, G)")

    # ---- 9. figure -------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    lim = [0.03, 3000]
    for a, col, title in [(ax[0, 0], "poisson_max3", "Poisson gamma"),
                          (ax[0, 1], "mlip_max3", "MLIP gamma (300 K)")]:
        x, y = u[col], u.klat
        groups = [((x > LOW) & (y > LOW), "#cccccc", "both high"),
                  ((x > LOW) & (y <= LOW), "#c9803a", "missed"),
                  ((x <= LOW) & (y > LOW), "#b03a3a", "false low"),
                  ((x <= LOW) & (y <= LOW), "#3b6ea8", "confirmed low")]
        for m, c, lab in groups:
            a.scatter(x[m], y[m], s=10, color=c, label=f"{lab} ({m.sum()})")
        a.plot(lim, lim, "k-", lw=0.6)
        a.axvline(LOW, color="k", ls="--", lw=1)
        a.axhline(LOW, color="k", ls="--", lw=1)
        a.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
              xlabel=f"screen kappa_max3, {title} (W/m/K)", ylabel="PhoNIX DFT klat (W/m/K)",
              title=f"{title}: same {len(u)} crystals (sample counts)")
        a.legend(fontsize=8, loc="upper left")

    a = ax[1, 0]
    a.hist(dboot[np.isfinite(dboot)], bins=60, color="#7a9cc6")
    a.axvspan(-1, -0.05, color="#b03a3a", alpha=0.08)
    a.axvspan(0.05, 1, color="#3b6ea8", alpha=0.08)
    a.axvline(0, color="k", lw=1)
    a.axvline(delta, color="k", ls="--", lw=1.5, label=f"measured {delta:+.3f}")
    a.axvline(lo, color="grey", ls=":")
    a.axvline(hi, color="grey", ls=":", label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    a.set(xlim=(min(-0.3, lo - 0.05), max(0.3, hi + 0.05)),
          xlabel="precision, MLIP chain minus Poisson chain",
          ylabel="bootstrap redraws",
          title="Pre-registered test (red: MLIP worse, blue: MLIP better)")
    a.legend(fontsize=8)

    a = ax[1, 1]
    names = list(rules)
    yy = np.arange(len(names))[::-1]
    pr = k.loc[names]
    a.errorbar(pr.precision, yy, xerr=[pr.precision - pr.precision_lo95, pr.precision_hi95 - pr.precision],
               fmt="o", color="k", capsize=3)
    n_ok, n, p, l95, h95 = poisson_prec(drop)
    a.errorbar([p], [-1], xerr=[[p - l95], [h95 - p]], fmt="s", color="#b03a3a", capsize=3)
    a.axvline(k.base_rate[P], color="grey", ls=":", label="blind pick (base rate)")
    a.set_yticks(list(yy) + [-1])
    a.set_yticklabels([n.replace(" [added]", "*") for n in names]
                      + [f"Poisson, on the {len(drop)} MLIP dropped"], fontsize=7)
    a.set(xlim=(0, 1), xlabel="precision: of the crystals called low, fraction DFT-low",
          title="Every rule, weighted to all 2,520 (95% CI)\n* = added after the plan")
    a.legend(fontsize=8, loc="upper left")    # left is empty: every precision is above 0.6
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "89_mlip_vs_poisson.png"), dpi=150)
    print("\nwrote 89_mlip_vs_poisson_crystals.csv, _summary.csv, _secondary.csv, .png")


if __name__ == "__main__":
    main()
