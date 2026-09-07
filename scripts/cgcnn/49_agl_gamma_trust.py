#!/usr/bin/env python3
"""
STEP 49 - How trustworthy is AFLOW-AGL's Gruneisen parameter?
================================================================================

    python scripts/cgcnn/49_agl_gamma_trust.py

WHY THIS MATTERS
------------------
The gamma head in 37_train_gamma.py is trained on AFLOW-AGL's tabulated
`agl_gruneisen`. That number is not measured: it comes from a quasi-harmonic
Debye-Gruneisen fit to first-principles energy-volume curves. If those fits
carry the same isotropic-Debye assumptions the project is trying to escape,
then a gamma head trained on them inherits the bias no matter how well it fits
its labels, and its ceiling is set by the labels rather than by the network.

The only independent check available in this repo is PINK's Table 1, whose
Gruneisen column the paper describes as "obtained from the AFLOW database and
experimental data" - so it is partly literature/experiment and partly AFLOW.
That mixed provenance is not a footnote; it is the single largest threat to
this comparison, so this script MEASURES it (see the circularity section
below) instead of assuming one way or the other.

WHAT IS COMPUTED
------------------
  1. how many Table 1 materials are also in AFLOW-AGL, and under which
     matching rule;
  2. for that overlap: count, mean absolute difference in gamma, the same in
     log10 units, Pearson and Spearman correlation, and the bias;
  3. a CIRCULARITY TEST - how many Table 1 gamma values agree with AGL's to
     within the precision Table 1 prints them at. A value printed as "1.9"
     that AGL also gives as 1.90 is evidence that this row's gamma WAS the
     AFLOW value, in which case it cannot serve as an independent check of it.
     Every statistic is reported twice: on the whole overlap, and on the
     subset that is NOT consistent with having been copied from AFLOW;
  4. the calibration the request asks for, against the project's own measured
     cross-source noise on the shear modulus.

THE CALIBRATION, SPELLED OUT
------------------------------
The moduli comparison, corrected by step 59. An earlier version of this script
quoted a single ratio:

    AFLOW vs matbench, 555 matched compounds   |d log10 G| = 0.1085
    the model's own held-out error                          0.0781
                                                  ratio     1.39

and concluded from it that "the labels, not the network, set the floor".
THAT WAS WRONG, for a reason worth stating plainly: the 0.1085 numerator was
measured on the SOFT SUBSET only, while the 0.0781 denominator is a
whole-test-set number. Measured on matched populations:

    full population   0.0539 / 0.0781 = 0.69
    soft population   0.1044 / 0.1033 = 1.01

Per stiffness band the model's own error exceeds the label disagreement in ALL
of them (ratios 0.68-0.86), so on the moduli the network, not the labels, is
the weaker half. Step 59 also showed the soft subset's apparent -0.047 offset
is a selection artifact - aflow_soft.csv is chosen because AFLOW says the
crystal is soft, which over-represents AFLOW's downward errors, and the bias
flips sign if matbench does the selecting instead.

None of this rehabilitates AGL gamma. The comparison below uses the LARGER
(soft-matched, 1.01) of the two valid moduli ratios, which is the hardest
baseline for the gamma claim to beat; gamma still loses to it by a wide margin,
so the conclusion of this script is unchanged and now rests on a like-for-like
comparison. The same ratio is computed here for gamma, using the gamma head's
own measured error
from results/cgcnn/summary_37_r9_full_s*.json (`mae_gamma`, which 37 computes
in ABSOLUTE gamma units - so the conversion to log10 is done explicitly on
this sample's own gamma distribution rather than assumed).

SAMPLE SIZE
-------------
Table 1 has 46 materials. Whatever overlaps AFLOW will be smaller than that,
and every conclusion drawn here is a small-sample conclusion. The count is
printed before any statistic and repeated in the conclusion sentence.
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "table1_csv": "results/cgcnn/table1_reference.csv",          # 46 rows transcribed from PINK Table 1 (gamma_paper)
    "table1_moduli_csv": "results/cgcnn/table1_moduli_predictions.csv",   # supplies n_sites for the strict match rule
    "aflow_csv": "data_full/aflow_agl.csv",                      # AFLOW-AGL export (agl_gruneisen)
    "gamma_head_summaries": "results/cgcnn/summary_37_r9_full_s*.json",   # the gamma head's own measured error
    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/agl_target/csv",
    "png_dir": "results/cgcnn/agl_target/png",
    # ---- matching / analysis -----------------------------------------------
    "match_rule": "formula",        # "formula" = reduced formula only; "formula+atoms" also requires equal atom count
    "aggregate": "median",          # how to collapse multiple AFLOW entries sharing one key
    # Reference numbers for the calibration paragraph, from this project's own
    # measurements. Kept here so they can be updated in one place.
    # Cross-source label noise on G, and the model error to compare it against.
    # Step 59 re-measured these from the raw label files and found the value
    # this script used to hard-code (0.1085) is the SOFT SUBSET's number, while
    # the 0.0781 it was divided by is a WHOLE-TEST-SET number - two different
    # populations, so the old 1.39 ratio was not a like-for-like comparison.
    # Both populations are kept here, each with its own matching model error.
    "ref_G_soft_cross_source": 0.1044,   # aflow_soft.csv overlap, n=450, formula+atoms
    "ref_G_soft_model_error":  0.1033,   # ensemble held-out MAE on the same G range
    "ref_G_full_cross_source": 0.0539,   # full matbench/AFLOW overlap, n=2,666
    "ref_G_full_model_error":  0.0781,   # ensemble held-out MAE, whole test set
}
# =============================================================================

import os                    # path joining and directory creation
import sys                   # sys.path manipulation and early exit
import csv                   # reading Table 1 as raw text, to recover printed precision
import glob                  # expanding the gamma-head summary glob
import json                  # reading those summaries

import torch  # noqa: F401   # import order rule: torch before numpy/pandas
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from scipy.stats import spearmanr, pearsonr  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def reduced_formula(formula):
    """Same normalisation 35_merge_aflow.py and 48 use, so keys are comparable."""
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def printed_decimals(text):
    """How many decimal places a number was PRINTED with, from its raw string.

    '1.9' -> 1, '0.923' -> 3, '2' -> 0. This is what sets the rounding window
    for the circularity test: a paper value of 1.9 is consistent with any true
    value in [1.85, 1.95), so agreement inside that window is not evidence of
    an independent measurement.
    """
    text = str(text).strip()
    return len(text.split(".")[1]) if "." in text else 0


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 49 - is AFLOW-AGL's gamma trustworthy enough to train a head on?")
    print("=" * 78)
    print()

    # ---- Table 1, keeping the printed precision of each gamma ---------------
    t1_path = os.path.join(PROJECT_ROOT, cfg["table1_csv"])
    raw_rows = list(csv.DictReader(open(t1_path)))
    t1 = pd.read_csv(t1_path)
    t1["gamma_decimals"] = [printed_decimals(r["gamma_paper"]) for r in raw_rows]
    t1 = t1.dropna(subset=["gamma_paper"])
    print(f"  PINK Table 1: {len(t1)} materials carrying a gamma value")

    # n_sites, where a previous step already resolved the structure for it.
    mod_path = os.path.join(PROJECT_ROOT, cfg["table1_moduli_csv"])
    if os.path.exists(mod_path):
        mods = pd.read_csv(mod_path)[["material_id", "n_sites"]]
        t1 = t1.merge(mods, on="material_id", how="left")
    else:
        t1["n_sites"] = np.nan

    # ---- AFLOW-AGL ----------------------------------------------------------
    aflow = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["aflow_csv"]))
    aflow = aflow[np.isfinite(aflow["agl_gruneisen"]) & (aflow["agl_gruneisen"] > 0)]
    aflow["_rf"] = [reduced_formula(c) for c in aflow["compound"]]
    aflow = aflow.dropna(subset=["_rf"])
    t1["_rf"] = [reduced_formula(f) for f in t1["formula"]]

    if cfg["match_rule"] == "formula+atoms":
        aflow["_key"] = aflow["_rf"] + "|" + aflow["natoms"].astype(int).astype(str)
        t1["_key"] = [f"{rf}|{int(n)}" if pd.notna(n) else None
                      for rf, n in zip(t1["_rf"], t1["n_sites"])]
    else:
        aflow["_key"] = aflow["_rf"]
        t1["_key"] = t1["_rf"]

    agg = aflow.groupby("_key").agg(
        agl_gamma=("agl_gruneisen", cfg["aggregate"]),
        agl_gamma_min=("agl_gruneisen", "min"),
        agl_gamma_max=("agl_gruneisen", "max"),
        agl_kappa=("agl_thermal_conductivity_300K", cfg["aggregate"]),
        agl_compound=("compound", "first"),
        agl_spacegroup=("spacegroup_relax", "first"),
        n_aflow_entries=("compound", "size"),
    ).reset_index()

    both = t1.dropna(subset=["_key"]).merge(agg, on="_key", how="inner")
    n = len(both)

    print(f"  matching rule: {cfg['match_rule']}")
    print(f"  OVERLAP WITH AFLOW-AGL: {n} of {len(t1)} Table 1 materials")
    if n:
        multi = int((both["n_aflow_entries"] > 1).sum())
        print(f"    {multi} of them map to more than one AFLOW entry "
              f"(collapsed by {cfg['aggregate']}); within-key gamma spread up to "
              f"{(both['agl_gamma_max'] / both['agl_gamma_min']).max():.2f}x")
    print()
    if n < 5:
        sys.exit("ERROR: fewer than 5 matched materials - no statistic would mean anything.")

    # ---- the circularity test ----------------------------------------------
    # A Table 1 gamma printed to d decimals is consistent with the AFLOW value
    # if they agree inside the rounding window of that printed precision.
    window = 0.5 * 10.0 ** (-both["gamma_decimals"].astype(float))
    both["consistent_with_aflow_copy"] = (both["gamma_paper"] - both["agl_gamma"]).abs() <= window
    n_copy = int(both["consistent_with_aflow_copy"].sum())
    print("  CIRCULARITY TEST - is Table 1's gamma independent of AFLOW at all?")
    print("  " + "-" * 74)
    print(f"    {n_copy} of {n} Table 1 gammas agree with AGL to within their own")
    print(f"    printed precision, i.e. are consistent with having BEEN the AFLOW value.")
    print(f"    {n - n_copy} differ by more than rounding and are treated as independent below.")
    print("    (The paper states Table 1's gammas came from 'the AFLOW database and")
    print("     experimental data' - this quantifies the split it does not itemise.)")
    print()

    # ---- the statistics, on the full overlap and on the independent subset --
    def stats(frame, name):
        g_exp = frame["gamma_paper"].values.astype(float)
        g_agl = frame["agl_gamma"].values.astype(float)
        ok = np.isfinite(g_exp) & np.isfinite(g_agl) & (g_exp > 0) & (g_agl > 0)
        g_exp, g_agl = g_exp[ok], g_agl[ok]
        if len(g_exp) < 3:
            return None
        d = g_agl - g_exp
        dl = np.log10(g_agl) - np.log10(g_exp)
        out = {
            "subset": name,
            "n": int(len(g_exp)),
            "mean_abs_diff_gamma": float(np.abs(d).mean()),
            "median_abs_diff_gamma": float(np.median(np.abs(d))),
            "bias_gamma_agl_minus_exp": float(d.mean()),
            "mean_abs_diff_log10_gamma": float(np.abs(dl).mean()),
            "median_abs_diff_log10_gamma": float(np.median(np.abs(dl))),
            "bias_log10": float(dl.mean()),
            "pearson_r": float(pearsonr(g_exp, g_agl)[0]),
            "spearman_rho": float(spearmanr(g_exp, g_agl).correlation),
            "median_gamma_exp": float(np.median(g_exp)),
        }
        return out

    all_stats = [s for s in [stats(both, "all matched"),
                             stats(both[~both["consistent_with_aflow_copy"]],
                                   "excluding AFLOW-consistent rows")] if s]
    st = pd.DataFrame(all_stats)

    print("  EXPERIMENTAL/LITERATURE GAMMA vs AGL GAMMA")
    print("  " + "-" * 74)
    for s in all_stats:
        print(f"    {s['subset']}  (n = {s['n']})")
        print(f"      mean |d gamma|        {s['mean_abs_diff_gamma']:.3f}   "
              f"median {s['median_abs_diff_gamma']:.3f}")
        print(f"      mean |d log10 gamma|  {s['mean_abs_diff_log10_gamma']:.4f}   "
              f"median {s['median_abs_diff_log10_gamma']:.4f}")
        print(f"      bias (AGL - exp)      {s['bias_gamma_agl_minus_exp']:+.3f} "
              f"({s['bias_log10']:+.4f} in log10)")
        print(f"      pearson r {s['pearson_r']:+.3f}   spearman rho {s['spearman_rho']:+.3f}")
        print()

    # ---- calibration against the gamma head's own achievable error ---------
    summaries = sorted(glob.glob(os.path.join(PROJECT_ROOT, cfg["gamma_head_summaries"])))
    head_mae_abs, head_mae_derived = [], []
    for p in summaries:
        blob = json.load(open(p))
        res = blob.get("results", {}).get("test", {})
        if "mae_gamma" in res:
            head_mae_abs.append(float(res["mae_gamma"]))
        if "mae_gamma_derived" in res:
            head_mae_derived.append(float(res["mae_gamma_derived"]))

    print("  CALIBRATION - is the gamma label noise bigger than the head's own error?")
    print("  " + "-" * 74)
    ref = all_stats[0]
    if head_mae_abs:
        head_abs = float(np.mean(head_mae_abs))
        # 37 reports mae_gamma in ABSOLUTE gamma units. Convert to log10 using
        # this sample's own median gamma - d(log10 g) = d g / (g * ln 10) -
        # so the two numbers are finally in the same units as the G comparison.
        med_g = ref["median_gamma_exp"]
        head_log = head_abs / (med_g * np.log(10.0))
        print(f"    gamma head (round 9, {len(head_mae_abs)} seeds): MAE {head_abs:.3f} in absolute")
        print(f"      gamma units -> ~{head_log:.4f} in log10 at this sample's median gamma "
              f"({med_g:.2f})")
        if head_mae_derived:
            print(f"    for scale, the Poisson-derived gamma it replaces: MAE "
                  f"{float(np.mean(head_mae_derived)):.3f} absolute "
                  f"({float(np.mean(head_mae_derived)) / (med_g * np.log(10.0)):.4f} log10)")
        ratio_gamma = ref["mean_abs_diff_log10_gamma"] / head_log
        # Two valid, population-matched ratios. The comparison below uses the
        # LARGER one, so the gamma verdict is stated against the most
        # favourable moduli baseline rather than the most convenient.
        ratio_G_full = cfg["ref_G_full_cross_source"] / cfg["ref_G_full_model_error"]
        ratio_G_soft = cfg["ref_G_soft_cross_source"] / cfg["ref_G_soft_model_error"]
        ratio_G = max(ratio_G_full, ratio_G_soft)
        print()
        print(f"    {'quantity':<34}{'label noise':>13}{'model error':>13}{'ratio':>8}")
        print(f"    {'shear modulus G (full population)':<34}"
              f"{cfg['ref_G_full_cross_source']:>13.4f}"
              f"{cfg['ref_G_full_model_error']:>13.4f}{ratio_G_full:>8.2f}")
        print(f"    {'shear modulus G (soft population)':<34}"
              f"{cfg['ref_G_soft_cross_source']:>13.4f}"
              f"{cfg['ref_G_soft_model_error']:>13.4f}{ratio_G_soft:>8.2f}")
        print(f"    {'gamma (this script)':<34}"
              f"{ref['mean_abs_diff_log10_gamma']:>13.4f}{head_log:>13.4f}{ratio_gamma:>8.2f}")
        print()
        if ratio_gamma > ratio_G:
            print(f"    The gamma situation is WORSE than the shear-modulus one: the")
            print(f"    disagreement between AGL and literature gamma is {ratio_gamma:.1f}x the error")
            print(f"    the head already achieves, against {ratio_G:.1f}x for G.")
        else:
            print(f"    The gamma situation is NOT worse than the shear-modulus one")
            print(f"    ({ratio_gamma:.1f}x vs {ratio_G:.1f}x).")
        st["gamma_head_mae_abs"] = head_abs
        st["gamma_head_mae_log10_equiv"] = head_log
    else:
        ratio_gamma = np.nan
        print("    (no gamma-head summaries found - calibration skipped)")
    print()

    # ---- outputs ------------------------------------------------------------
    out_cols = ["material_id", "formula", "_key", "gamma_paper", "gamma_decimals",
                "agl_gamma", "agl_gamma_min", "agl_gamma_max", "n_aflow_entries",
                "consistent_with_aflow_copy", "agl_compound", "agl_spacegroup",
                "kappa_exp", "agl_kappa", "provenance"]
    out = both[[c for c in out_cols if c in both.columns]].copy()
    out["d_gamma_agl_minus_paper"] = out["agl_gamma"] - out["gamma_paper"]
    out["d_log10_gamma"] = np.log10(out["agl_gamma"]) - np.log10(out["gamma_paper"])
    p1 = os.path.join(csv_dir, "49_table1_gamma_vs_agl.csv")
    p2 = os.path.join(csv_dir, "49_gamma_trust_summary.csv")
    out.sort_values("d_log10_gamma", key=np.abs, ascending=False).to_csv(p1, index=False)
    st.to_csv(p2, index=False)
    print(f"  Wrote {p1}")
    print(f"  Wrote {p2}")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    m = both["consistent_with_aflow_copy"].values
    ax.scatter(both["gamma_paper"][~m], both["agl_gamma"][~m], s=46, color=BLUE,
               edgecolors="white", linewidths=0.6, label=f"differs from AGL ({int((~m).sum())})")
    ax.scatter(both["gamma_paper"][m], both["agl_gamma"][m], s=46, color=MUTED,
               edgecolors="white", linewidths=0.6,
               label=f"equals AGL to printed precision ({int(m.sum())})")
    lim = [0, float(max(both["gamma_paper"].max(), both["agl_gamma"].max())) * 1.1]
    ax.plot(lim, lim, color=INK_SOFT, lw=1.2, ls="--", label="equal")
    # Direct-label the biggest disagreements only - a label on every point is
    # unreadable and this is the information the reader actually wants.
    worst = out.reindex(out["d_log10_gamma"].abs().sort_values(ascending=False).index).head(5)
    for _, r in worst.iterrows():
        ax.annotate(str(r["formula"]), (r["gamma_paper"], r["agl_gamma"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK_SOFT)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("$\\gamma$ from PINK Table 1 (experiment / literature)", color=INK_SOFT)
    ax.set_ylabel("$\\gamma$ from AFLOW-AGL", color=INK_SOFT)
    ax.set_title(f"n = {n} matched materials\nmean |$\\Delta$| = "
                 f"{ref['mean_abs_diff_gamma']:.3f}   r = {ref['pearson_r']:+.3f}",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel 2: the calibration, as one bar chart in shared log10 units.
    ax = axes[1]
    names, vals, cols = [], [], []
    # full population, so the bar pair is like-for-like (see step 59)
    names.append("G: cross-source noise"); vals.append(cfg["ref_G_full_cross_source"]); cols.append(ORANGE)
    names.append("G: model error");        vals.append(cfg["ref_G_full_model_error"]);  cols.append(BLUE)
    names.append("$\\gamma$: exp vs AGL"); vals.append(ref["mean_abs_diff_log10_gamma"]); cols.append(ORANGE)
    if head_mae_abs:
        names.append("$\\gamma$: head error"); vals.append(head_log); cols.append(BLUE)
    bars = ax.barh(names, vals, color=cols, height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("mean |$\\Delta$ log$_{10}$| (same units for both quantities)", color=INK_SOFT)
    ax.set_title("Label noise (orange) against\nthe error the model achieves (blue)",
                 color=INK, fontsize=11)
    ax.set_xlim(0, max(vals) * 1.28)

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.tight_layout()
    png_path = os.path.join(png_dir, "49_table1_gamma_vs_agl.png")
    fig.savefig(png_path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  Wrote {png_path}")


if __name__ == "__main__":
    main()
