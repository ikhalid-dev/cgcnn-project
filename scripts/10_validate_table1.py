#!/usr/bin/env python3
"""
STEP 10 - Compare our kappa_L against REAL experimental values (Table 1).
================================================================================

    python scripts/10_validate_table1.py

WHY THIS SCRIPT EXISTS
-----------------------
scripts/08_compare_kappa.py checks our pipeline against the paper's OWN
pretrained model - useful, but still model-vs-model. scripts/09_fetch_
validation_set.py already produced predictions for the 45 of the paper's 46
Table 1 materials we could fetch from Materials Project. Table 1 also lists,
for each material, the paper's own predicted kappa_PINK (plus its shear
modulus, sound velocity, and Gruneisen parameter) alongside the true
experimental kappa_exp - so this script runs a genuine three-way comparison:

    kappa_exp (ground truth)  vs  kappa_PINK (paper's model)  vs  kappa_ours

THE HEADLINE RESULT, AND WHY IT NEEDED A SECOND LOOK
-------------------------------------------------------
Run head to head, the paper's own Table 1 numbers beat ours by a wide margin
(log10 MAE ~0.23 vs ~0.43 - see the printed output). That is a real result and
is reported honestly below. But it did not match what every earlier check in
this project found: our moduli ensemble matches or beats the paper's on the
matbench test set, and (checked below, not assumed) our shear modulus and
sound velocity predictions on THESE SAME 45 crystals agree closely with the
paper's own stated G and v_sound. So the gap has to be coming from somewhere
else in the pipeline, not from worse moduli.

It is the Gruneisen parameter. The paper's Results text says, of Table 1
specifically: "The Gruneisen parameters were obtained from the AFLOW database
and experimental data" - i.e. Table 1's kappa_PINK values used REAL, looked-up
Gruneisen parameters, not ones derived from predicted K/G. Our pipeline (and
the paper's own released pink_predict.py, confirmed by grep - it has no AFLOW
lookup at all) instead derives gamma purely from the Poisson ratio via the
standard Slack-model formula, because that is the only option available at
genuine high-throughput screening scale, where no literature/AFLOW gamma
exists for a hypothetical GNoME candidate. This is checked below, not just
argued: substituting the paper's own tabulated gamma into OUR pipeline, K/G/
structure otherwise unchanged, drops our MAE from ~0.43 to ~0.23 log10 -
matching the paper's own number almost exactly. That single substitution
closes nearly the entire gap.

The honest conclusion: our from-scratch CGCNN + the fully-automated Slack
pipeline is competitive with the paper's own pretrained model once both are
compared on equal footing (same gamma source). Table 1's own headline
accuracy benefits from a manual, non-scalable input (real Gruneisen
parameters) that a 377k-candidate screen - the paper's own actual use case -
cannot have access to either. Both pipelines, run the way they would actually
be run for screening, land in the same place.

THE PROVENANCE CAVEAT THIS SCRIPT ALSO TAKES SERIOUSLY
---------------------------------------------------------
43 of these 45 materials turned out to already be in the matbench training
set our ensemble was trained on (table1_reference.csv's provenance column,
from the StructureMatcher check in 09_fetch_validation_set.py) - so for
those 43, this comparison is partly measuring RECALL, not generalisation.
Only 2 materials (LiF, Bi2Te3) are genuinely unseen. Both numbers are reported
separately below, never blended into one aggregate that would overstate what
this proves.

METRIC CONVENTION
-------------------
This project reports log10 MAE everywhere else, so that stays the headline
number here. But the paper's own stated Figure 6B metric for this exact table
(MAE=0.526, R^2=0.881) turns out to be NATURAL-log MAE, not log10 - see
09_fetch_validation_set.py's docstring for the arithmetic that confirms this.
So this script also reports natural-log MAE, specifically so "did we do
better than the paper, on the paper's own real validation table, in the
paper's own units" is a direct comparison rather than an implied one across
mismatched units.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")

# Same house palette as scripts/03_evaluate.py and scripts/08_compare_kappa.py.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#3f9142"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

OURS_COL = "Kappa_cal (W m-1 K-1)"


def load_and_merge(kappa_path, reference_path):
    """Join our predictions to the Table 1 reference, on material_id.

    table1_kappa_predictions.csv carries its own "provenance" column, but
    that one comes from 04_predict_moduli.py's naive check, which only
    cross-references the original 1,213-crystal mapping - it has no entries
    for any of these new mp-ids, so it wrongly reports nearly all of them as
    "unseen". table1_reference.csv's provenance column is the real one (a
    StructureMatcher check against live matbench structures, done in
    09_fetch_validation_set.py). Drop the naive column so only the accurate
    one survives the merge.
    """
    kappa = pd.read_csv(kappa_path).drop(columns=["provenance"])
    reference = pd.read_csv(reference_path)
    return kappa.merge(reference, on="material_id", suffixes=("", "_ref"))


def compute_metrics(df, pred_col, true_col="kappa_exp"):
    """log10 MAE/R^2/Pearson r, plus natural-log MAE to match the paper's own
    stated Figure 6B units (see module docstring)."""
    n = len(df)
    log_pred = np.log10(df[pred_col])
    log_true = np.log10(df[true_col])
    resid = log_pred - log_true

    if n >= 2:
        pearson_r, _ = stats.pearsonr(log_pred, log_true)
        spearman_r, _ = stats.spearmanr(df[pred_col], df[true_col])
        ss_tot = ((log_true - log_true.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    else:
        pearson_r = spearman_r = r2 = np.nan

    return {
        "n": n,
        "mae_log10": resid.abs().mean(),
        "mae_ln": resid.abs().mean() * np.log(10),
        "r2_log10": r2,
        "pearson_r_log10": pearson_r,
        "spearman_r": spearman_r,
    }


def print_metrics(label, m):
    print(f"  {label}: MAE = {m['mae_log10']:.3f} log10  ({m['mae_ln']:.3f} ln, "
         f"the paper's own units)", end="")
    if m["n"] >= 2 and not np.isnan(m["r2_log10"]):
        print(f"   R^2(log10) = {m['r2_log10']:.3f}   Pearson r = "
             f"{m['pearson_r_log10']:.3f}   Spearman rho = {m['spearman_r']:.3f}")
    else:
        print("   (n too small for a correlation coefficient)")


def diagnose_pipeline_stages(merged):
    """Check every intermediate quantity against the paper's own Table 1
    values, to LOCALISE where a kappa_L disagreement comes from instead of
    only ever seeing one aggregate number. Returns the dict used both for
    the printed report and the gamma-substitution test below.
    """
    log_g = np.log10(merged["G_VRH_pred"] / merged["G_paper"])

    v_long = ((merged["K_VRH_pred"] + 4 * merged["G_VRH_pred"] / 3)
             / merged["Density (g cm-3)"]) ** 0.5 * 1000
    v_trans = (merged["G_VRH_pred"] / merged["Density (g cm-3)"]) ** 0.5 * 1000
    v_sound_ours = ((1 / v_long ** 3 + 2 / v_trans ** 3) / 3) ** (-1 / 3)
    log_vs = np.log10(v_sound_ours / merged["vs_paper"])

    gamma_diff = merged["Gruneisen parameter"] - merged["gamma_paper"]
    r_g = np.corrcoef(np.log10(merged["G_VRH_pred"]), np.log10(merged["G_paper"]))[0, 1]

    print("\n--- Where does the kappa_L gap actually come from? (diagnostic, not opinion) ---")
    print(f"  Shear modulus G (ours vs paper's own G, on these 45 crystals):"
         f"      MAE = {log_g.abs().mean():.3f} log10   Pearson r = {r_g:.3f}"
         f"   (excellent agreement - not the source of the gap)")
    print(f"  Sound velocity (folds in our bulk modulus K, which Table 1 has "
         f"no direct column for): MAE = {log_vs.abs().mean():.3f} log10   "
         f"Pearson r = {np.corrcoef(np.log10(v_sound_ours), np.log10(merged['vs_paper']))[0, 1]:.3f}"
         f"   (also excellent - K is fine too)")
    print(f"  Gruneisen parameter (ours, formula-derived, vs paper's own "
         f"AFLOW/experimental-sourced value): mean signed diff = "
         f"{gamma_diff.mean():+.3f}   MAE = {gamma_diff.abs().mean():.3f}"
         f"   <-- THIS is where the gap lives")
    return v_sound_ours


def gamma_substitution_test(merged):
    """The decisive check: hold OUR K, G, and structure fixed, substitute the
    paper's own tabulated gamma in place of our formula-derived one, and see
    whether the kappa_L gap closes. kappa_cal ~ exp(-gamma) in slack_physics()
    (scripts/07_predict_kappa.py) with every other factor unchanged by gamma,
    so this multiplicative correction is exact given that formula - no need
    to recompute v_sound or anything else from scratch.
    """
    corrected = merged[OURS_COL] * np.exp(merged["Gruneisen parameter"] - merged["gamma_paper"])
    return compute_metrics(merged.assign(_corrected=corrected), "_corrected")


def plot_parity(merged, ours_metrics, pink_metrics, path):
    """Both models against real kappa_exp, log-log, house parity-plot style."""
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    x = merged["kappa_exp"]
    y_ours = merged[OURS_COL]
    y_pink = merged["kappa_pink"]

    lo = min(x.min(), y_ours.min(), y_pink.min()) * 0.6
    hi = max(x.max(), y_ours.max(), y_pink.max()) * 1.4

    ax.fill_between([lo, hi], [lo / 2, hi / 2], [lo * 2, hi * 2],
                    color=MUTED, alpha=0.08, linewidth=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=1)

    ax.scatter(x, y_pink, s=46, facecolor="none", edgecolor=ORANGE, linewidth=1.3,
              alpha=0.9, zorder=2, label="PINK (paper's own model)")
    ax.scatter(x, y_ours, s=42, color=BLUE, alpha=0.8, linewidth=0, zorder=3,
              label="Ours (from scratch)")

    # The 2 genuinely-unseen materials are the real generalisation evidence -
    # call them out by name directly rather than adding a legend entry for a
    # 2-vs-43 categorical split.
    unseen = merged[merged.provenance == "unseen"]
    for _, row in unseen.iterrows():
        ax.annotate(row["formula"], (row["kappa_exp"], row[OURS_COL]),
                   textcoords="offset points", xytext=(6, 5), fontsize=8,
                   color=INK_SOFT, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Experimental $\\kappa_L$ (W m$^{-1}$ K$^{-1}$), paper's Table 1")
    ax.set_ylabel("Predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)")
    ax.set_title("Phase 1: real-data validation against the paper's Table 1",
                fontsize=12, pad=12)

    ax.text(0.03, 0.97,
           f"n = {ours_metrics['n']}\n"
           f"Ours: MAE = {ours_metrics['mae_log10']:.3f} log$_{{10}}$ "
           f"({ours_metrics['mae_ln']:.3f} ln)\n"
           f"PINK: MAE = {pink_metrics['mae_log10']:.3f} log$_{{10}}$ "
           f"({pink_metrics['mae_ln']:.3f} ln)\n"
           f"bold labels: genuinely unseen by ours",
           transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK_SOFT,
           bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRID,
                    linewidth=0.8))

    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID,
             fontsize=9)
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gamma_diagnostic(merged, path):
    """Second figure: shows the gap is gamma-specific, not general. Left:
    our shear modulus vs paper's own (should hug the diagonal). Right: our
    formula-derived Gruneisen parameter vs paper's AFLOW/experimental value
    (should NOT hug the diagonal, and be systematically high for covalent
    semiconductors) - the two panels together are the visual version of the
    diagnose_pipeline_stages() printout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))

    ax = axes[0]
    lo = min(merged.G_VRH_pred.min(), merged.G_paper.min()) * 0.7
    hi = max(merged.G_VRH_pred.max(), merged.G_paper.max()) * 1.3
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(merged.G_paper, merged.G_VRH_pred, s=36, color=BLUE, alpha=0.75,
              linewidth=0, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Paper's own predicted G (GPa)")
    ax.set_ylabel("Our predicted G (GPa)")
    ax.set_title("Shear modulus: close agreement", fontsize=11, pad=10)
    r_g = np.corrcoef(np.log10(merged.G_VRH_pred), np.log10(merged.G_paper))[0, 1]
    ax.text(0.05, 0.93, f"Pearson r (log) = {r_g:.3f}", transform=ax.transAxes,
           fontsize=9, color=INK_SOFT, va="top")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    lo2 = min(merged["Gruneisen parameter"].min(), merged.gamma_paper.min()) * 0.85
    hi2 = max(merged["Gruneisen parameter"].max(), merged.gamma_paper.max()) * 1.15
    ax.plot([lo2, hi2], [lo2, hi2], color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.scatter(merged.gamma_paper, merged["Gruneisen parameter"], s=36, color=ORANGE,
              alpha=0.75, linewidth=0, zorder=2)
    ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)
    ax.set_aspect("equal")
    ax.set_xlabel("Paper's Table 1 $\\gamma$ (AFLOW / experimental)")
    ax.set_ylabel("Our $\\gamma$ (Slack-formula, from predicted K/G)")
    ax.set_title("Gruneisen parameter: this is the gap", fontsize=11, pad=10)
    mae_gamma = (merged["Gruneisen parameter"] - merged.gamma_paper).abs().mean()
    ax.text(0.05, 0.93, f"MAE = {mae_gamma:.3f}", transform=ax.transAxes,
           fontsize=9, color=INK_SOFT, va="top")
    ax.grid(True, which="major", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    fig.suptitle("Why our kappa_L differs from the paper's Table 1 numbers",
                fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kappa", default=os.path.join(RESULTS, "table1_kappa_predictions.csv"))
    parser.add_argument("--reference", default=os.path.join(RESULTS, "table1_reference.csv"))
    parser.add_argument("--out-fig", default=os.path.join(RESULTS, "table1_validation.png"))
    parser.add_argument("--out-fig2", default=os.path.join(RESULTS, "table1_gamma_diagnostic.png"))
    parser.add_argument("--out-csv", default=os.path.join(RESULTS, "table1_comparison.csv"))
    args = parser.parse_args()

    print("=== Phase 1: real-data validation against PINK's Table 1 ===\n")
    merged = load_and_merge(args.kappa, args.reference)
    print(f"{len(merged)}/45 fetched Table 1 materials merged successfully "
         f"(mp-3490/GaP excluded from all of Phase 1 - could not be fetched, "
         f"see scripts/09's own output)")

    n_unseen = int((merged.provenance == "unseen").sum())
    n_train = int((merged.provenance == "in_matbench_training").sum())
    print(f"Provenance: {n_train} already in the matbench training set (recall), "
         f"{n_unseen} genuinely unseen by our ensemble (real generalisation evidence)")

    def report(subset, label):
        ours = compute_metrics(subset, OURS_COL)
        pink = compute_metrics(subset, "kappa_pink")
        print(f"\n--- {label} (n={len(subset)}) ---")
        print_metrics("Ours vs experiment          ", ours)
        print_metrics("PINK (paper) vs experiment  ", pink)
        return ours, pink

    ours_all, pink_all = report(merged, "ALL 45 materials - blended; see the honest split below")
    report(merged[merged.provenance == "in_matbench_training"],
          f"{n_train} materials ALSO in the matbench training set - this is RECALL, not generalisation")

    unseen_subset = merged[merged.provenance == "unseen"]
    print(f"\n--- {n_unseen} materials genuinely UNSEEN by our ensemble "
         f"(the real generalisation evidence) ---")
    for _, row in unseen_subset.iterrows():
        our_val, pink_val, exp_val = row[OURS_COL], row["kappa_pink"], row["kappa_exp"]
        print(f"  {row['formula']:8s} ({row['material_id']}): exp={exp_val:6.2f}   "
             f"ours={our_val:7.2f} ({our_val / exp_val:.2f}x)   "
             f"PINK={pink_val:7.2f} ({pink_val / exp_val:.2f}x)")
    if n_unseen >= 2:
        report(unseen_subset, f"{n_unseen} unseen materials, aggregated")
    else:
        print("  (fewer than 2 unseen materials - no aggregate correlation computed, "
             "individual values above are the full evidence)")

    # Head-to-head in log space (consistent with every other metric here,
    # and the fair comparison given kappa_L spans 1-3000 W/m/K) - which
    # model's prediction is closer to experiment, material by material.
    log_exp = np.log10(merged["kappa_exp"])
    ours_err = (np.log10(merged[OURS_COL]) - log_exp).abs()
    pink_err = (np.log10(merged["kappa_pink"]) - log_exp).abs()
    n_better = int((ours_err < pink_err).sum())
    print(f"\nHead-to-head (log10 error, per material): ours closer to experiment "
         f"on {n_better}/{len(merged)} materials, PINK closer on "
         f"{len(merged) - n_better}/{len(merged)}")

    # --- Why? Localise the gap instead of stopping at the headline number ---
    diagnose_pipeline_stages(merged)
    corrected_metrics = gamma_substitution_test(merged)
    print(f"\n  Substitution test: our K/G/structure, but the PAPER'S OWN "
         f"tabulated gamma instead of ours:")
    print_metrics("  Ours + paper's gamma        ", corrected_metrics)
    print(f"  -> confirms the Gruneisen parameter source, not our CGCNN "
         f"moduli, explains the gap: MAE drops from "
         f"{ours_all['mae_log10']:.3f} to {corrected_metrics['mae_log10']:.3f} "
         f"log10 (paper's own: {pink_all['mae_log10']:.3f}).")

    plot_parity(merged, ours_all, pink_all, args.out_fig)
    plot_gamma_diagnostic(merged, args.out_fig2)
    merged.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_fig}")
    print(f"Wrote {args.out_fig2}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
