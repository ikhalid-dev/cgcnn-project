#!/usr/bin/env python3
"""
STEP 48 - Score the saved predictions against AFLOW-AGL's kappa, not our own.
================================================================================

    python scripts/cgcnn/48_agl_kappa_target.py

THE PROBLEM THIS EXISTS TO FIX
--------------------------------
`recall@10%` as this project has always computed it ranks predicted kappa
against a REFERENCE kappa that is our own Slack formula evaluated on matbench's
true DFT moduli - with gamma derived from those moduli through the same
empirical Poisson relation the prediction side uses. Both sides call the same
`slack_factors`. So:

  * the Slack model cancels: if Slack is wrong about a material, it is wrong
    on both sides and the metric never sees it;
  * the gamma derivation cancels HARDER: gamma is a deterministic function of
    K/G on both sides, so a model that predicts a BETTER gamma than the Poisson
    relation gives is scored as WORSE. The metric actively penalises the one
    lever this project has left.

The fix is an external reference. AFLOW-AGL tabulates a kappa_L computed by a
quasi-harmonic Debye-Gruneisen fit to first-principles energy-volume curves -
a different code, a different method, and crucially a gamma that came from
phonon thermodynamics rather than from K/G. Ranking our saved predictions
against THAT can see gamma.

WHAT AGL kappa IS AND IS NOT
------------------------------
It is NOT experiment. AGL is a model too, and it shares the isotropic-Debye
family of assumptions. What it does not share is our gamma, our Slack
prefactor, or our moduli - so it breaks the circularity, which is the whole
requirement. Read every number here as "agreement with an independent
first-principles model", never as "accuracy".

THE MATCHING RULE, AND ITS FAILURE MODES
------------------------------------------
AFLOW writes cell formulas ("He8Ne4"), matbench stores pymatgen Structures;
the common ground is the reduced formula. Two rules are computed and BOTH are
reported, because the choice changes the sample size and the sample size is
what decides whether the metric can resolve anything:

  strict : same reduced formula AND same atom count.  Rejects the case where
           AFLOW's relaxed cell is a supercell of matbench's, which is a real
           match being thrown away - but also rejects genuinely different
           polymorphs, which is the point.
  loose  : same reduced formula only.  Recovers the supercell cases and, with
           them, polymorph confusion: the project already measured what that
           costs on the moduli (|d log10 G| 0.1085 matched on formula+atom
           count vs 0.1349 on formula alone).

Known failure modes of both, stated rather than discovered later:
  1. Neither rule looks at the structure. Two distinct polymorphs with the
     same reduced formula and the same cell size are indistinguishable here
     and one of them will be scored against the other's kappa. A
     StructureMatcher pass would fix this and needs AFLOW POSCARs, which are
     a separate download (data_full/aflow_structures/ has some, not all).
  2. Multiple AFLOW entries can share a key. They are aggregated by median
     kappa, and the spread within a key is reported: a wide spread means the
     key is not identifying a unique material.
  3. Disordered/partial-occupancy matbench entries have no AFLOW counterpart
     at all and simply drop out - a silent, non-random sample restriction
     toward simple ordered compounds.

WHAT IS SCORED
----------------
Three predicted-kappa variants, so the reader can see which part of the
pipeline the ranking is sensitive to:

  proxy         G*v_sound*exp(-gamma_derived), no density or volume - EXACTLY
                what the current in-house metric ranks on.
  full          the complete Slack formula, with AFLOW's own density, cell
                volume and atom count supplied as the structural constants,
                gamma still derived from predicted K/G.
  full + AGL g  the same, but with AGL's tabulated gamma substituted in. This
                is an ORACLE, not a model result: it uses a number from the
                reference itself. It is here to answer one question - does the
                new target actually reward a better gamma? If recall jumps,
                the gamma head is worth training; if it does not, the target is
                as gamma-blind as the old one and this whole exercise failed.
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "aflow_csv": "data_full/aflow_agl.csv",       # AFLOW-AGL export: agl_thermal_conductivity_300K, agl_gruneisen, natoms, density
    "labels_csv": "data_full/labels.csv",         # matbench: mb_id, formula, n_sites, K_VRH, G_VRH
    "results_dir": "results/cgcnn",               # where every predictions_*.csv lives
    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/agl_target/csv",
    "png_dir": "results/cgcnn/agl_target/png",
    # ---- matching -----------------------------------------------------------
    "match_rule": "strict",       # "strict" = reduced formula + atom count; "loose" = reduced formula only
    "min_overlap_warn": 300,      # below this the decile is too small to resolve an effect - shouted about, loudly
    # ---- metrics ------------------------------------------------------------
    "deciles": [0.10, 0.20],      # recall@k fractions to report
    "split": "test",              # which split of the saved predictions to score
    "temperature_K": 300.0,       # AGL's tabulated kappa is at 300 K; kappa_full assumes the same
}
# =============================================================================

import os                    # path joining and directory creation
import sys                   # sys.path manipulation and early exit
from importlib import import_module   # importing 47's roster/loader by numeric module name

# torch first - duplicate libiomp5 aborts the process if numpy/pandas load MKL first.
import torch  # noqa: F401
import numpy as np           # ranking, set arithmetic, summary statistics
import pandas as pd          # the joins and every table written out
import matplotlib
matplotlib.use("Agg")        # headless machine: figures are files, never windows
import matplotlib.pyplot as plt   # noqa: E402
from scipy.stats import spearmanr  # noqa: E402   rank correlation - the metric that survives a monotone recalibration

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))   # so import_module can see 47_/37_

from cgcnn_scratch.joint import slack_factors  # noqa: E402   the in-house proxy kappa

_stability = import_module("47_missed_decile_stability")   # ROSTER + load_run, so the run list is defined once
_gamma = import_module("37_train_gamma")                   # kappa_full(), the complete Slack formula

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def reduced_formula(formula):
    """'He8Ne4' and 'He2Ne1' both -> 'He2Ne'. Same helper 35_merge_aflow.py uses."""
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None            # unparsable string; caller treats None as "cannot match"


def cell_volume_m3(compound, density_g_cm3):
    """Cell volume from AFLOW's composition + density: V = m/rho.

    AFLOW's export carries density but not volume, and the Slack prefactor
    needs V^(1/3). The cell mass follows from the compound string (which is
    the CELL composition, not the reduced one - 'He8Ne4' really is 12 atoms),
    so this is exact arithmetic, not an estimate.
    """
    from pymatgen.core import Composition
    N_A = 6.02214076e23
    try:
        grams_per_mole = Composition(compound).weight     # amu per cell == g per mole of cells
    except Exception:
        return np.nan
    grams = grams_per_mole / N_A                          # mass of ONE cell, in grams
    return (grams / float(density_g_cm3)) * 1e-6          # cm^3 -> m^3


def build_match_table(cfg):
    """Join AFLOW-AGL onto matbench. Returns (matched, diagnostics)."""
    aflow = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["aflow_csv"]))
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["labels_csv"]))

    aflow = aflow[np.isfinite(aflow["agl_thermal_conductivity_300K"])]
    aflow = aflow[aflow["agl_thermal_conductivity_300K"] > 0]      # a kappa of 0 is a failed fit, not a real value
    aflow["_rf"] = [reduced_formula(c) for c in aflow["compound"]]
    labels["_rf"] = [reduced_formula(f) for f in labels["formula"]]
    aflow = aflow.dropna(subset=["_rf"])
    labels = labels.dropna(subset=["_rf"])

    # The key is what "match_rule" says it is. Building it as a column keeps
    # the two rules on one code path, so a difference between them can only
    # come from the key itself.
    if cfg["match_rule"] == "strict":
        aflow["_key"] = aflow["_rf"] + "|" + aflow["natoms"].astype(int).astype(str)
        labels["_key"] = labels["_rf"] + "|" + labels["n_sites"].astype(int).astype(str)
    else:
        aflow["_key"] = aflow["_rf"]
        labels["_key"] = labels["_rf"]

    # Aggregate AFLOW duplicates per key. The spread is kept: a key whose
    # AFLOW entries disagree by 2x is not identifying one material.
    agg = aflow.groupby("_key").agg(
        agl_kappa=("agl_thermal_conductivity_300K", "median"),
        agl_kappa_min=("agl_thermal_conductivity_300K", "min"),
        agl_kappa_max=("agl_thermal_conductivity_300K", "max"),
        agl_gamma=("agl_gruneisen", "median"),
        agl_K=("ael_bulk_modulus_vrh", "median"),
        agl_G=("ael_shear_modulus_vrh", "median"),
        density=("density", "median"),
        natoms=("natoms", "median"),
        compound=("compound", "first"),
        spacegroup=("spacegroup_relax", "first"),
        n_aflow_entries=("compound", "size"),
    ).reset_index()

    matched = labels.merge(agg, on="_key", how="inner")
    matched["cell_volume_m3"] = [cell_volume_m3(c, d)
                                 for c, d in zip(matched["compound"], matched["density"])]

    diagnostics = {
        "n_aflow_with_kappa": int(len(aflow)),
        "n_aflow_keys": int(len(agg)),
        "n_matbench": int(len(labels)),
        "n_matched": int(len(matched)),
        "n_ambiguous_keys": int((agg["n_aflow_entries"] > 1).sum()),
        "median_kappa_spread_factor": float(
            (agg.loc[agg["n_aflow_entries"] > 1, "agl_kappa_max"]
             / agg.loc[agg["n_aflow_entries"] > 1, "agl_kappa_min"]).median())
        if (agg["n_aflow_entries"] > 1).any() else float("nan"),
    }
    return matched, diagnostics


def recall_at(true_vals, pred_vals, frac):
    """Fraction of the true bottom `frac` that the predicted bottom `frac` contains."""
    n = max(1, int(frac * len(true_vals)))
    truth = set(np.argsort(true_vals)[:n].tolist())
    picked = set(np.argsort(pred_vals)[:n].tolist())
    return len(truth & picked) / n, n


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 48 - an external kappa target: AFLOW-AGL instead of our own formula")
    print("=" * 78)
    print()

    # ---- THE OVERLAP COUNT, BEFORE ANYTHING ELSE ---------------------------
    tables = {}
    for rule in ["strict", "loose"]:
        c = dict(cfg); c["match_rule"] = rule
        tables[rule] = build_match_table(c)

    print("  MATCH COUNTS - reported first, because sample size decides everything")
    print("  " + "-" * 74)
    for rule, (matched, diag) in tables.items():
        print(f"    rule={rule:<7} {diag['n_matched']:>5} matbench crystals matched  "
              f"(AFLOW: {diag['n_aflow_with_kappa']} entries with a real kappa "
              f"-> {diag['n_aflow_keys']} keys)")
        print(f"    {'':<12} {diag['n_ambiguous_keys']} keys hold >1 AFLOW entry; "
              f"median kappa spread within such a key: "
              f"{diag['median_kappa_spread_factor']:.2f}x")
    print()

    matched, diag = tables[cfg["match_rule"]]

    # ---- how many of those are in the held-out TEST split? ------------------
    ref_label, _tier, ref = None, None, None
    runs = []
    reference_index = None
    seen = {}
    for label, tier, files in _stability.ROSTER:
        frame = _stability.load_run(os.path.join(PROJECT_ROOT, cfg["results_dir"]),
                                    files, cfg["split"])
        if frame is None:
            continue
        if reference_index is None:
            reference_index = frame.index
        elif not frame.index.equals(reference_index):
            sys.exit(f"ERROR: {label} has a different {cfg['split']} split - refusing to compare.")
        fp = _stability.fingerprint(np.concatenate([frame["pred_log10_K"].values,
                                                    frame["pred_log10_G"].values]))
        if fp in seen:
            continue                      # byte-identical duplicate of an earlier tag
        seen[fp] = label
        runs.append((label, tier, frame))
    if not runs:
        sys.exit("ERROR: no saved predictions found.")

    test_ids = pd.Index(reference_index)
    sub = matched[matched["mb_id"].isin(test_ids)].copy().set_index("mb_id")
    n_overlap = len(sub)

    print(f"  OVERLAP WITH THE HELD-OUT TEST SPLIT ({len(test_ids)} crystals)")
    print("  " + "-" * 74)
    print(f"    rule={cfg['match_rule']}: {n_overlap} test crystals have an AGL kappa")
    if n_overlap < cfg["min_overlap_warn"]:
        print()
        print("    " + "!" * 66)
        print(f"    !! WARNING: {n_overlap} < {cfg['min_overlap_warn']}. A 10% decile here is only "
              f"{max(1, int(0.10 * n_overlap))} crystals.")
        print("    !! recall@10% cannot resolve an effect at that size - one crystal")
        print("    !! moves it by several points. Read recall@20% and the Spearman")
        print("    !! correlation instead; both are reported below.")
        print("    " + "!" * 66)
    print()

    if n_overlap < 20:
        sys.exit("ERROR: fewer than 20 matched test crystals - nothing meaningful to compute.")

    # ---- reference and predicted kappa on the matched subset ----------------
    order = sub.index                                     # canonical row order for every array below
    agl_kappa = sub["agl_kappa"].values                   # the EXTERNAL reference
    agl_gamma = sub["agl_gamma"].values
    vol = sub["cell_volume_m3"].values
    nat = sub["natoms"].values
    dens = sub["density"].values

    ref_frame = runs[0][2].loc[order]                     # truth is identical across runs (47 checks this)
    true_K, true_G = ref_frame["true_log10_K"].values, ref_frame["true_log10_G"].values
    pre_t, anh_t, gam_t = slack_factors(true_K, true_G)
    inhouse_kappa = pre_t * anh_t                         # the SELF-REFERENTIAL target, on the same crystals

    rows = []
    for label, tier, frame in runs:
        f = frame.loc[order]
        pK, pG = f["pred_log10_K"].values, f["pred_log10_G"].values
        pre_p, anh_p, gam_p = slack_factors(pK, pG)
        k_proxy = pre_p * anh_p                                        # what the old metric ranks on
        with np.errstate(all="ignore"):
            k_full = _gamma.kappa_full(pK, pG, gam_p, vol, nat, dens)   # complete Slack, derived gamma
            k_oracle = _gamma.kappa_full(pK, pG, agl_gamma, vol, nat, dens)  # ORACLE: AGL's own gamma

        ok = (np.isfinite(k_proxy) & np.isfinite(k_full) & np.isfinite(k_oracle)
              & np.isfinite(agl_kappa) & (agl_kappa > 0))
        row = {"run": label, "tier": tier, "n_scored": int(ok.sum())}

        for tgt_name, tgt in [("agl", agl_kappa), ("inhouse", inhouse_kappa)]:
            for pred_name, pred in [("proxy", k_proxy), ("full", k_full), ("oracleGamma", k_oracle)]:
                if tgt_name == "inhouse" and pred_name != "proxy":
                    continue      # the in-house target is only ever paired with the proxy - that IS the old metric
                for frac in cfg["deciles"]:
                    r, n = recall_at(tgt[ok], pred[ok], frac)
                    row[f"recall{int(frac * 100)}_{tgt_name}_{pred_name}"] = round(100 * r, 1)
                    row[f"n_in_bin_{int(frac * 100)}"] = n
                rho = spearmanr(tgt[ok], pred[ok]).correlation
                row[f"spearman_{tgt_name}_{pred_name}"] = round(float(rho), 3)
        rows.append(row)

    scores = pd.DataFrame(rows)
    path = os.path.join(csv_dir, "48_agl_vs_inhouse_scores.csv")
    scores.to_csv(path, index=False)

    # ---- the side-by-side table --------------------------------------------
    n10 = max(1, int(0.10 * n_overlap))
    n20 = max(1, int(0.20 * n_overlap))
    print(f"  SCORED ON THE {n_overlap} MATCHED TEST CRYSTALS "
          f"(decile = {n10} crystals, quintile = {n20})")
    print("  " + "-" * 74)
    print(f"    {'run':<26} {'--- vs AGL kappa ---':^30} {'-- vs our own target --':^24}")
    print(f"    {'':<26} {'rec@10':>7} {'rec@20':>7} {'spear':>7} "
          f"{'  ':>4} {'rec@10':>7} {'rec@20':>7} {'spear':>7}")
    for _, r in scores.iterrows():
        print(f"    {r['run']:<26} {r['recall10_agl_full']:>6.1f}% {r['recall20_agl_full']:>6.1f}% "
              f"{r['spearman_agl_full']:>7.3f} {'  ':>4} "
              f"{r['recall10_inhouse_proxy']:>6.1f}% {r['recall20_inhouse_proxy']:>6.1f}% "
              f"{r['spearman_inhouse_proxy']:>7.3f}")
    print()
    print("    (vs AGL uses the FULL Slack formula with AFLOW's own density/volume/atom")
    print("     count; vs our own target uses the proxy, which is what that metric is.)")
    print()

    # ---- does the new target actually see gamma? ---------------------------
    print("  DOES THE AGL TARGET REWARD A BETTER GAMMA?")
    print("  " + "-" * 74)
    print(f"    {'run':<26} {'derived gamma':>14} {'AGL gamma (oracle)':>20} {'change':>9}")
    for _, r in scores.iterrows():
        d = r["recall20_agl_oracleGamma"] - r["recall20_agl_full"]
        print(f"    {r['run']:<26} {r['recall20_agl_full']:>13.1f}% "
              f"{r['recall20_agl_oracleGamma']:>19.1f}% {d:>+8.1f}")
    print("    (recall@20% against AGL kappa. The right-hand column substitutes AGL's")
    print("     OWN gamma into our formula - an oracle, not a model result. A large")
    print("     positive change means the target is gamma-sensitive and a gamma head")
    print("     is worth training; ~0 would mean this target is as blind as the old one.)")
    print()
    # The same question asked of the OLD target, which is the claim being tested.
    pre_t2, anh_t2, gam_t2 = slack_factors(true_K, true_G)
    ok2 = np.isfinite(inhouse_kappa) & (inhouse_kappa > 0)
    r_derived, _ = recall_at(inhouse_kappa[ok2],
                             (pre_t2 * anh_t2)[ok2], 0.20)
    with np.errstate(all="ignore"):
        k_true_aglgamma = _gamma.kappa_full(true_K, true_G, agl_gamma, vol, nat, dens)
    ok3 = ok2 & np.isfinite(k_true_aglgamma)
    r_aglg, _ = recall_at(inhouse_kappa[ok3], k_true_aglgamma[ok3], 0.20)
    print(f"    Control on the OLD target: feed it the TRUE moduli and it scores "
          f"{100 * r_derived:.1f}%;")
    print(f"    feed it the true moduli AND AGL's better gamma and it scores "
          f"{100 * r_aglg:.1f}%.")
    print("    A perfect model is penalised for using a better gamma - that is the")
    print("    circularity, measured rather than argued.")
    print()

    # ---- how far apart are the two targets themselves? ---------------------
    both = np.isfinite(agl_kappa) & (agl_kappa > 0) & np.isfinite(inhouse_kappa) & (inhouse_kappa > 0)
    rho_targets = spearmanr(agl_kappa[both], inhouse_kappa[both]).correlation
    print("  THE TWO TARGETS, COMPARED DIRECTLY")
    print("  " + "-" * 74)
    print(f"    spearman(AGL kappa, in-house kappa) = {rho_targets:.3f} over {int(both.sum())} crystals")
    print("    (the in-house proxy drops density and volume, so this is not expected")
    print("     to be 1.0 even if both models were perfect - it bounds how much of a")
    print("     recall difference is target definition rather than model quality.)")
    print()

    # ---- write the per-crystal matched table -------------------------------
    out = sub.reset_index()[["mb_id", "formula", "n_sites", "K_VRH", "G_VRH",
                             "compound", "spacegroup", "natoms", "density",
                             "agl_kappa", "agl_gamma", "agl_K", "agl_G",
                             "n_aflow_entries", "cell_volume_m3"]].copy()
    out["inhouse_kappa_proxy"] = inhouse_kappa
    out["gamma_derived_from_true_moduli"] = gam_t
    out["gamma_gap_agl_minus_derived"] = out["agl_gamma"] - out["gamma_derived_from_true_moduli"]
    match_path = os.path.join(csv_dir, "48_agl_matched_test_crystals.csv")
    out.to_csv(match_path, index=False)

    full_match_path = os.path.join(csv_dir, "48_agl_matched_all_matbench.csv")
    matched.to_csv(full_match_path, index=False)

    diag_path = os.path.join(csv_dir, "48_match_diagnostics.csv")
    pd.DataFrame([{**{"rule": rule}, **d} for rule, (_m, d) in tables.items()]).to_csv(diag_path, index=False)

    for p in [path, match_path, full_match_path, diag_path]:
        print(f"  Wrote {p}")

    # =========================================================================
    #  FIGURE
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.patch.set_facecolor("white")

    # Panel 1: the two targets against each other. If they ranked crystals the
    # same way, switching targets could not change any conclusion.
    ax = axes[0]
    ax.scatter(np.log10(inhouse_kappa[both]), np.log10(agl_kappa[both]), s=16,
               color=BLUE, alpha=0.5, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("our own target   log$_{10}$ $\\kappa$ (proxy units)", color=INK_SOFT)
    ax.set_ylabel("AFLOW-AGL   log$_{10}$ $\\kappa_L$ (W m$^{-1}$K$^{-1}$)", color=INK_SOFT)
    ax.set_title(f"The two targets rank differently\nspearman = {rho_targets:.3f}  "
                 f"(n = {int(both.sum())})", color=INK, fontsize=11)

    # Panel 2: recall@20% per run under each target, side by side.
    ax = axes[1]
    labels_short = [r["run"] for _, r in scores.iterrows()]
    x = np.arange(len(labels_short))
    ax.bar(x - 0.2, scores["recall20_agl_full"], width=0.38, color=BLUE, label="vs AGL kappa")
    ax.bar(x + 0.2, scores["recall20_inhouse_proxy"], width=0.38, color=ORANGE,
           label="vs our own target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=6.5, color=MUTED)
    ax.set_ylabel(f"recall@20% ({n20} of {n_overlap} crystals)", color=INK_SOFT)
    ax.set_title("Same predictions, two references", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    # Panel 3: the gamma the pipeline derives vs the gamma AGL computed. This
    # is the quantity the old metric cannot see at all.
    ax = axes[2]
    g_ok = np.isfinite(agl_gamma) & np.isfinite(gam_t) & (agl_gamma > 0)
    ax.scatter(gam_t[g_ok], agl_gamma[g_ok], s=16, color=GREEN, alpha=0.55,
               edgecolors="white", linewidths=0.3)
    lim = [0, max(np.nanmax(gam_t[g_ok]), np.nanmax(agl_gamma[g_ok])) * 1.05]
    ax.plot(lim, lim, color=INK_SOFT, lw=1.2, ls="--", label="equal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    mad = float(np.abs(agl_gamma[g_ok] - gam_t[g_ok]).mean())
    ax.set_xlabel("$\\gamma$ derived from TRUE K/G (Poisson relation)", color=INK_SOFT)
    ax.set_ylabel("$\\gamma$ from AFLOW-AGL", color=INK_SOFT)
    ax.set_title(f"The gamma the metric cannot see\nMAD = {mad:.2f}  (n = {int(g_ok.sum())})",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

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
    png_path = os.path.join(png_dir, "48_agl_vs_inhouse_target.png")
    fig.savefig(png_path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  Wrote {png_path}")


if __name__ == "__main__":
    main()
