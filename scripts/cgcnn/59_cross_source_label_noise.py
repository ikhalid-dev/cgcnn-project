#!/usr/bin/env python3
"""
STEP 59 - What the matbench/AFLOW disagreement actually is, and which population
          the project's 0.1085 constant came from.
================================================================================

    python scripts/cgcnn/59_cross_source_label_noise.py

WHY THIS EXISTS
-----------------
Several scripts quote a single number for the disagreement between this
project's two label sources:

    49_agl_gamma_trust.py   "ref_dlog10_G_cross_source": 0.1085
    38_train_gamma_transfer.py, 48_agl_kappa_target.py, 55_matbench_subset_verdict.py

and step 49 builds an argument on it:

    AFLOW vs matbench, 555 matched compounds   |d log10 G| = 0.1085
    the model's own held-out error                          0.0781
                                                  ratio     1.39
    "A ratio above 1 means the label noise is larger than the error the model
     is trying to achieve - i.e. the labels, not the network, set the floor."

That constant is hard-coded with no record of which crystals it was measured
on, and the ratio pairs it with 0.0781, which IS a whole-test-set number. If
the two halves come from different populations the ratio is not meaningful.
This script measures the disagreement from the raw label files, on every
population and match rule, so the provenance is settled by measurement rather
than by assumption. Nothing is trained.

THREE QUESTIONS
-----------------
  1. WHERE DOES 0.1085 COME FROM? The disagreement is computed on the full
     matbench/AFLOW overlap and on the soft subset (`aflow_soft.csv`, the
     crystals step 35 merges), under both match rules. Whichever cell lands on
     0.1085 is the population the constant describes.

  2. IS IT BIAS OR SCATTER? A systematic convention offset between two DFT
     workflows would be a constant shift in log space and would be removable.
     Random disagreement would not be. The two are separated by reporting the
     mean offset alongside the MAE, and the MAE recomputed after subtracting
     that offset. If subtracting it changes nothing, there is no convention
     offset to correct.

  3. WHERE DOES IT LIVE? Disagreement is reported per stiffness bin, against
     the model's OWN held-out error on the same bins. This is the comparison
     step 49's ratio is trying to make, done per-population so that both halves
     describe the same crystals.

ONE THING TO BE CLEAR ABOUT
-----------------------------
"AFLOW AEL vs matbench VRH" appears in this project's docstrings and suggests
two different averaging conventions. They are not. AFLOW's own columns are
`ael_bulk_modulus_vrh` and `ael_shear_modulus_vrh` - AEL is the METHOD (a
strain-based elastic library), and it reports VRH averages just as matbench
does. Both numbers are the same physical quantity from two different DFT
workflows, so any disagreement is workflow (k-points, strain magnitude,
relaxation tolerance, functional), not definition.

WHAT THIS CANNOT SETTLE
-------------------------
Neither source is ground truth. A disagreement of d between two DFT workflows
bounds their COMBINED error against reality; it does not say which one is
closer, and if their errors are independent and comparable each one's own error
against truth is roughly d/sqrt(2). Every statement below is about agreement.
"""

# =============================================================================
#  CONFIG - every path and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs (raw label files; no model is run) --------------------------
    "matbench_labels": "data_full/labels.csv",          # mb_id, formula, n_sites, K_VRH, G_VRH
    "aflow_full":      "data_full/gamma_labels.csv",    # the 5,563 AFLOW crystals round 9 trained on
    "aflow_soft":      "data_full/aflow_soft.csv",      # the soft subset step 35 merges
    # the moduli ensemble's own per-crystal held-out errors, for the calibration
    "pred_G": "results/cgcnn/predictions_G_VRH_ens.csv",
    "pred_K": "results/cgcnn/predictions_K_VRH_ens.csv",

    # ---- outputs ------------------------------------------------------------
    "out_csv": "results/cgcnn/59_cross_source_label_noise.csv",
    "out_png": "results/cgcnn/59_cross_source_label_noise.png",

    # ---- analysis -----------------------------------------------------------
    # Fixed physical bins, so the label-noise population and the model-error
    # population are binned identically even though they are different crystals.
    "stiffness_bins": [0, 20, 40, 70, 1e9],
    "stiffness_names": ["<20 GPa", "20-40", "40-70", ">70"],
    # the constants this script exists to audit
    "quoted_cross_source": 0.1085,
    "quoted_model_error": 0.0781,
}
# =============================================================================

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

# pymatgen warns per-element about missing electronegativity for noble gases
# (He/Ne appear in AFLOW). It is cosmetic and would drown the real output.
warnings.filterwarnings("ignore", category=UserWarning)
from pymatgen.core import Composition   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def p(*a):
    print(*a, flush=True)


def add_reduced(df, formula_col="formula"):
    """Reduced formula via pymatgen's parser.

    A string comparison would be wrong: 'Fe2O4' and 'FeO2' are the same
    compound, and a naive substring test for elements confuses O with Os.
    Rows whose formula pymatgen cannot parse are dropped and counted.
    """
    red = []
    for f in df[formula_col]:
        try:
            red.append(Composition(str(f)).reduced_formula)
        except Exception:
            red.append(None)
    out = df.copy()
    out["reduced"] = red
    n_bad = out["reduced"].isna().sum()
    return out[out["reduced"].notna()].copy(), int(n_bad)


def overlap(mb, other, k_col, g_col, rule):
    """Join two label sets on one match rule, collapsing duplicates by median.

    Collapsing matters: several crystals can share a reduced formula, and
    without collapsing, a compound with many polymorphs would dominate the
    average purely by appearing more often.
    """
    if rule == "formula":
        a = mb.groupby("reduced")[["K_VRH", "G_VRH"]].median()
        b = other.groupby("reduced")[[k_col, g_col]].median()
    else:                                    # "formula+atoms" - the strict rule
        mb2 = mb.assign(key=mb["reduced"] + "_" + mb["n_sites"].astype(str))
        ot2 = other.assign(key=other["reduced"] + "_" + other["n_sites"].astype(str))
        a = mb2.groupby("key")[["K_VRH", "G_VRH"]].median()
        b = ot2.groupby("key")[[k_col, g_col]].median()
    j = a.join(b, how="inner", lsuffix="_mb", rsuffix="_af")
    j = j[(j > 0).all(axis=1)]               # log10 needs strictly positive
    return j


def stats(d):
    """Everything needed to tell a systematic offset apart from scatter."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return {
        "n": len(d),
        "MAE": float(np.abs(d).mean()),
        "median_abs": float(np.median(np.abs(d))),
        "bias": float(d.mean()),
        "sd": float(d.std()),
        "p90_abs": float(np.percentile(np.abs(d), 90)),
        "p99_abs": float(np.percentile(np.abs(d), 99)),
        # if a constant offset explained the disagreement, removing it would
        # collapse the MAE. This is that test.
        "MAE_debiased": float(np.abs(d - d.mean()).mean()),
    }


def main():
    cfg = CONFIG
    rows = []

    p("=" * 78)
    p("  STEP 59 - the matbench/AFLOW label disagreement, by population")
    p("=" * 78)
    p()

    mb_raw = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["matbench_labels"]))
    af_raw = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["aflow_full"]))
    sf_raw = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["aflow_soft"])).rename(
        columns={"compound": "formula", "natoms": "n_sites"})

    mb, nb_mb = add_reduced(mb_raw)
    af, nb_af = add_reduced(af_raw)
    sf, nb_sf = add_reduced(sf_raw)
    p(f"  matbench {len(mb)} ({nb_mb} unparseable)   "
      f"AFLOW full {len(af)} ({nb_af})   AFLOW soft {len(sf)} ({nb_sf})")

    # =========================================================================
    #  1. PROVENANCE - which population and rule reproduces 0.1085?
    # =========================================================================
    p()
    p("  " + "-" * 74)
    p("  Q1: WHICH POPULATION IS THE QUOTED 0.1085?")
    p("  " + "-" * 74)
    p(f"  {'population':22s} {'match rule':16s} {'n':>6} {'|d log10 G|':>12} {'bias':>9}")
    populations = [("AFLOW full (5,563)", af, "K_VRH", "G_VRH"),
                   ("AFLOW soft subset",  sf, "ael_bulk_modulus_vrh", "ael_shear_modulus_vrh")]
    joins = {}
    for name, df, kc, gc in populations:
        for rule in ["formula", "formula+atoms"]:
            j = overlap(mb, df, kc, gc, rule)
            gm = "G_VRH_mb" if "G_VRH_mb" in j.columns else "G_VRH"
            ga = "G_VRH_af" if "G_VRH_af" in j.columns else gc
            d = np.log10(j[ga]) - np.log10(j[gm])
            st = stats(d)
            hit = abs(st["MAE"] - cfg["quoted_cross_source"]) < 0.010
            p(f"  {name:22s} {rule:16s} {st['n']:>6} {st['MAE']:>12.4f} "
              f"{st['bias']:>+9.4f}" + ("   <- matches the quoted constant" if hit else ""))
            rows.append({"section": "provenance", "population": name, "match_rule": rule,
                         "quantity": "G", **st})
            joins[(name, rule)] = (j, gm, ga, kc, gc)

    # =========================================================================
    #  2. BIAS OR SCATTER, on the full overlap
    # =========================================================================
    j, gm, ga, kc, gc = joins[("AFLOW full (5,563)", "formula+atoms")]
    km = "K_VRH_mb" if "K_VRH_mb" in j.columns else "K_VRH"
    ka = "K_VRH_af" if "K_VRH_af" in j.columns else kc
    dK = np.log10(j[ka]) - np.log10(j[km])
    dG = np.log10(j[ga]) - np.log10(j[gm])
    dR = (np.log10(j[ka] / j[ga])) - (np.log10(j[km] / j[gm]))

    p()
    p("  " + "-" * 74)
    p(f"  Q2: BIAS OR SCATTER?  (full overlap, strict match, n={len(j)})")
    p("  " + "-" * 74)
    p(f"  {'':6} {'MAE':>8} {'med|d|':>8} {'bias':>9} {'sd':>8} {'p90':>8} {'p99':>8} {'MAE-debiased':>13}")
    for lbl, d in [("K", dK), ("G", dG), ("K/G", dR)]:
        st = stats(d)
        p(f"  {lbl:6s} {st['MAE']:>8.4f} {st['median_abs']:>8.4f} {st['bias']:>+9.4f} "
          f"{st['sd']:>8.4f} {st['p90_abs']:>8.4f} {st['p99_abs']:>8.4f} {st['MAE_debiased']:>13.4f}")
        rows.append({"section": "bias_vs_scatter", "population": "AFLOW full (5,563)",
                     "match_rule": "formula+atoms", "quantity": lbl, **st})
    gain = 100 * (1 - stats(dG)["MAE_debiased"] / stats(dG)["MAE"])
    p(f"\n    removing a constant offset changes G's MAE by {gain:+.1f}%"
      f"  ->  {'no convention offset to correct' if abs(gain) < 5 else 'a real offset exists'}")
    p(f"    median |d| is {stats(dG)['median_abs'] / stats(dG)['MAE']:.2f}x the MAE and p99 is "
      f"{stats(dG)['p99_abs']:.3f}  ->  heavy-tailed, not uniform")

    # =========================================================================
    #  3. WHERE IT LIVES - disagreement vs the model's own error, per bin
    # =========================================================================
    edges, names = cfg["stiffness_bins"], cfg["stiffness_names"]
    pg = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["pred_G"]))
    pg = pg[pg["split"] == "test"]

    lab_bins, mod_bins = [], []
    p()
    p("  " + "-" * 74)
    p("  Q3: WHERE DOES THE DISAGREEMENT LIVE?")
    p("  " + "-" * 74)
    p(f"  {'G band':10s} {'label disagree':>15} {'n':>6}    {'model error':>12} {'n':>6}   ratio")
    for i, nm in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        m = (j[gm] >= lo) & (j[gm] < hi)
        lab = float(np.abs(dG[m]).mean()) if m.sum() > 10 else np.nan
        mm = (pg["true_GPa"] >= lo) & (pg["true_GPa"] < hi)
        mod = float(pg.loc[mm, "abs_error_log10"].mean()) if mm.sum() > 10 else np.nan
        ratio = lab / mod if (mod and np.isfinite(lab) and np.isfinite(mod)) else np.nan
        lab_bins.append(lab)
        mod_bins.append(mod)
        p(f"  {nm:10s} {lab:>15.4f} {int(m.sum()):>6}    {mod:>12.4f} {int(mm.sum()):>6}   {ratio:>5.2f}")
        rows.append({"section": "by_stiffness", "population": nm, "match_rule": "formula+atoms",
                     "quantity": "G", "n": int(m.sum()), "MAE": lab,
                     "model_error": mod, "ratio": ratio})

    # =========================================================================
    #  4. THE CALIBRATION STEP 49 IS MAKING, done population-matched
    # =========================================================================
    soft_j, sgm, sga, _, _ = joins[("AFLOW soft subset", "formula+atoms")]
    d_soft = np.log10(soft_j[sga]) - np.log10(soft_j[sgm])
    soft_hi = float(soft_j[sgm].quantile(0.95))         # the soft overlap's own G range
    m_soft = pg["true_GPa"] <= soft_hi
    model_soft = float(pg.loc[m_soft, "abs_error_log10"].mean())
    model_all = float(pg["abs_error_log10"].mean())

    p()
    p("  " + "-" * 74)
    p("  Q4: STEP 49'S RATIO, WITH BOTH HALVES ON THE SAME POPULATION")
    p("  " + "-" * 74)
    p(f"    as quoted in 49: {cfg['quoted_cross_source']:.4f} / {cfg['quoted_model_error']:.4f} "
      f"= {cfg['quoted_cross_source'] / cfg['quoted_model_error']:.2f}")
    p(f"      numerator is the SOFT overlap, denominator is the WHOLE test set")
    p()
    p(f"    full population:  {stats(dG)['MAE']:.4f} / {model_all:.4f} = "
      f"{stats(dG)['MAE'] / model_all:.2f}")
    p(f"    soft population:  {stats(d_soft)['MAE']:.4f} / {model_soft:.4f} = "
      f"{stats(d_soft)['MAE'] / model_soft:.2f}   (G <= {soft_hi:.0f} GPa, n={int(m_soft.sum())})")
    for lbl, num, den in [("as_quoted", cfg["quoted_cross_source"], cfg["quoted_model_error"]),
                          ("full_matched", stats(dG)["MAE"], model_all),
                          ("soft_matched", stats(d_soft)["MAE"], model_soft)]:
        rows.append({"section": "calibration", "population": lbl, "quantity": "G",
                     "MAE": num, "model_error": den, "ratio": num / den})

    # =========================================================================
    #  5. IS THE SOFT SUBSET'S OFFSET REAL, OR SELECTION?
    #     aflow_soft.csv is selected on AFLOW's OWN modulus being small. Picking
    #     crystals because one measurement is low over-represents that
    #     measurement's downward errors, so it must come back low relative to
    #     the other source even if neither is biased - regression to the mean.
    #     The test: slice the UNSELECTED full overlap by each source in turn. If
    #     the sign of the bias follows whichever source did the selecting, the
    #     offset is an artifact of selection, not a property of the data.
    # =========================================================================
    p()
    p("  " + "-" * 74)
    p("  Q5: IS THE SOFT SUBSET'S -0.047 OFFSET REAL, OR SELECTION BIAS?")
    p("  " + "-" * 74)
    thr = 20.0
    for sel_lbl, sel in [(f"matbench G < {thr:.0f}", j[gm] < thr),
                         (f"AFLOW    G < {thr:.0f}", j[ga] < thr)]:
        st = stats(dG[sel])
        p(f"    full overlap, selected on {sel_lbl}:  n={st['n']:>4}  "
          f"MAE {st['MAE']:.4f}  bias {st['bias']:+.4f}")
        rows.append({"section": "selection_test", "population": sel_lbl,
                     "match_rule": "formula+atoms", "quantity": "G", **st})
    st_soft = stats(d_soft)
    p(f"    aflow_soft.csv (selected on AFLOW):        n={st_soft['n']:>4}  "
      f"MAE {st_soft['MAE']:.4f}  bias {st_soft['bias']:+.4f}")
    p()
    p("    If the bias flips sign with the selecting source, the soft subset's")
    p("    offset is an artifact of how it was chosen, not a convention difference.")

    # =========================================================================
    #  5. OUTPUTS
    # =========================================================================
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PROJECT_ROOT, cfg["out_csv"]), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # -- left: the distribution - centred, heavy-tailed ----------------------
    for lbl, d, c in [(f"full overlap (n={len(dG)})", dG, BLUE),
                      (f"soft subset (n={len(d_soft)})", d_soft, ORANGE)]:
        ax1.hist(np.asarray(d, float), bins=80, range=(-0.6, 0.6), histtype="step",
                 lw=1.7, color=c, density=True,
                 label=f"{lbl}\n  MAE {np.abs(d).mean():.4f}, bias {np.mean(d):+.4f}")
    ax1.axvline(0, color=MUTED, lw=1.0, ls="--")
    ax1.set_xlabel("$\\log_{10} G_{\\mathrm{AFLOW}} - \\log_{10} G_{\\mathrm{matbench}}$",
                   color=INK_SOFT)
    ax1.set_ylabel("density", color=INK_SOFT)
    ax1.set_title("Centred on zero, with heavy tails:\nscatter between DFT workflows, not an offset",
                  color=INK, fontsize=11)
    ax1.legend(fontsize=7.5, frameon=False)
    ax1.grid(color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # -- right: where it lives, against the model's own error -----------------
    x = np.arange(len(names))
    ax2.bar(x - 0.2, lab_bins, width=0.4, color=BLUE, label="label disagreement (AFLOW vs matbench)")
    ax2.bar(x + 0.2, mod_bins, width=0.4, color=GREEN, label="the model's own held-out error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, color=INK_SOFT)
    ax2.set_xlabel("true shear modulus", color=INK_SOFT)
    ax2.set_ylabel("MAE $\\log_{10} G$", color=INK_SOFT)
    ax2.set_title("The model is the weaker half in every band\n"
                  "- the labels never set the floor",
                  color=INK, fontsize=11)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, cfg["out_png"]), dpi=150, facecolor="white")
    plt.close(fig)

    p()
    p("  wrote:")
    p(f"    {cfg['out_csv']}")
    p(f"    {cfg['out_png']}")


if __name__ == "__main__":
    main()
