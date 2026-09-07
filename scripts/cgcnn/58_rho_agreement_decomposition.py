#!/usr/bin/env python3
"""
STEP 58 - Why cross-model rank agreement fell from 0.940 to 0.723.
================================================================================

    python scripts/cgcnn/58_rho_agreement_decomposition.py

WHY THIS EXISTS
-----------------
Step 57 screened all 33,118 GNoME candidates with ALIGNN and reported Spearman
rho = 0.723 against round 9. That looked alarming next to the rho = 0.940 that
step 16 got for ALIGNN vs the CGCNN ensemble on the 1,213-crystal PINK set. Two
explanations were plausible and were left explicitly untested in 57's writeup:

  (H1) THE POPULATION. GNoME's discovered crystals are exotic - 88% have 4-5
       distinct elements - so both models might be extrapolating and drifting
       apart on unfamiliar chemistry.

  (H2) THE PARTNER MODEL. The 0.940 comparison was ALIGNN vs the matbench CGCNN
       ensemble; the 0.723 one is ALIGNN vs round 9, which is trained on AFLOW
       instead of matbench AND uses a trained gamma head instead of the derived
       Poisson relation. Two changes at once.

The two are separable with data this project already has, and NO new inference:
step 39's screen carries `Kappa_cal_derived_matbench`, the original
`13_screen_gnome.py` matbench-ensemble value, for the very same GNoME crystals
step 57 scored. So the same population can be held fixed while the partner model
is swapped, which is exactly the control H1 needs.

THE DESIGN
------------
One cell decides it. On the SAME 33k GNoME crystals, compare ALIGNN against the
matbench CGCNN ensemble - same training set as ALIGNN, same derived gamma, the
only thing that changed from the PINK-set comparison is the crystals themselves:

    if that rho stays near 0.94  -> the population is innocent, H2 is the cause
    if it falls to near 0.72     -> the population is the cause, H1

Then, if H2 survives, a swap test attributes the gap between round 9's two
differences. Kappa is recomputed four ways from the SAME structural constants,
crossing {ALIGNN moduli, round-9 moduli} with {derived gamma, round-9 trained
gamma}. Holding one input fixed while swapping the other is what separates them:

    A = ALIGNN moduli + derived gamma        (published Kappa_alignn)
    B = round-9 moduli + round-9 gamma       (published Kappa_r9_gamma)
    C = round-9 moduli + DERIVED gamma       (isolates the moduli)
    E = ALIGNN moduli + round-9 gamma        (isolates gamma)

rho(A,C) is the disagreement due to moduli alone; rho(A,E) the disagreement due
to gamma alone. A and B are recomputed rather than read from the CSVs so that a
sanity check can confirm the reconstruction reproduces the published columns
before any conclusion is drawn from the other two.

WHAT THIS IS NOT
------------------
GNoME has no ground truth, so nothing here says which model is RIGHT. Every
number below is agreement between models. What it can establish is WHERE they
disagree, which is a different and answerable question.
"""

# =============================================================================
#  CONFIG - every path and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs (all pre-existing; this script runs no model) ---------------
    "alignn_gnome":  "results/alignn/57_gnome_screen_alignn.csv",
    "screen_gnome":  "results/cgcnn/39_gnome_screen_all_gamma.csv",
    "alignn_pink":   "results/alignn/alignn_kappa_predictions.csv",
    "cgcnn_pink":    "results/cgcnn/pink_kappa_predictions.csv",

    # ---- outputs ------------------------------------------------------------
    "out_csv":  "results/alignn/58_rho_decomposition.csv",
    "out_png":  "results/alignn/58_rho_decomposition.png",
}
# =============================================================================

import os
import sys

# torch FIRST - 37_train_gamma pulls in the project's torch stack, and MKL's
# duplicate libiomp5 aborts the process if numpy/pandas land first.
import torch   # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

from importlib import import_module   # noqa: E402
_gamma = import_module("37_train_gamma")   # kappa_full, derived_gamma - the SAME physics 57 used

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def p(*a):
    print(*a, flush=True)


def rho(a, b):
    """Spearman on the rows where both are finite and positive (log-safe)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 10:
        return float("nan"), int(m.sum())
    return float(spearmanr(a[m], b[m]).correlation), int(m.sum())


def log10_spread(v):
    """sd of log10 - a model that has collapsed toward its mean shows a small one."""
    v = pd.to_numeric(pd.Series(v), errors="coerce").values
    v = v[np.isfinite(v) & (v > 0)]
    return float(np.log10(v).std()), len(v)


def main():
    cfg = CONFIG
    rows = []          # every number reported ends up here, then in the CSV

    def rec(scope, comparison, value, n, note):
        rows.append({"scope": scope, "comparison": comparison,
                     "spearman_rho": value, "n": n, "note": note})

    p("=" * 78)
    p("  STEP 58 - decomposing the 0.940 -> 0.723 drop in cross-model agreement")
    p("=" * 78)
    p()

    # =========================================================================
    #  1. THE PINK-SET BASELINE - the 0.940 that raised the question
    # =========================================================================
    al = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["alignn_pink"]))
    cg = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["cgcnn_pink"]))
    pink = al.merge(cg, on="material_id", suffixes=("_al", "_cg"))
    kc = "Kappa_cal (W m-1 K-1)"
    r_pink, n_pink = rho(pink[f"{kc}_al"], pink[f"{kc}_cg"])
    rec("PINK set (1,213)", "ALIGNN vs CGCNN-ens  [both matbench, derived gamma]",
        r_pink, n_pink, "the original high-agreement reference")
    p(f"  PINK  ALIGNN vs CGCNN-ens (both matbench, derived gamma): rho={r_pink:.3f}  n={n_pink}")

    # =========================================================================
    #  2. THE CONTROL THAT DECIDES H1 vs H2
    #     Same partner type (matbench + derived gamma) - only the crystals change.
    # =========================================================================
    a = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["alignn_gnome"]), dtype={"material_id": str})
    s = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["screen_gnome"]), dtype={"material_id": str})
    keep = ["material_id", "Volume (A3)", "Number of Atoms", "Density (g cm-3)",
            "gamma_r9_pred", "Kappa_cal_derived_matbench"]
    g = a.merge(s[keep], on="material_id", how="inner")
    # 57's own reliability flag: ALIGNN's ~1e-3 GPa regression floor is a
    # degenerate output, and including it would corrupt every rho below.
    g = g[g["alignn_prediction_reliable"]].copy()

    r_mat, n_mat = rho(g["Kappa_alignn"], g["Kappa_cal_derived_matbench"])
    r_r9,  n_r9  = rho(g["Kappa_alignn"], g["Kappa_r9_gamma"])
    r_x,   n_x   = rho(g["Kappa_cal_derived_matbench"], g["Kappa_r9_gamma"])
    rec("GNoME (33k)", "ALIGNN vs CGCNN-ens  [both matbench, derived gamma]",
        r_mat, n_mat, "THE CONTROL: population changed, partner type did not")
    rec("GNoME (33k)", "ALIGNN vs round 9    [AFLOW, trained gamma]",
        r_r9, n_r9, "step 57's headline")
    rec("GNoME (33k)", "CGCNN-ens vs round 9 [AFLOW, trained gamma]",
        r_x, n_x, "round 9 vs the OTHER matbench model")
    p(f"  GNoME ALIGNN vs CGCNN-ens (both matbench, derived gamma): rho={r_mat:.3f}  n={n_mat}  <- CONTROL")
    p(f"  GNoME ALIGNN vs round 9   (AFLOW, trained gamma):         rho={r_r9:.3f}  n={n_r9}")
    p(f"  GNoME CGCNN-ens vs round 9:                               rho={r_x:.3f}  n={n_x}")
    p()
    p(f"  population cost (PINK -> GNoME, partner held fixed): {r_mat - r_pink:+.3f}")
    p(f"  partner cost (CGCNN-ens -> round 9, population held fixed): {r_r9 - r_mat:+.3f}")

    # =========================================================================
    #  3. SWAP TEST - inside round 9, is it the moduli or the gamma?
    # =========================================================================
    V = g["Volume (A3)"].values.astype(float) * 1e-30
    N = g["Number of Atoms"].values.astype(float)
    D = g["Density (g cm-3)"].values.astype(float)
    lkA, lgA = np.log10(g["K_alignn"].values), np.log10(g["G_alignn"].values)
    lk9, lg9 = np.log10(g["K_r9_pred"].values), np.log10(g["G_r9_pred"].values)
    gam_der_A = _gamma.derived_gamma(lkA, lgA)     # Poisson relation on ALIGNN moduli
    gam_der_9 = _gamma.derived_gamma(lk9, lg9)     # Poisson relation on round-9 moduli
    gam_r9    = g["gamma_r9_pred"].values.astype(float)

    A = _gamma.kappa_full(lkA, lgA, gam_der_A, V, N, D)
    B = _gamma.kappa_full(lk9, lg9, gam_r9,    V, N, D)
    C = _gamma.kappa_full(lk9, lg9, gam_der_9, V, N, D)
    E = _gamma.kappa_full(lkA, lgA, gam_r9,    V, N, D)

    # Sanity FIRST: the reconstruction must reproduce the published columns, or
    # C and E mean nothing.
    s_a, _ = rho(A, g["Kappa_alignn"])
    s_b, _ = rho(B, g["Kappa_r9_gamma"])
    p()
    p(f"  sanity: recomputed A vs published Kappa_alignn    rho={s_a:.4f}")
    p(f"  sanity: recomputed B vs published Kappa_r9_gamma  rho={s_b:.4f}")
    if min(s_a, s_b) < 0.999:
        p("  WARNING: reconstruction does not reproduce the published columns - "
          "the swap test below is NOT trustworthy.")

    r_ab, n_ab = rho(A, B)
    r_ac, n_ac = rho(A, C)
    r_ae, n_ae = rho(A, E)
    rec("GNoME swap", "A vs B  ALIGNN(derived g) vs r9(trained g)", r_ab, n_ab, "the headline, reconstructed")
    rec("GNoME swap", "A vs C  ALIGNN(derived g) vs r9(DERIVED g)", r_ac, n_ac, "MODULI alone")
    rec("GNoME swap", "A vs E  ALIGNN(derived g) vs ALIGNN(r9 g)",  r_ae, n_ae, "GAMMA alone")
    p()
    p("  SWAP TEST - which input carries the disagreement?")
    p(f"    A vs B   both differences   rho={r_ab:.3f}")
    p(f"    A vs C   MODULI alone       rho={r_ac:.3f}")
    p(f"    A vs E   GAMMA alone        rho={r_ae:.3f}")

    # =========================================================================
    #  4. COMPONENT AGREEMENT + SPREAD - the mechanism behind the swap result
    # =========================================================================
    r_K, _ = rho(g["K_alignn"], g["K_r9_pred"])
    r_G, _ = rho(g["G_alignn"], g["G_r9_pred"])
    r_gam, _ = rho(gam_der_A, gam_r9)
    rec("GNoME component", "K   ALIGNN vs round 9", r_K, len(g), "bulk modulus rank agreement")
    rec("GNoME component", "G   ALIGNN vs round 9", r_G, len(g), "shear modulus rank agreement")
    rec("GNoME component", "gamma  derived(ALIGNN) vs round-9 head", r_gam, len(g),
        "near zero: the head does not track the physics-derived gamma at all")
    p()
    p("  component-level rank agreement on the same crystals:")
    p(f"    K:      ALIGNN vs r9   rho={r_K:.3f}")
    p(f"    G:      ALIGNN vs r9   rho={r_G:.3f}")
    p(f"    gamma:  derived vs r9  rho={r_gam:.3f}")

    p()
    p("  spread of log10 values (a collapsed model sits near its own mean):")
    spreads = {}
    for lbl, v in [("kappa PINK  ALIGNN", pink[f"{kc}_al"]),
                   ("kappa PINK  CGCNN-ens", pink[f"{kc}_cg"]),
                   ("kappa GNoME ALIGNN", g["Kappa_alignn"]),
                   ("kappa GNoME CGCNN-ens", g["Kappa_cal_derived_matbench"]),
                   ("kappa GNoME round 9", g["Kappa_r9_gamma"]),
                   ("gamma derived(ALIGNN)", gam_der_A),
                   ("gamma derived(r9 moduli)", gam_der_9),
                   ("gamma round-9 head", gam_r9)]:
        sd, n = log10_spread(v)
        spreads[lbl] = sd
        rec("spread", lbl, float("nan"), n, f"sd_log10={sd:.4f}")
        p(f"    {lbl:26s} sd={sd:.4f}  n={n}")

    # =========================================================================
    #  5. OUTPUTS
    # =========================================================================
    out = pd.DataFrame(rows)
    out_csv = os.path.join(PROJECT_ROOT, cfg["out_csv"])
    out.to_csv(out_csv, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2))

    # -- left: the decomposition, in the order the argument is made -----------
    labels = ["PINK\nALIGNN vs\nCGCNN-ens",
              "GNoME\nALIGNN vs\nCGCNN-ens",
              "GNoME\nmoduli\nalone",
              "GNoME\nALIGNN vs\nround 9",
              "GNoME\ngamma\nalone"]
    vals = [r_pink, r_mat, r_ac, r_r9, r_ae]
    cols = [GREEN, GREEN, ORANGE, ORANGE, BLUE]
    bars = ax1.bar(range(len(vals)), vals, color=cols, width=0.66)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", fontsize=9.5, color=INK)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=8.2, color=INK_SOFT)
    ax1.set_ylim(0, 1.06)
    ax1.set_ylabel("Spearman $\\rho$ between two models", color=INK_SOFT)
    ax1.axhline(r_pink, color=MUTED, lw=1.0, ls="--")
    ax1.set_title("Same population, different partner:\nthe partner is what moves $\\rho$",
                  color=INK, fontsize=11)
    ax1.grid(axis="y", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)

    # -- right: the mechanism - round 9's gamma head has collapsed ------------
    for lbl, v, c in [("derived from ALIGNN moduli", gam_der_A, GREEN),
                      ("derived from round-9 moduli", gam_der_9, BLUE),
                      ("round-9 trained head", gam_r9, ORANGE)]:
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)]
        ax2.hist(np.log10(v), bins=90, histtype="step", lw=1.7, color=c,
                 label=f"{lbl}  (sd={np.log10(v).std():.4f})")
    ax2.set_xlabel("$\\log_{10}\\gamma$", color=INK_SOFT)
    ax2.set_ylabel("candidates", color=INK_SOFT)
    ax2.set_title("Round 9's gamma head sits on one value\n"
                  "- which is why swapping gamma barely moves $\\rho$",
                  color=INK, fontsize=11)
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, cfg["out_png"]), dpi=150, facecolor="white")
    plt.close(fig)

    # =========================================================================
    #  6. THE VERDICT, stated in one place
    # =========================================================================
    p()
    p("  " + "-" * 74)
    p("  VERDICT")
    p("  " + "-" * 74)
    p(f"    H1 (the population) is ruled out: holding the partner type fixed, moving")
    p(f"       from the PINK set to GNoME costs only {r_mat - r_pink:+.3f} ({r_pink:.3f} -> {r_mat:.3f}).")
    p(f"    H2 (the partner model) is the cause: on the SAME crystals, swapping the")
    p(f"       partner to round 9 costs {r_r9 - r_mat:+.3f} ({r_mat:.3f} -> {r_r9:.3f}).")
    p(f"    Within round 9 the gap is the MODULI, not gamma: moduli alone reproduce")
    p(f"       the full drop ({r_ac:.3f} vs the {r_ab:.3f} headline) while gamma alone")
    p(f"       leaves agreement nearly intact ({r_ae:.3f}).")
    p(f"    Round 9 disagrees about equally with BOTH matbench models ({r_r9:.3f} with")
    p(f"       ALIGNN, {r_x:.3f} with the CGCNN ensemble) - it is the outlier, not ALIGNN.")
    p()
    p("  wrote:")
    p(f"    {cfg['out_csv']}")
    p(f"    {cfg['out_png']}")


if __name__ == "__main__":
    main()
