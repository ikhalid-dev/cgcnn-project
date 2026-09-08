#!/usr/bin/env python3
"""
STEP 63 - Experimental kappa_L vs this project's best prediction, PINK Table 1 style.
================================================================================

    python scripts/cgcnn/63_table1_experimental_vs_predicted.py

WHY THIS EXISTS
-----------------
Almost everything in this project is model-vs-model or model-vs-database. PINK's
Table 1 is the one place with **measured** lattice thermal conductivity, and
step 09 already fetched 45 of its 46 materials (mp-3490/GaP is deprecated in
Materials Project, so 45 is the ceiling, not a sampling choice).

This assembles the comparison sheet: one row per material, the experimental
value beside this project's prediction and the paper's own.

WHICH PREDICTION IS "OURS"
----------------------------
Two moduli models are available and they are NOT equally good. ALIGNN beat the
CGCNN ensemble on both moduli on the identical matbench split (bulk 0.0539 vs
0.0630 log10 MAE, shear 0.0725 vs 0.0781) and was the only model improvement in
this project that survived re-scoring against AGL's independent kappa. But it
had never been run on Table 1. This script runs it, scores both, and reports
which actually wins on measured data rather than assuming ALIGNN does.

THE GAMMA COLUMN IS THE POINT OF THE WHOLE TABLE
--------------------------------------------------
Step 10 established the single most important fact about this comparison, and
any sheet that hides it is misleading:

    our kappa, derived gamma   MAE 0.427 log10   <- looks far worse than the paper
    our kappa, paper's gamma   MAE 0.226 log10   <- matches the paper's own 0.227

Table 1's gamma values come from AFLOW/experimental lookups - the paper's own
text says so - not from the Poisson-ratio formula that both our pipeline AND
the paper's released screening code actually use for real screening. So
comparing our formula-derived gamma against a table built with looked-up gamma
is not a like-for-like test of the model; it is a test of the gamma source.

Both are therefore reported per material:
    kappa_pred_*            our own pipeline end to end, nothing borrowed
    kappa_pred_*_papergamma the same moduli with Table 1's gamma substituted,
                            which is the fair comparison against kappa_pink

WHAT THE ERROR COLUMNS MEAN
-----------------------------
`abs_log10_err` is |log10(pred) - log10(exp)|. Errors on kappa span four orders
of magnitude across this table (0.2 to 2000 W/m/K), so a linear error would be
dominated entirely by diamond and BN. Log space is the only sane scale here, and
it is what the paper reports too.
"""

# =============================================================================
#  CONFIG - every path and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "reference_csv": "results/cgcnn/table1_reference.csv",       # kappa_exp, kappa_pink, gamma_paper
    "cgcnn_csv":     "results/cgcnn/table1_kappa_predictions.csv",  # the CGCNN ensemble's run
    "cif_dir":       "data/table1_validation",
    "alignn_k_dir":  "results/alignn/alignn_bulk_modulus_kv",
    "alignn_g_dir":  "results/alignn/alignn_shear_modulus_gv",

    # ---- outputs ------------------------------------------------------------
    "out_csv": "results/cgcnn/63_experimental_vs_predicted.csv",
    "out_png": "results/cgcnn/63_experimental_vs_predicted.png",

    "device": "cpu",     # 45 structures; a GPU would be pure overhead
}
# =============================================================================

import os
import sys

# torch FIRST - MKL loads its own OpenMP runtime and a duplicate libiomp5
# aborts the process if numpy/pandas/pymatgen get in first.
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "alignn"))

from importlib import import_module   # noqa: E402
_al = import_module("15_alignn_predict_moduli")    # load_alignn_target, predict_pair
_kap = import_module("07_predict_kappa")           # slack_physics - the unchanged physics

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def p(*a):
    print(*a, flush=True)


def mae_log10(pred, exp):
    """Mean |log10 error|. Only rows where both are finite and positive count."""
    pred, exp = np.asarray(pred, float), np.asarray(exp, float)
    m = np.isfinite(pred) & np.isfinite(exp) & (pred > 0) & (exp > 0)
    if m.sum() == 0:
        return np.nan, 0
    return float(np.abs(np.log10(pred[m]) - np.log10(exp[m])).mean()), int(m.sum())


def main():
    cfg = CONFIG
    ref = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["reference_csv"]))
    cg = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["cgcnn_csv"]))

    p("=" * 78)
    p("  STEP 63 - measured kappa_L vs predicted, on PINK Table 1")
    p("=" * 78)
    p()
    p(f"  materials with an experimental value: {len(ref)}")
    p(f"  (46 in the paper; mp-3490/GaP is deprecated in Materials Project)")

    d = ref.merge(cg, on="material_id", how="left", suffixes=("", "_cg"))

    # =========================================================================
    #  1. ALIGNN's moduli on the same 45 structures
    # =========================================================================
    device = torch.device(cfg["device"])
    k_model, k_cfg = _al.load_alignn_target(os.path.join(PROJECT_ROOT, cfg["alignn_k_dir"]), device)
    g_model, _ = _al.load_alignn_target(os.path.join(PROJECT_ROOT, cfg["alignn_g_dir"]), device)
    # Graph settings come from the MODEL'S OWN config: this project's ALIGNN was
    # trained at cutoff 5.0, and building 8.0-radius graphs for inference would
    # silently feed it a neighbourhood it never saw in training.
    p(f"  ALIGNN graph settings (from the model): {k_cfg}")

    from jarvis.core.atoms import pmg_to_atoms
    from pymatgen.core import Structure

    kk, gg = [], []
    for mid in d["material_id"]:
        path = os.path.join(PROJECT_ROOT, cfg["cif_dir"], f"{mid}.cif")
        if not os.path.exists(path):
            kk.append(np.nan); gg.append(np.nan)
            continue
        atoms = pmg_to_atoms(Structure.from_file(path))
        k_gpa, g_gpa = _al.predict_pair(k_model, g_model, atoms, k_cfg, device)
        kk.append(k_gpa); gg.append(g_gpa)
    d["K_alignn"], d["G_alignn"] = kk, gg
    p(f"  ALIGNN scored {int(np.isfinite(d['K_alignn']).sum())} of {len(d)} structures")

    # =========================================================================
    #  2. kappa from each moduli source, with BOTH gamma choices
    #     slack_physics() is imported unchanged - the physics is identical for
    #     every row, so any difference between columns is the moduli or gamma,
    #     never the formula.
    # =========================================================================
    # slack_physics(k_gpa, g_gpa, volume_a3, density, mass_amu, n_atoms) ->
    # dict with gruneisen / kappa_slack / kappa_cal. Imported unchanged, so the
    # physics is identical for every column below; only the moduli differ.
    V   = d["Volume (A3)"].to_numpy(float)
    RHO = d["Density (g cm-3)"].to_numpy(float)
    M   = d["Atomic mass (amu)"].to_numpy(float)
    N   = d["Number of Atoms"].to_numpy(float)

    def run(K, G):
        return _kap.slack_physics(np.asarray(K, float), np.asarray(G, float), V, RHO, M, N)

    res_cg = run(d["K_VRH_pred"], d["G_VRH_pred"])
    res_al = run(d["K_alignn"],  d["G_alignn"])

    # kappa_cal is PINK's own simplified formula and is the number this project
    # quotes everywhere, so it is what gets compared here.
    d["kappa_pred_cgcnn"]  = res_cg["kappa_cal"]
    d["kappa_pred_alignn"] = res_al["kappa_cal"]
    d["gamma_derived_cgcnn"]  = res_cg["gruneisen"]
    d["gamma_derived_alignn"] = res_al["gruneisen"]

    # ---- the paper-gamma variants -------------------------------------------
    # kappa_cal = C(moduli, structure) * exp(-gamma), so swapping gamma is an
    # EXACT rescale by exp(gamma_derived - gamma_paper). Doing it as a rescale
    # rather than a re-run guarantees literally nothing else changed between the
    # two columns. NOTE the exponential: kappa_slack would go as gamma^-2, but
    # kappa_cal - the column actually reported - does not.
    gp = pd.to_numeric(d["gamma_paper"], errors="coerce").to_numpy(float)
    for src in ["cgcnn", "alignn"]:
        gd = np.asarray(d[f"gamma_derived_{src}"], float)
        d[f"kappa_pred_{src}_papergamma"] = d[f"kappa_pred_{src}"] * np.exp(gd - gp)

    # =========================================================================
    #  3. errors and the verdict on which model is actually ours-at-its-best
    # =========================================================================
    cols = ["kappa_pred_cgcnn", "kappa_pred_alignn",
            "kappa_pred_cgcnn_papergamma", "kappa_pred_alignn_papergamma", "kappa_pink"]
    p()
    p("  " + "-" * 74)
    p("  MEAN |log10 error| AGAINST THE MEASURED VALUE")
    p("  " + "-" * 74)
    scores = {}
    for c in cols:
        m, n = mae_log10(d[c], d["kappa_exp"])
        scores[c] = m
        label = {"kappa_pink": "the PINK paper's own"}.get(c, c)
        p(f"    {label:34s} {m:.3f}   (n={n})")

    ours = min(["kappa_pred_cgcnn", "kappa_pred_alignn"], key=lambda c: scores[c])
    ours_pg = ours + "_papergamma"
    p()
    p(f"    our best self-contained model : {ours}  ({scores[ours]:.3f})")
    p(f"    same moduli, Table 1's gamma  : {scores[ours_pg]:.3f}"
      f"   vs the paper's {scores['kappa_pink']:.3f}")
    p()
    p("    Read the second line, not the first, when comparing against the paper:")
    p("    Table 1's gamma comes from lookups, not from the Poisson formula that")
    p("    our pipeline and the paper's own screening code both use.")

    d["kappa_pred_best"] = d[ours]
    d["kappa_pred_best_papergamma"] = d[ours_pg]
    # the alias columns inherit the score of whichever model won, so the figure
    # can label them without recomputing
    scores["kappa_pred_best"] = scores[ours]
    scores["kappa_pred_best_papergamma"] = scores[ours_pg]
    d["best_model"] = "ALIGNN" if ours.endswith("alignn") else "CGCNN ensemble"
    for c in ["kappa_pred_best", "kappa_pred_best_papergamma", "kappa_pink"]:
        d[f"abs_log10_err_{c.replace('kappa_', '')}"] = (
            np.log10(pd.to_numeric(d[c], errors="coerce")) - np.log10(d["kappa_exp"])).abs()

    # =========================================================================
    #  4. the sheet
    # =========================================================================
    out_cols = [
        "material_id", "formula", "provenance",
        "kappa_exp",                       # the measured value - the point of the table
        "kappa_pred_best", "kappa_pred_best_papergamma", "kappa_pink",
        "abs_log10_err_pred_best", "abs_log10_err_pred_best_papergamma",
        "abs_log10_err_pink",
        "best_model",
        "kappa_pred_cgcnn", "kappa_pred_alignn",
        "kappa_pred_cgcnn_papergamma", "kappa_pred_alignn_papergamma",
        "K_alignn", "G_alignn", "K_VRH_pred", "G_VRH_pred",
        "gamma_derived_cgcnn", "gamma_derived_alignn", "gamma_paper",
        "Kappa_cal_p05", "Kappa_cal_p95",
        "Number of Atoms", "Volume (A3)", "Density (g cm-3)",
    ]
    out = d[[c for c in out_cols if c in d.columns]].copy()
    # Column names say which gamma each prediction used, because that is the
    # single distinction a reader of this sheet has to keep straight:
    #   derived_gamma  - our pipeline end to end, nothing borrowed
    #   equal_footing  - the same moduli with Table 1's own gamma substituted,
    #                    which is the only like-for-like comparison to kappa_pink
    out = out.rename(columns={
        "kappa_pred_best":                     "kappa_pred_derived_gamma",
        "kappa_pred_best_papergamma":          "kappa_pred_equal_footing",
        "abs_log10_err_pred_best":             "abs_log10_err_derived_gamma",
        "abs_log10_err_pred_best_papergamma":  "abs_log10_err_equal_footing",
        "abs_log10_err_pink":                  "abs_log10_err_pink",
        "kappa_pred_cgcnn":                    "kappa_cgcnn_derived_gamma",
        "kappa_pred_alignn":                   "kappa_alignn_derived_gamma",
        "kappa_pred_cgcnn_papergamma":         "kappa_cgcnn_equal_footing",
        "kappa_pred_alignn_papergamma":        "kappa_alignn_equal_footing",
        "K_VRH_pred": "K_cgcnn", "G_VRH_pred": "G_cgcnn",
        "Kappa_cal_p05": "kappa_cgcnn_p05", "Kappa_cal_p95": "kappa_cgcnn_p95",
        "Number of Atoms": "n_atoms", "Volume (A3)": "volume_A3",
        "Density (g cm-3)": "density_g_cm3",
    })
    out = out.sort_values("kappa_exp").reset_index(drop=True)
    out.to_csv(os.path.join(PROJECT_ROOT, cfg["out_csv"]), index=False)

    # =========================================================================
    #  5. figure - predicted against measured, log-log
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    e = out["kappa_exp"]
    for col, colour, lbl, skey in [
        ("kappa_pred_derived_gamma", BLUE,
         f"this work ({d['best_model'].iloc[0]}, derived $\\gamma$)", "kappa_pred_best"),
        ("kappa_pred_equal_footing", GREEN,
         "this work, equal footing (Table 1's $\\gamma$)", "kappa_pred_best_papergamma"),
        ("kappa_pink", ORANGE, "PINK paper", "kappa_pink"),
    ]:
        v = pd.to_numeric(out[col], errors="coerce")
        m = np.isfinite(v) & np.isfinite(e) & (v > 0) & (e > 0)
        ax.scatter(e[m], v[m], s=34, alpha=0.75, color=colour, edgecolor="white",
                   linewidth=0.5, label=f"{lbl}   MAE {scores[skey]:.3f}")

    lim = [0.1, 3000]
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=0)
    for f, style in [(2, ":"), (0.5, ":")]:
        ax.plot(lim, [f * x for x in lim], color=GRID, lw=0.9, ls=style, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)", color=INK_SOFT)
    ax.set_ylabel("predicted $\\kappa_L$ (W m$^{-1}$ K$^{-1}$)", color=INK_SOFT)
    ax.set_title(f"PINK Table 1: {len(out)} materials with measured $\\kappa_L$\n"
                 "dashed = parity, dotted = factor of 2", color=INK, fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, cfg["out_png"]), dpi=150, facecolor="white")
    plt.close(fig)

    p()
    p("  wrote:")
    p(f"    {cfg['out_csv']}   ({len(out)} materials)")
    p(f"    {cfg['out_png']}")


if __name__ == "__main__":
    main()
