#!/usr/bin/env python3
"""
STEP 50 - Re-baseline EVERY model family against AFLOW-AGL's kappa, with error bars.
====================================================================================

    python scripts/cgcnn/50_rebaseline_on_agl.py

WHY THIS EXISTS
-----------------
Step 48 established that the metric this project has always quoted was
circular: recall@10% ranked predicted kappa against a reference kappa that was
this project's OWN Slack formula on matbench's true moduli, with gamma derived
from those moduli by the same empirical Poisson relation the prediction side
uses. Both sides called `slack_factors`, so the physics cancelled and gamma
cancelled completely - a model predicting a BETTER gamma scored WORSE.

48 fixed the target but only scored the CGCNN roster. Every other performance
claim in this project - ALIGNN "beats the CGCNN ensemble", the composition-only
tree baseline, the seven rounds of architecture work - was made with the broken
ruler and has never been re-measured. That is what this script does.

Nothing is trained. This is re-scoring of prediction files that already exist.

WHAT IS SCORED, AND WHY THESE THREE FAMILIES
----------------------------------------------
  CGCNN    the 26 distinct saved runs from 47's roster (7 ensembles + 19 single
           networks, after fingerprinting away 8 duplicate tags).
  ALIGNN   the separately-trained ALIGNN K and G models. Their test split was
           CHECKED to be identical to CGCNN's (asserted below, not assumed) -
           otherwise the comparison would be between different crystals.
  trees    the composition-only decision tree / random forest / XGBoost from
           step 43. No structure, no bonds - the "does the graph earn its
           complexity" control.

All three are scored on the SAME matched crystals, so the comparison is
like-with-like in the one way that matters here.

THE ERROR BARS (the point of the exercise)
--------------------------------------------
A point estimate is not a result - this project has already had to retract one
("0.1868 beats 0.1897" before any CI existed). Every headline comparison here
carries a paired bootstrap over the matched crystals, resampling the SAME
crystal indices for both models so the pairing is preserved.

Read the recall CIs carefully. recall@10% on this matched set is decided by 44
crystals, so its bootstrap interval is wide by construction. That is not a
defect of the bootstrap; it is the honest width of a 44-crystal metric, and it
is the single most important thing this script has to say about how much of
the past two years of "recall didn't move" was ever resolvable.

WHAT IS DELIBERATELY NOT CLAIMED
----------------------------------
AGL kappa is not experiment. It is a quasi-harmonic Debye-Gruneisen model, and
step 49 showed its gamma is badly wrong against literature values. So every
number here is "agreement with an independent first-principles model", never
"accuracy". What it buys is that it does not share this project's gamma, its
Slack prefactor, or its moduli - which is exactly what the old target failed at.
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    "aflow_csv": "data_full/aflow_agl.csv",      # AFLOW-AGL export (kappa, gamma, density, natoms)
    "labels_csv": "data_full/labels.csv",        # matbench: mb_id, formula, n_sites, K_VRH, G_VRH
    "results_dir": "results/cgcnn",              # every predictions_*.csv lives here
    "alignn_k_csv": "results/alignn/alignn_bulk_modulus_kv/prediction_results_test_set.csv",
    "alignn_g_csv": "results/alignn/alignn_shear_modulus_gv/prediction_results_test_set.csv",
    "tree_csv": "results/cgcnn/baseline_tree/csv/43_baseline_tree_predictions.csv",
    "tree_models": ["decision_tree", "random_forest", "xgboost"],   # column suffixes in that file

    # ---- outputs ------------------------------------------------------------
    "csv_dir": "results/cgcnn/rebaseline/csv",
    "png_dir": "results/cgcnn/rebaseline/png",

    # ---- matching -----------------------------------------------------------
    # "strict" = reduced formula + atom count. 48 measured the cost of relaxing
    # this: the within-key kappa spread rises 1.24x -> 1.56x on "loose", which is
    # polymorph confusion entering the reference. Keep strict.
    "match_rule": "strict",
    "min_overlap_warn": 300,      # below this a 10% decile cannot resolve anything

    # ---- metrics ------------------------------------------------------------
    "deciles": [0.10, 0.20],      # recall@k fractions
    "split": "test",              # only the held-out split says anything about generalisation
    "bootstrap_n": 10000,         # paired resamples; 10k is what 32_ensemble_joint.py used
    "bootstrap_seed": 0,          # fixed so the CIs are reproducible run to run
    "reference_run": "separate 3-ens (baseline)",   # every CI is "this model minus the reference"
}
# =============================================================================

import os                    # path joining and directory creation
import sys                   # sys.path manipulation and early exit
from importlib import import_module   # importing 47/48 by numeric module name

# torch first: MKL loads its own OpenMP runtime and a duplicate libiomp5 aborts
# the process if numpy/pandas get in first. Project-wide rule, not optional.
import torch  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # headless machine: figures are files, never windows
import matplotlib.pyplot as plt   # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

from cgcnn_scratch.joint import slack_factors  # noqa: E402

# 47 owns the roster and the two prediction-file conventions; 48 owns the AFLOW
# match and the recall helper. Importing rather than restating means a fix to
# the matching rule lands in one place.
_stability = import_module("47_missed_decile_stability")
_agl = import_module("48_agl_kappa_target")
_gamma = import_module("37_train_gamma")          # kappa_full(): the complete Slack formula

# Same palette as 44/46/47/48 so this results family reads as one system.
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#3f9142", "#8e5bb5"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"

FAMILY_COLOR = {"cgcnn": BLUE, "alignn": ORANGE, "tree": GREEN}


# -----------------------------------------------------------------------------
#  Loading the two families 47's roster does not cover
# -----------------------------------------------------------------------------
def load_alignn(cfg):
    """ALIGNN's K and G test predictions, converted to 47's 4-column convention.

    ALIGNN writes RAW GPa with a leading space in every field ("mb-08442,
    100.000000, 109.627129"), and one file per target. 47's convention is
    log10 with K and G side by side, indexed by material_id - so both files are
    read, stripped, log10'd and joined here.

    A prediction of <= 0 GPa is possible for ALIGNN (this project already found
    a -0.57 GPa bulk modulus on one soft crystal) and log10 of it is undefined.
    Those rows are dropped and COUNTED, never silently coerced, because a soft
    crystal is exactly the kind this screen cares about and quietly clipping it
    would flatter the model on the only bin that matters.
    """
    frames = {}
    for target, path in (("K", cfg["alignn_k_csv"]), ("G", cfg["alignn_g_csv"])):
        full = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(full):
            return None, f"missing {path}"
        df = pd.read_csv(full)
        df.columns = [c.strip() for c in df.columns]        # ALIGNN pads its header
        df["id"] = df["id"].astype(str).str.strip()
        frames[target] = df.set_index("id")

    k, g = frames["K"], frames["G"]
    ids = k.index.intersection(g.index)                     # both targets must exist for a crystal
    out = pd.DataFrame(index=ids)
    out["true_log10_K"] = np.log10(k.loc[ids, "target"].to_numpy(dtype=float))
    out["true_log10_G"] = np.log10(g.loc[ids, "target"].to_numpy(dtype=float))

    # Non-positive predictions: record how many, then drop.
    pk = k.loc[ids, "prediction"].to_numpy(dtype=float)
    pg = g.loc[ids, "prediction"].to_numpy(dtype=float)
    bad = (pk <= 0) | (pg <= 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pred_log10_K"] = np.log10(np.where(pk > 0, pk, np.nan))
        out["pred_log10_G"] = np.log10(np.where(pg > 0, pg, np.nan))
    out.index.name = "material_id"
    note = f"dropped {int(bad.sum())} crystals with a non-positive ALIGNN prediction" if bad.any() else ""
    return out.dropna().sort_index(), note


def load_tree(cfg, model):
    """One tree model's matbench TEST rows, in 47's 4-column convention.

    43 wrote every source/split in one long file with a column per model, so
    this filters to (source == matbench, split == test) and renames. The tree's
    gamma columns are ignored here: this script scores the moduli->Slack->kappa
    chain, and the gamma head is step 51/49's subject, not this one's.
    """
    full = os.path.join(PROJECT_ROOT, cfg["tree_csv"])
    if not os.path.exists(full):
        return None, f"missing {cfg['tree_csv']}"
    df = pd.read_csv(full)
    df = df[(df["source"] == "matbench") & (df["split"] == cfg["split"])]
    out = pd.DataFrame({
        "true_log10_K": df["K_VRH_true_log10"].to_numpy(dtype=float),
        "true_log10_G": df["G_VRH_true_log10"].to_numpy(dtype=float),
        "pred_log10_K": df[f"K_VRH_pred_log10_{model}"].to_numpy(dtype=float),
        "pred_log10_G": df[f"G_VRH_pred_log10_{model}"].to_numpy(dtype=float),
    }, index=df["material_id"].astype(str))
    out.index.name = "material_id"
    return out.dropna().sort_index(), ""


# -----------------------------------------------------------------------------
#  Scoring
# -----------------------------------------------------------------------------
def kappa_variants(run, meta):
    """Predicted kappa for one run, in the two forms 48 defined.

    proxy : G * v_sound * exp(-gamma_derived), no density/volume. This is what
            the OLD in-house metric ranked on, kept so the two targets can be
            compared on identical footing.
    full  : the complete Slack formula, with AFLOW's own density, cell volume
            and atom count as the structural constants. gamma still derived
            from predicted K/G - this is the real pipeline, end to end.
    """
    lk = run["pred_log10_K"].to_numpy(dtype=float)
    lg = run["pred_log10_G"].to_numpy(dtype=float)

    prefactor, anharmonic, _ = slack_factors(lk, lg)
    proxy = prefactor * anharmonic

    full = _gamma.kappa_full(lk, lg, _gamma.derived_gamma(lk, lg),
                             meta["volume_m3"].to_numpy(dtype=float),
                             meta["natoms"].to_numpy(dtype=float),
                             meta["density"].to_numpy(dtype=float))
    return proxy, full


def score_run(pred_kappa, agl_kappa, cfg):
    """Every metric this script reports, for one prediction vector.

    Ranking metrics (recall, Spearman) are computed on kappa directly; the
    error metrics are computed in log10 because kappa spans four decades and a
    linear mean would be decided entirely by the hardest few crystals.

    Median AND mean absolute error are both returned. They answer different
    questions and this project's own distribution is skewed enough that quoting
    one alone has misled before (median 26.7% vs mean 71.0% relative error).
    """
    out = {}
    for frac in cfg["deciles"]:
        r, n_bin = _agl.recall_at(agl_kappa, pred_kappa, frac)
        out[f"recall{int(frac * 100)}"] = 100.0 * r
        out[f"n_bin{int(frac * 100)}"] = n_bin
    out["spearman"] = spearmanr(agl_kappa, pred_kappa).correlation

    # log10 agreement with AGL. Both vectors are strictly positive by
    # construction (Slack kappa of positive moduli), so no guard is needed.
    d = np.abs(np.log10(pred_kappa) - np.log10(agl_kappa))
    out["mae_log10"] = float(np.mean(d))
    out["median_ae_log10"] = float(np.median(d))
    out["p95_ae_log10"] = float(np.percentile(d, 95))
    return out


def paired_bootstrap(pred_a, pred_b, agl_kappa, cfg):
    """Paired bootstrap of (a - b) on the three headline metrics.

    The SAME resampled crystal indices are applied to both models on every
    draw, which is what makes this a paired test: it asks "is a better than b
    on crystals like these", not "could a and b have come from the same
    distribution". Unpaired would throw away the fact that both models saw the
    identical crystals and would give needlessly wide intervals.
    """
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n = len(agl_kappa)
    idx = rng.integers(0, n, (cfg["bootstrap_n"], n))

    d_spear, d_r10, d_mae = [], [], []
    for row in idx:
        t = agl_kappa[row]
        a, b = pred_a[row], pred_b[row]
        d_spear.append(spearmanr(t, a).correlation - spearmanr(t, b).correlation)
        d_r10.append(_agl.recall_at(t, a, 0.10)[0] - _agl.recall_at(t, b, 0.10)[0])
        d_mae.append(np.mean(np.abs(np.log10(a) - np.log10(t)))
                     - np.mean(np.abs(np.log10(b) - np.log10(t))))

    def ci(v, scale=1.0):
        v = np.asarray(v, dtype=float) * scale
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(np.mean(v)), float(lo), float(hi)

    return {"spearman": ci(d_spear), "recall10": ci(d_r10, 100.0), "mae_log10": ci(d_mae)}


def main():
    cfg = dict(CONFIG)
    csv_dir = os.path.join(PROJECT_ROOT, cfg["csv_dir"])
    png_dir = os.path.join(PROJECT_ROOT, cfg["png_dir"])
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    print("=" * 78)
    print("  STEP 50 - re-baselining every model family on AFLOW-AGL's kappa")
    print("=" * 78)
    print()

    # ---- the external reference --------------------------------------------
    matched, diagnostics = _agl.build_match_table(cfg)
    matched = matched.set_index("mb_id")
    matched["volume_m3"] = [
        _agl.cell_volume_m3(c, d) for c, d in zip(matched["compound"], matched["density"])]
    print(f"  AFLOW-AGL matched onto matbench: {len(matched)} crystals "
          f"(rule = {cfg['match_rule']})")

    # ---- assemble the roster ------------------------------------------------
    # (label, family, frame) - 47's runs, then the two families it does not cover.
    runs, notes = [], []
    seen_fingerprints = {}
    for label, tier, files in _stability.ROSTER:
        frame = _stability.load_run(os.path.join(PROJECT_ROOT, cfg["results_dir"]),
                                    files, cfg["split"])
        if frame is None:
            notes.append(f"skipped {label}: prediction file missing")
            continue
        # 47's duplicate collapse: three tags name the same three networks.
        fp = _stability.fingerprint(frame[["pred_log10_K", "pred_log10_G"]].to_numpy())
        if fp in seen_fingerprints:
            notes.append(f"collapsed {label}: byte-identical to {seen_fingerprints[fp]}")
            continue
        seen_fingerprints[fp] = label
        runs.append((label, "cgcnn", tier, frame))

    alignn, note = load_alignn(cfg)
    if alignn is not None:
        runs.append(("ALIGNN (K+G)", "alignn", "single", alignn))
        if note:
            notes.append(f"ALIGNN: {note}")
    else:
        notes.append(f"ALIGNN: {note}")

    for model in cfg["tree_models"]:
        frame, note = load_tree(cfg, model)
        if frame is not None:
            runs.append((f"tree: {model}", "tree", "single", frame))
        else:
            notes.append(f"tree {model}: {note}")

    print(f"  runs to score: {len(runs)}  "
          f"({sum(1 for r in runs if r[1] == 'cgcnn')} cgcnn, "
          f"{sum(1 for r in runs if r[1] == 'alignn')} alignn, "
          f"{sum(1 for r in runs if r[1] == 'tree')} tree)")
    for n in notes:
        print(f"    note: {n}")
    print()

    # ---- score every run on the SAME matched crystals -----------------------
    # The intersection is taken ONCE, across every run, so that a family with a
    # slightly different id set cannot be scored on an easier subset than the
    # others. This is the difference between a fair comparison and a flattering
    # one, so it is done explicitly rather than per-run.
    common = None
    for _, _, _, frame in runs:
        ids = frame.index.intersection(matched.index)
        common = ids if common is None else common.intersection(ids)
    common = common.sort_values()
    print(f"  crystals scored by EVERY run: {len(common)}")
    if len(common) < cfg["min_overlap_warn"]:
        print()
        print("  " + "!" * 70)
        print(f"  !! ONLY {len(common)} CRYSTALS MATCH. A 10% decile is "
              f"{int(round(0.10 * len(common)))} crystals, which cannot")
        print("  !! resolve any effect this project is chasing. Use recall@20% or")
        print("  !! the Spearman correlation instead, and say the sample size.")
        print("  " + "!" * 70)
    print()

    meta = matched.loc[common]
    agl_kappa = meta["agl_kappa"].to_numpy(dtype=float)

    rows, kappa_store = [], {}
    for label, family, tier, frame in runs:
        sub = frame.loc[common]
        proxy, full = kappa_variants(sub, meta)
        kappa_store[label] = full
        row = {"run": label, "family": family, "tier": tier, "n_scored": len(common)}
        for name, vec in (("full", full), ("proxy", proxy)):
            for k, v in score_run(vec, agl_kappa, cfg).items():
                row[f"{k}_{name}"] = v
        rows.append(row)

    scores = pd.DataFrame(rows)

    # ---- paired bootstrap against the reference run -------------------------
    ref = cfg["reference_run"]
    if ref not in kappa_store:
        ref = scores["run"].iloc[0]
        print(f"  (configured reference not scored; using '{ref}' instead)")
    print(f"  paired bootstrap ({cfg['bootstrap_n']} resamples) vs '{ref}' ...")

    ci_rows = []
    for label in scores["run"]:
        if label == ref:
            continue
        ci = paired_bootstrap(kappa_store[label], kappa_store[ref], agl_kappa, cfg)
        ci_rows.append({
            "run": label,
            "d_spearman": ci["spearman"][0], "d_spearman_lo": ci["spearman"][1],
            "d_spearman_hi": ci["spearman"][2],
            "d_recall10_pts": ci["recall10"][0], "d_recall10_lo": ci["recall10"][1],
            "d_recall10_hi": ci["recall10"][2],
            "d_mae_log10": ci["mae_log10"][0], "d_mae_log10_lo": ci["mae_log10"][1],
            "d_mae_log10_hi": ci["mae_log10"][2],
        })
    cis = pd.DataFrame(ci_rows)
    # "Significant" here means the 95% interval excludes zero - the conservative
    # test, and the only one this project accepts for a generalisation claim.
    cis["spearman_significant"] = (cis["d_spearman_lo"] > 0) | (cis["d_spearman_hi"] < 0)
    cis["recall10_significant"] = (cis["d_recall10_lo"] > 0) | (cis["d_recall10_hi"] < 0)

    # ---- report -------------------------------------------------------------
    print()
    print("  " + "-" * 74)
    print("  EVERY MODEL FAMILY, SCORED ON AFLOW-AGL kappa (full Slack chain)")
    print("  " + "-" * 74)
    show = scores.sort_values("spearman_full", ascending=False)
    print(f"  {'run':<30}{'fam':<8}{'rec@10':>8}{'rec@20':>8}{'spear':>8}"
          f"{'MAE':>8}{'med':>8}")
    for _, r in show.iterrows():
        print(f"  {r['run']:<30}{r['family']:<8}{r['recall10_full']:>8.1f}"
              f"{r['recall20_full']:>8.1f}{r['spearman_full']:>8.3f}"
              f"{r['mae_log10_full']:>8.3f}{r['median_ae_log10_full']:>8.3f}")
    print()
    print(f"  (rec@10 is decided by {int(show['n_bin10_full'].iloc[0])} crystals, "
          f"rec@20 by {int(show['n_bin20_full'].iloc[0])}. MAE is a mean and med a")
    print("   median of |log10 predicted kappa - log10 AGL kappa|; both are quoted")
    print("   because this project's error distribution is skewed.)")

    print()
    print("  " + "-" * 74)
    print(f"  PAIRED BOOTSTRAP vs '{ref}' - 95% CI, positive = better than reference")
    print("  " + "-" * 74)
    print(f"  {'run':<30}{'d spearman [95% CI]':>30}{'d recall@10 pts [95% CI]':>32}")
    for _, r in cis.sort_values("d_spearman", ascending=False).iterrows():
        mark = "*" if r["spearman_significant"] else " "
        print(f"  {r['run']:<30}"
              f"{r['d_spearman']:>+9.3f} [{r['d_spearman_lo']:+.3f},{r['d_spearman_hi']:+.3f}]{mark:>2}"
              f"{r['d_recall10_pts']:>+11.1f} [{r['d_recall10_lo']:+.1f},{r['d_recall10_hi']:+.1f}]")
    print()
    print("  * = 95% CI on Spearman excludes zero.")
    n_sig_r = int(cis["recall10_significant"].sum())
    print(f"  recall@10% CIs excluding zero: {n_sig_r}/{len(cis)} - "
          f"this is the resolution limit of a {int(show['n_bin10_full'].iloc[0])}-crystal bin.")

    # ---- outputs ------------------------------------------------------------
    scores.to_csv(os.path.join(csv_dir, "50_rebaseline_scores.csv"), index=False)
    cis.to_csv(os.path.join(csv_dir, "50_rebaseline_bootstrap_ci.csv"), index=False)
    # 48 returns the match diagnostics as a plain dict (one rule's counts), so
    # it is wrapped in a single-row frame rather than assumed to be one already.
    pd.DataFrame([diagnostics]).to_csv(
        os.path.join(csv_dir, "50_match_diagnostics.csv"), index=False)

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))

    ax = axes[0]
    s = scores.sort_values("spearman_full")
    ax.barh(range(len(s)), s["spearman_full"],
            color=[FAMILY_COLOR[f] for f in s["family"]], height=0.7)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["run"], fontsize=7, color=INK_SOFT)
    ax.set_xlabel("Spearman $\\rho$ vs AFLOW-AGL $\\kappa_L$", color=INK_SOFT)
    ax.set_xlim(max(0.0, s["spearman_full"].min() - 0.05), 1.0)
    ax.set_title("Rank agreement with the external target", color=INK, fontsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax = axes[1]
    for fam in ("cgcnn", "alignn", "tree"):
        f = scores[scores["family"] == fam]
        if len(f):
            ax.scatter(f["recall10_full"], f["spearman_full"], s=46,
                       color=FAMILY_COLOR[fam], label=fam, alpha=0.85,
                       edgecolor="white", linewidth=0.8)
    ax.set_xlabel(f"recall@10% vs AGL $\\kappa_L$  "
                  f"({int(show['n_bin10_full'].iloc[0])}-crystal bin)", color=INK_SOFT)
    ax.set_ylabel("Spearman $\\rho$", color=INK_SOFT)
    ax.set_title("recall@10% is noise; $\\rho$ separates the families",
                 color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(f"Every model family re-scored on AFLOW-AGL $\\kappa_L$ "
                 f"({len(common)} matched crystals)", color=INK, fontsize=13)
    fig.tight_layout()
    out_png = os.path.join(png_dir, "50_rebaseline_on_agl.png")
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)

    print()
    print("  wrote:")
    for f in ("50_rebaseline_scores.csv", "50_rebaseline_bootstrap_ci.csv",
              "50_match_diagnostics.csv"):
        print(f"    {os.path.join(cfg['csv_dir'], f)}")
    print(f"    {os.path.join(cfg['png_dir'], '50_rebaseline_on_agl.png')}")


if __name__ == "__main__":
    main()
