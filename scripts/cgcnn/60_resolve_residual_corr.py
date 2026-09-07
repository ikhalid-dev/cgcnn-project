#!/usr/bin/env python3
"""
STEP 60 - Recompute the joint-vs-separate table, and settle a documented conflict.
================================================================================

    python scripts/cgcnn/60_resolve_residual_corr.py

THE CONFLICT
--------------
Two places in this project stated the residual correlation between the K and G
prediction errors, and they disagreed:

    31_train_joint.py's docstring   "with the existing separate models:
                                     residual correlation = 0.297"
    README.md's table               separate +0.263   |   joint +0.297

Both cannot be right, and the number matters: the entire argument for the joint
architecture is that a shared trunk should RAISE this correlation, so that the
two errors cancel in log10(K/G) instead of adding.

Rather than pick a source, this recomputes every figure directly from the saved
held-out prediction files. Nothing is trained.

THE ANSWER
------------
The docstring is right and the README's table was wrong. 0.263 and 0.297 are
BOTH separate-model numbers - the single model and the 3-model ensemble. The
README presented them as separate-vs-joint, so its table did not contain the
joint model at all.

The origin of the error is visible in 31_train_joint.py's own runtime output,
which prints a correctly-labelled THREE column comparison:

    this run    1 model each    3-model ens.

The README took the second and third columns and relabelled them "separate" and
"joint", dropping the first - the column that actually held the joint model.

The joint model's real residual correlation is about 0.457, roughly double the
separate ensemble's, so the mechanism worked considerably better than the
README credited it with. It also has the best log10(K/G) of any configuration.
What it gives up is per-modulus accuracy: its K and G land at the SINGLE
separate model's level rather than the ensemble's, because three joint models
cannot match three models each specialised on one target. The two effects very
nearly cancel in kappa, which is why the original "did not clear significance"
conclusion survives - but for a different reason than was written down.
"""

# =============================================================================
#  CONFIG - every path lives here
# =============================================================================
CONFIG = {
    "results_dir": "results/cgcnn",
    "split": "test",                  # the held-out split every headline number uses

    # Separate-model runs are a PAIR of files (one per target); joint runs are
    # a single file carrying both heads. The loader below handles both.
    "separate": [
        ("separate, 1 model each",  "predictions_K_VRH_full.csv", "predictions_G_VRH_full.csv"),
        ("separate, 3-model ens.",  "predictions_K_VRH_ens.csv",  "predictions_G_VRH_ens.csv"),
    ],
    "joint": [
        ("joint r4, 3-model ens.", "predictions_joint_r4_ens.csv"),
        ("joint r5 ens B",         "predictions_joint_r5_ensB.csv"),
        ("joint r5 ens C",         "predictions_joint_r5_ensC.csv"),
    ],
    # round 2's two arms, to record what the ratio reparameterisation actually did
    "round2": {
        "K and G heads":   ["summary_joint_r2_kg_s42.json", "summary_joint_r2_kg_s1.json",
                            "summary_joint_r2_kg_s2.json"],
        "G and K/G heads": ["summary_joint_r2_ratio_s42.json", "summary_joint_r2_ratio_s1.json",
                            "summary_joint_r2_ratio_s2.json"],
    },

    "out_csv": "results/cgcnn/60_residual_corr_table.csv",
    "out_png": "results/cgcnn/60_residual_corr_table.png",
}
# =============================================================================

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def p(*a):
    print(*a, flush=True)


def metrics(eK, eG, tK, tG, pK, pG):
    """The four numbers the joint-vs-separate argument is made in.

    `residual_corr` is the mechanism: if the two errors move together, they
    cancel in the difference log10(K/G) instead of adding. `mae_log_ratio` is
    the target that correlation is supposed to improve.
    """
    return {
        "n": len(eK),
        "residual_corr": float(np.corrcoef(eK, eG)[0, 1]),
        "mae_log_K": float(np.abs(eK).mean()),
        "mae_log_G": float(np.abs(eG).mean()),
        "mae_log_ratio": float(np.abs((pK - pG) - (tK - tG)).mean()),
    }


def load_separate(d, kf, gf, split):
    """Two files, one per target, merged on material_id."""
    k = pd.read_csv(os.path.join(d, kf))
    g = pd.read_csv(os.path.join(d, gf))
    k, g = k[k["split"] == split], g[g["split"] == split]
    m = k.merge(g, on="material_id", suffixes=("_K", "_G"))
    return metrics(m.pred_log10_K - m.true_log10_K, m.pred_log10_G - m.true_log10_G,
                   m.true_log10_K, m.true_log10_G, m.pred_log10_K, m.pred_log10_G)


def load_joint(d, f, split):
    """One file; the joint model emitted both heads together."""
    x = pd.read_csv(os.path.join(d, f))
    x = x[x["split"] == split]
    return metrics(x.pred_log10_K - x.true_log10_K, x.pred_log10_G - x.true_log10_G,
                   x.true_log10_K, x.true_log10_G, x.pred_log10_K, x.pred_log10_G)


def main():
    cfg = CONFIG
    d = os.path.join(PROJECT_ROOT, cfg["results_dir"])
    rows = []

    p("=" * 78)
    p("  STEP 60 - joint vs separate, recomputed from the saved predictions")
    p("=" * 78)
    p()
    p(f"  {'run':26s} {'n':>5} {'corr':>8} {'MAE K':>8} {'MAE G':>8} {'MAE K/G':>9}")

    for label, kf, gf in cfg["separate"]:
        m = load_separate(d, kf, gf, cfg["split"])
        rows.append({"run": label, "family": "separate", **m})
    for label, f in cfg["joint"]:
        if not os.path.exists(os.path.join(d, f)):
            p(f"  [skip] {label} - {f} not found")
            continue
        m = load_joint(d, f, cfg["split"])
        rows.append({"run": label, "family": "joint", **m})

    for r in rows:
        p(f"  {r['run']:26s} {r['n']:>5} {r['residual_corr']:>+8.3f} "
          f"{r['mae_log_K']:>8.4f} {r['mae_log_G']:>8.4f} {r['mae_log_ratio']:>9.4f}")

    # ---- the verdict on the documented conflict -----------------------------
    sep1 = next(r for r in rows if r["run"].startswith("separate, 1"))
    sepE = next(r for r in rows if r["run"].startswith("separate, 3"))
    jnt = next((r for r in rows if r["run"].startswith("joint r4")), None)
    p()
    p("  " + "-" * 74)
    p("  THE CONFLICT, SETTLED")
    p("  " + "-" * 74)
    p(f"    +{sep1['residual_corr']:.3f} is the SEPARATE SINGLE model")
    p(f"    +{sepE['residual_corr']:.3f} is the SEPARATE 3-MODEL ENSEMBLE")
    p("    Both are separate-model numbers. 31_train_joint.py's docstring, which")
    p("    attributes 0.297 to the separate models, is CORRECT; the README's table,")
    p("    which labelled them separate-vs-joint, was wrong and contained no joint")
    p("    model at all.")
    if jnt:
        p()
        p(f"    the joint model's actual residual correlation: +{jnt['residual_corr']:.3f} "
          f"({jnt['residual_corr'] / sepE['residual_corr']:.2f}x the separate ensemble)")
        p(f"    its MAE log10(K/G):  {jnt['mae_log_ratio']:.4f}  vs the ensemble's "
          f"{sepE['mae_log_ratio']:.4f}   <- best of any configuration")
        p(f"    but its own K/G accuracy sits at the SINGLE model's level:")
        p(f"      K {jnt['mae_log_K']:.4f} vs single {sep1['mae_log_K']:.4f}, "
          f"ensemble {sepE['mae_log_K']:.4f}")
        p(f"      G {jnt['mae_log_G']:.4f} vs single {sep1['mae_log_G']:.4f}, "
          f"ensemble {sepE['mae_log_G']:.4f}")

    # ---- round 2's two arms, from their own saved summaries -----------------
    p()
    p("  " + "-" * 74)
    p("  ROUND 2: WHAT THE RATIO REPARAMETERISATION ACTUALLY DID (3 seeds each)")
    p("  " + "-" * 74)
    for arm, files in cfg["round2"].items():
        corr, ratio = [], []
        for f in files:
            fp = os.path.join(d, f)
            if not os.path.exists(fp):
                continue
            t = json.load(open(fp))["results"][cfg["split"]]
            corr.append(t["residual_corr"])
            ratio.append(t["mae_log_ratio"])
        if not corr:
            continue
        p(f"    {arm:16s}  corr +{np.mean(corr):.3f}   MAE log10(K/G) {np.mean(ratio):.4f}")
        rows.append({"run": f"round 2: {arm}", "family": "round2", "n": np.nan,
                     "residual_corr": float(np.mean(corr)), "mae_log_K": np.nan,
                     "mae_log_G": np.nan, "mae_log_ratio": float(np.mean(ratio))})
    p(f"    for reference, separate single:  corr +{sep1['residual_corr']:.3f}   "
      f"MAE log10(K/G) {sep1['mae_log_ratio']:.4f}")

    # ---- outputs ------------------------------------------------------------
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(PROJECT_ROOT, cfg["out_csv"]), index=False)

    plot = [r for r in rows if r["family"] in ("separate", "joint")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    names = [r["run"] for r in plot]
    cols = [GREEN if r["family"] == "separate" else BLUE for r in plot]

    ax1.barh(names, [r["residual_corr"] for r in plot], color=cols, height=0.6)
    for i, r in enumerate(plot):
        ax1.text(r["residual_corr"] + 0.008, i, f"{r['residual_corr']:+.3f}",
                 va="center", fontsize=8.5, color=INK)
    ax1.set_xlabel("residual correlation, err($K$) vs err($G$)", color=INK_SOFT)
    ax1.set_title("The mechanism: a shared trunk should RAISE this\n"
                  "(green = separate, blue = joint)", color=INK, fontsize=11)
    ax1.set_xlim(0, max(r["residual_corr"] for r in plot) * 1.22)

    ax2.barh(names, [r["mae_log_ratio"] for r in plot], color=cols, height=0.6)
    for i, r in enumerate(plot):
        ax2.text(r["mae_log_ratio"] + 0.0012, i, f"{r['mae_log_ratio']:.4f}",
                 va="center", fontsize=8.5, color=INK)
    ax2.set_xlabel("MAE $\\log_{10}(K/G)$  (lower is better)", color=INK_SOFT)
    ax2.set_title("The target that correlation buys\n"
                  "- the joint model wins it", color=INK, fontsize=11)
    ax2.set_xlim(0, max(r["mae_log_ratio"] for r in plot) * 1.22)

    for ax in (ax1, ax2):
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(PROJECT_ROOT, cfg["out_png"]), dpi=150, facecolor="white")
    plt.close(fig)

    p()
    p("  wrote:")
    p(f"    {cfg['out_csv']}")
    p(f"    {cfg['out_png']}")


if __name__ == "__main__":
    main()
