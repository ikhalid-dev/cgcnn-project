#!/usr/bin/env python
"""
Step 66 - the DFT queue as it actually stands, in one slide-ready table.

WHY THIS EXISTS
---------------
Step 61 built an 18-material DFT validation set in three arms. Step 64 then
settled one whole arm WITHOUT running any DFT, by checking round 9's predicted
moduli against every real measured modulus for the same chemistry. The
presentation still shows the original 18 and tells the audience that "only DFT
settles those", which is no longer true.

This script produces the CURRENT queue: which materials still need a
calculation, which were removed and why, and - for each one - the prediction
that should be believed rather than the average of all four models.

WHICH PREDICTION TO BELIEVE, AND WHY
    ALIGNN is the number quoted. It is the best moduli model measured
    (MAE 0.0539 / 0.0725 log10, R2 0.920 / 0.901) and, in arm B, the only one
    whose predictions fall inside the range of real measured moduli for that
    chemistry. round 9 is reported alongside as the disagreement, never as the
    answer, because step 64 showed it exceeds every real value ever measured
    for those chemistries.

RUN
    ml_env/bin/python scripts/cgcnn/66_dft_queue.py

OUTPUTS
    results/cgcnn/66_dft_queue.csv         every material, with status and reason
    results/cgcnn/66_dft_queue_slide.csv   the compact table for a slide
    results/cgcnn/66_dft_queue.png         predicted kappa with the model spread
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")

QUEUE = os.path.join(RESULTS, "61_dft_validation_set.csv")
VERDICT = os.path.join(RESULTS, "64_armB_verdict.csv")


def main():
    q = pd.read_csv(QUEUE)
    v = pd.read_csv(VERDICT)

    # The verdict table is keyed by formula and carries one boolean per row:
    # does round 9's modulus exceed every real measured modulus for that
    # chemistry? Where it does, arm B's disagreement is resolved against round 9
    # and no calculation is needed.
    refuted = set(v.loc[v.round9_exceeds_all_real, "formula"])

    def status(row):
        if row.arm.startswith("A"):
            return "QUEUE", "four models agree it is ultralow - the actual candidates"
        if row.arm.startswith("C"):
            return "CONTROL", "stiff by every model - guards against a pipeline that calls everything soft"
        if row.formula in refuted:
            return "DROPPED", ("round 9 refuted by step 64: its K exceeds every real "
                               "measured K for this chemistry")
        return "QUEUE", "arm B, disagreement not resolved"

    q[["status", "reason"]] = q.apply(lambda r: pd.Series(status(r)), axis=1)

    # The believable prediction, and how far the models spread around it. The
    # spread is reported as a RATIO because these are log-scale quantities: a
    # spread of 160x means the models disagree by more than two decades.
    q["kappa_alignn"] = q["ALIGNN"]
    q["model_spread_x"] = q["kappa_max"] / q["kappa_min"]

    print("=" * 78)
    print("STEP 66 - THE DFT QUEUE AS IT STANDS")
    print("=" * 78)
    for st in ("QUEUE", "CONTROL", "DROPPED"):
        sub = q[q.status == st]
        print(f"\n{st}: {len(sub)} materials")
        print("-" * 78)
        if st == "DROPPED":
            print(f"  {sub.reason.iloc[0]}")
        print(f"  {'formula':<12}{'atoms':>6}{'spacegroup':>12}"
              f"{'ALIGNN kappa':>14}{'spread':>9}")
        for r in sub.sort_values("kappa_alignn").itertuples():
            print(f"  {r.formula:<12}{getattr(r, '_5'):>6}{r.space_group:>12}"
                  f"{r.kappa_alignn:>14.3f}{r.model_spread_x:>8.1f}x")

    n_before, n_now = len(q), int((q.status == "QUEUE").sum())
    print(f"\n  queue was {n_before} materials; it is now {n_now} plus "
          f"{int((q.status=='CONTROL').sum())} controls.")
    print(f"  {int((q.status=='DROPPED').sum())} removed without spending a single "
          f"DFT calculation.")

    out = os.path.join(RESULTS, "66_dft_queue.csv")
    cols = ["status", "arm", "formula", "material_id", "Number of Atoms",
            "space_group", "crystal_system", "kappa_alignn", "CGCNN-ens",
            "round9", "tree", "kappa_min", "kappa_max", "model_spread_x",
            "K_alignn", "G_alignn", "n_models_low", "reason"]
    q[cols].sort_values(["status", "kappa_alignn"]).to_csv(out, index=False)
    print(f"\n-> {out}")

    # The slide version: only what fits on a projected line, only the live queue.
    slide = (q[q.status == "QUEUE"]
             .sort_values("kappa_alignn")
             .rename(columns={"Number of Atoms": "atoms",
                              "kappa_alignn": "kappa_pred_ALIGNN"})
             [["formula", "material_id", "atoms", "space_group", "crystal_system",
               "K_alignn", "G_alignn", "kappa_pred_ALIGNN", "model_spread_x"]]
             .round(3))
    slide_path = os.path.join(RESULTS, "66_dft_queue_slide.csv")
    slide.to_csv(slide_path, index=False)
    print(f"-> {slide_path}")
    print("\nTHE SLIDE TABLE")
    print("-" * 78)
    print(slide.to_string(index=False))

    figure(q, os.path.join(RESULTS, "66_dft_queue.png"))
    return 0


def figure(q, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colour = {"QUEUE": "#2563eb", "CONTROL": "#6E6B64", "DROPPED": "#dc2626"}

    d = q.sort_values(["status", "kappa_alignn"])
    y = np.arange(len(d))
    for i, r in enumerate(d.itertuples()):
        # the horizontal bar is the full spread across the four models; the dot
        # is the number to believe
        ax.plot([r.kappa_min, r.kappa_max], [i, i], color="#D8D5CE", lw=3,
                solid_capstyle="round", zorder=1)
        ax.scatter(r.kappa_alignn, i, s=42, color=colour[r.status], zorder=3)
        if r.status == "DROPPED":
            ax.scatter(r.round9, i, s=42, facecolors="none",
                       edgecolors="#dc2626", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.formula}" + ("  (dropped)" if r.status == "DROPPED"
                                          else "  (control)" if r.status == "CONTROL"
                                          else "")
                        for r in d.itertuples()], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"predicted $\kappa_{\mathrm{lat}}$, W m$^{-1}$K$^{-1}$  (log scale)")
    ax.set_title("The DFT queue: filled dot = ALIGNN (the number to believe),\n"
                 "grey bar = spread across four models, open circle = round 9 "
                 "(refuted by step 64)", fontsize=10)
    ax.axvline(1.0, color="#1BAF7A", ls="--", lw=1)
    ax.text(1.05, len(d) - 1.5, "1 W/m/K", color="#1BAF7A", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"-> {path}")


if __name__ == "__main__":
    sys.exit(main())
