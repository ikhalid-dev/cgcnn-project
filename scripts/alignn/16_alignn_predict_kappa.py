#!/usr/bin/env python3
"""
STEP 16 - Stage 2 for ALIGNN: turn its predicted moduli into kappa_L.
================================================================================

    python scripts/16_alignn_predict_kappa.py

WHERE THIS FITS
----------------
scripts/15_alignn_predict_moduli.py produced
results/alignn_moduli_predictions.csv - bulk and shear modulus for the same
1,213 PINK crystals CGCNN was scored on, but from ALIGNN. This script is the
second arrow, run with ALIGNN's moduli instead of CGCNN's:

    CIF -> ALIGNN -> (K, G) -> Slack model -> kappa_L

WHY THE PHYSICS ISN'T RE-DERIVED A THIRD TIME
-------------------------------------------------
`slack_physics()` already lives in 07_predict_kappa.py, imported here via
importlib (same digit-prefixed-module workaround 05/11/12 already use) rather
than copy-pasted - there is exactly one implementation of the Slack-model
formulas in this project, used by three different callers (the paper's own
pretrained model via pink_predict.py, our CGCNN ensemble via this same
function, and now ALIGNN).

WHY THERE IS NO MONTE CARLO STEP HERE
------------------------------------------
07_predict_kappa.py's uncertainty propagation draws from the CGCNN ENSEMBLE's
per-crystal K/G spread across 3 independently-seeded models - the whole point
being that disagreement between ensemble members is a free confidence signal.
ALIGNN here is one trained model, not an ensemble (see docs/method.tex
Section 9's own note: "a single ALIGNN model against a 3-model CGCNN
ensemble"), so there is no spread to propagate. Reporting a Monte Carlo
interval anyway - e.g. by inventing a fixed spread - would fabricate
uncertainty this run does not actually have. What's written out is a point
estimate only, honestly labelled as one.

WHAT THIS ADDS: A THIRD-WAY COMPARISON
-------------------------------------------
scripts/08_compare_kappa.py already checks our CGCNN-derived kappa_L against
the paper's own pretrained pipeline. This script adds the missing leg of that
triangle: does ALIGNN's kappa_L agree with our OWN CGCNN-derived kappa_L, on
the same crystals, through the identical physics? Reuses
08_compare_kappa.py's own summarise()/screening_overlap() rather than
recomputing the same statistics a third way.

OUTPUT
------
    results/alignn_kappa_predictions.csv
        material_id, formula, provenance, structural quantities,
        Poisson ratio, Gruneisen parameter,
        Kappa_Slack (W/m/K), Kappa_cal (W/m/K)   <- point estimates only
    Sorted ascending by Kappa_cal, same convention as 07's output.
"""

import argparse
import os
import sys
import warnings
from importlib import import_module

# torch first - see 07_predict_kappa.py's own note on MKL/libiomp5 ordering.
import torch  # noqa: F401

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# scripts/cgcnn/, not this file's own scripts/alignn/ - see 11's identical note.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))

_kappa = import_module("07_predict_kappa")
_compare = import_module("08_compare_kappa")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions",
                        default=os.path.join(PROJECT_ROOT, "results", "alignn",
                                             "alignn_moduli_predictions.csv"))
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"))
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "alignn",
                                                      "alignn_kappa_predictions.csv"))
    parser.add_argument("--compare-against",
                        default=os.path.join(PROJECT_ROOT, "results", "cgcnn",
                                             "pink_kappa_predictions.csv"),
                        help="our own CGCNN-derived kappa_L, for the "
                             "model-vs-model comparison at the end")
    args = parser.parse_args()

    print("=== Stage 2 for ALIGNN: predicted moduli -> kappa_L (Slack model) ===\n")
    moduli = pd.read_csv(args.predictions)
    print(f"Loaded {len(moduli)} crystals from {args.predictions}")

    print("Computing structural quantities from CIFs "
         "(primitive cell, matching pink_predict.py)...")
    structure = _kappa.structure_quantities(args.cif_dir, moduli.material_id)
    df = moduli.merge(structure, on="material_id", how="inner")
    dropped = len(moduli) - len(df)
    if dropped:
        print(f"  {dropped} crystals dropped: CIF failed to parse")

    # Point estimate only - see the module docstring for why no Monte Carlo.
    point = _kappa.slack_physics(
        df["K_VRH_pred"].values, df["G_VRH_pred"].values,
        df["Volume (A3)"].values, df["Density (g cm-3)"].values,
        df["Atomic mass (amu)"].values, df["Number of Atoms"].values)
    df["Poisson ratio"] = point["poisson"]
    df["Gruneisen parameter"] = point["gruneisen"]
    df["Kappa_Slack (W m-1 K-1)"] = point["kappa_slack"]
    df["Kappa_cal (W m-1 K-1)"] = point["kappa_cal"]

    df = df.sort_values("Kappa_cal (W m-1 K-1)").reset_index(drop=True)

    keep = ["material_id", "formula", "provenance", "Number of Atoms",
           "Volume (A3)", "Density (g cm-3)", "Atomic mass (amu)",
           "K_VRH_pred", "G_VRH_pred", "Poisson ratio", "Gruneisen parameter",
           "Kappa_Slack (W m-1 K-1)", "Kappa_cal (W m-1 K-1)"]
    df[keep].round(5).to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} rows to {args.out}")
    print("\nLowest predicted kappa_L (ALIGNN):")
    print(df.head(10)[["material_id", "formula", "K_VRH_pred", "G_VRH_pred",
                       "Gruneisen parameter", "Kappa_cal (W m-1 K-1)"]]
          .round(3).to_string(index=False))

    # --- Model-vs-model: does ALIGNN's kappa_L agree with our own CGCNN's? -
    if not os.path.exists(args.compare_against):
        print(f"\n(No {args.compare_against} to compare against - run "
             f"07_predict_kappa.py first for the full three-way picture.)")
        return

    print(f"\n=== Comparing against our own CGCNN-derived kappa_L "
         f"({args.compare_against}) ===")
    reference = pd.read_csv(args.compare_against)
    merged = df.rename(columns={"Kappa_cal (W m-1 K-1)": "Kappa_cal (W m-1 K-1)_ours"}) \
               .merge(reference[["material_id", "Kappa_cal (W m-1 K-1)"]]
                     .rename(columns={"Kappa_cal (W m-1 K-1)": "Kappa_cal (W m-1 K-1)_ref"}),
                     on="material_id", how="inner")
    print(f"Compared on {len(merged)} crystals present in both runs")

    summary = _compare.summarise(merged)
    print(f"\nPearson r (log10 space)  : {summary['pearson_r_log']:.4f}")
    print(f"Spearman rank correlation: {summary['spearman_r']:.4f}")
    print(f"MAE (log10 W/m/K)        : {summary['mae_log10']:.4f}  "
         f"({summary['rel_error_pct']:.1f}% typical multiplicative error)")

    overlap = _compare.screening_overlap(merged, top_n=20)
    print(f"\nOverlap in the 20 lowest-kappa_L candidates: {overlap}/20 "
         f"nominated by both ALIGNN and our CGCNN ensemble")


if __name__ == "__main__":
    main()
