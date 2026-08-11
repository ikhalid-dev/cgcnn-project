#!/usr/bin/env python3
"""
Predict K, G and kappa_L for your own CIF files - CGCNN + ALIGNN, cross-checked.
====================================================================================

    python3 predict.py my_cifs/
    python3 predict.py my_cifs/single_material.cif
    python3 predict.py my_cifs/ --out my_report.csv

This is the tool version of PREDICT_NEW_CIFS.txt's manual workflow: instead of
running scripts/cgcnn/04_predict_moduli.py, 07_predict_kappa.py,
scripts/alignn/15_alignn_predict_moduli.py and 16_alignn_predict_kappa.py by
hand and comparing 2-4 CSVs yourself, this runs all of it through
predict_engine.py in one call and prints a per-crystal verdict directly -
including a confidence tier (see predict_engine.confidence_tier's own
docstring for exactly what it checks). Nothing here is a new model or a new
physics formula; see predict_engine.py's module docstring for what's actually
new versus reused.

For the full explanation of how to read the numbers below (what "confidence"
means, what accuracy to actually expect, what this pipeline has never seen),
read PREDICT_NEW_CIFS.txt - this tool automates that guide's workflow, it
doesn't replace reading it once.
"""

import argparse
import os
import shutil
import sys
import tempfile

# torch first - see predict_engine.py's own note; it matters here too since
# this file's imports happen before predict_engine's do.
import torch  # noqa: F401

import pandas as pd

import predict_engine as engine

CONF_LABEL = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW",
             "unverified": "UNVERIFIED"}


def resolve_cif_dir(path):
    """Accept either a directory of .cif files or a single .cif file.

    predict_engine.predict_batch() (like every script it wraps) expects a
    directory - a lone file gets copied into a throwaway temp directory so
    the single-file case doesn't need its own code path through the engine.
    """
    if os.path.isdir(path):
        return path, None
    if os.path.isfile(path) and path.lower().endswith(".cif"):
        tmp_dir = tempfile.mkdtemp(prefix="predict_single_")
        shutil.copy2(path, os.path.join(tmp_dir, os.path.basename(path)))
        return tmp_dir, tmp_dir
    sys.exit(f"{path} is neither a directory nor a .cif file")


def print_report(df):
    for row in df.itertuples():
        header = f"{row.material_id} ({row.formula})"
        if row.provenance != "unseen":
            header += f"  [{row.provenance} split - DFT reference available]"
        else:
            header += "  [unseen - no DFT reference]"
        print("=" * 78)
        print(header)
        print("-" * 78)
        print(f"  Bulk modulus K   CGCNN {row.K_VRH_pred:8.2f} GPa   "
             f"ALIGNN {row.K_VRH_alignn:8.2f} GPa" +
             (f"   DFT {row.K_VRH_dft:8.2f} GPa" if row.provenance != "unseen"
              and not pd.isna(row.K_VRH_dft) else ""))
        print(f"  Shear modulus G  CGCNN {row.G_VRH_pred:8.2f} GPa   "
             f"ALIGNN {row.G_VRH_alignn:8.2f} GPa" +
             (f"   DFT {row.G_VRH_dft:8.2f} GPa" if row.provenance != "unseen"
              and not pd.isna(row.G_VRH_dft) else ""))
        print(f"  kappa_L          CGCNN {row.Kappa_cal_cgcnn:8.2f} W/m/K "
             f"[{row.Kappa_cal_cgcnn_p05:.1f}-{row.Kappa_cal_cgcnn_p95:.1f} "
             f"90% interval]   ALIGNN {row.Kappa_cal_alignn:8.2f} W/m/K")
        tier = CONF_LABEL[row.confidence]
        if row.confidence == "unverified":
            print(f"  Confidence: {tier}  (only one model produced a usable "
                 f"prediction for this crystal)")
        else:
            print(f"  Confidence: {tier}  (CGCNN vs ALIGNN kappa_L disagree "
                 f"by {row.model_disagreement_pct:.1f}%)")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cif_path", help="a directory of .cif files, or a single .cif file")
    parser.add_argument("--out", default=os.path.join(
        engine.PROJECT_ROOT, "predictions", "report.csv"))
    parser.add_argument("--device", default="auto",
                        help="ALIGNN device: auto / cpu / cuda (CGCNN always runs on CPU)")
    args = parser.parse_args()

    cif_dir, cleanup_dir = resolve_cif_dir(args.cif_path)
    try:
        print("Loading CGCNN ensemble + ALIGNN models...")
        models = engine.load_models(device=args.device)

        df, failures = engine.predict_batch(cif_dir, models)
        for label, failed in failures.items():
            if failed:
                print(f"\n{len(failed)} crystal(s) failed {label.upper()} "
                     f"featurisation: {failed[:3]}")

        print()
        print_report(df)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.round(4).to_csv(args.out, index=False)
        print(f"\nWrote {len(df)} rows to {args.out}")
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
