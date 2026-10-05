#!/usr/bin/env python
"""
Step 80 - how reproducible is gamma? Two accidents gave us the answer.
================================================================================

    ml_env/bin/python scripts/cgcnn/80_recut_reproducibility.py

WHY THIS EXISTS
---------------
A number is only as good as its reproducibility, and until now nothing in this
program had measured that for gamma. Two accidents during the halide recut
supplied the measurement for free:

  1. A DUPLICATE RUN. Shard s01 was pushed to both kaggle1 and kaggle3 - the
     driver's "already done" guard checks for a local CSV, and the CSV had not
     been downloaded yet, so it ran twice. 69 crystals, same code, same CIFs,
     two different machines. The spread between the two copies is pure
     run-to-run noise: MACE forces on a GPU are not bit-identical.

  2. A RE-RUN AT A DIFFERENT CUTOFF. The recut also recomputed gamma_c001,
     which reproduces the ORIGINAL definition exactly. So for all 138 halides
     we have the same quantity computed twice, months apart, on different
     machines.

THE RESULT, STATED UP FRONT
----------------------------
The buggy quantity is roughly an order of magnitude less reproducible than the
fixed one. That is a second, independent argument for the cutoff - separate
from the bias argument - because it does not depend on knowing the right
answer. A quantity that will not reproduce itself cannot be ranked on.

OUTPUT
    results/cgcnn/80_gamma_reproducibility.csv
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")
G = os.path.expanduser("~/Desktop/data_generation/results")
LADDER = ["gamma_c001", "gamma_c100", "gamma_c300", "gamma_c500", "gamma_c1000"]


def main():
    # ---- 1. the duplicate run: same shard, two accounts -------------------
    k1 = pd.read_csv(os.path.join(G, "01_gamma_phonon_halide_recut_s01.csv"),
                     dtype={"mp_id": str})
    k3 = pd.read_csv(os.path.join(G, "01_gamma_phonon_halide_recut_s01_kaggle3.csv"),
                     dtype={"mp_id": str})
    m = k1.merge(k3, on="mp_id", suffixes=("_k1", "_k3"))
    print("=" * 88)
    print(f"DUPLICATE RUN - shard s01 on two accounts, {len(m)} crystals")
    print("=" * 88)
    rows = []
    for c in LADDER:
        # .abs() then .max() - the worst single crystal, not the average, is
        # what decides whether a ranking built on this column is trustworthy.
        d = (m[c + "_k1"] - m[c + "_k3"]).abs()
        rows.append({"test": "duplicate run (69 crystals)", "column": c,
                     "max_abs_diff": d.max(), "mean_abs_diff": d.mean(),
                     "n": len(m)})
        print(f"   {c:12s}  max|diff| {d.max():.4f}   mean {d.mean():.2e}")
    print(f"   identical stability verdict: "
          f"{int((m.dynamically_stable_k1 == m.dynamically_stable_k3).sum())}/{len(m)}")
    print(f"   GPU time paid twice: {m.runtime_s_k1.sum()/3600:.2f} h")

    # ---- 2. the re-run: original definition, computed twice ---------------
    rc = pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in
                    sorted(glob.glob(os.path.join(
                        G, "01_gamma_phonon_halide_recut_s0?.csv")))],
                   ignore_index=True).drop_duplicates("mp_id")
    old = pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in
                     sorted(glob.glob(os.path.join(G, "01_gamma_phonon_halide_s0*.csv")))
                     + [os.path.join(G, "01_gamma_phonon_pilot_mb3.csv")]],
                    ignore_index=True).drop_duplicates("mp_id")
    r = rc.merge(old[["mp_id", "gamma_mlip", "gamma_acoustic", "min_freq_THz"]],
                 on="mp_id", suffixes=("", "_bank"))
    d = (r.gamma_c001 - r.gamma_mlip_bank).abs()
    print()
    print("=" * 88)
    print(f"RE-RUN - the ORIGINAL gamma definition computed twice, {len(r)} crystals")
    print("=" * 88)
    print(f"   gamma_c001 vs banked gamma_mlip : max|diff| {d.max():.4f}"
          f"   mean {d.mean():.4f}   median {d.median():.2e}")
    print(f"   agree within 1%: {int((d <= 0.01*r.gamma_mlip_bank.abs()).sum())}/{len(r)}")
    rows.append({"test": "re-run months apart (138 crystals)", "column": "gamma_c001",
                 "max_abs_diff": d.max(), "mean_abs_diff": d.mean(), "n": len(r)})

    worst = r.loc[d.idxmax()]
    print(f"\n   WORST CASE: {worst.formula}  ({worst.mp_id})")
    print(f"      banked gamma     {worst.gamma_mlip_bank:7.4f}"
          f"      recut gamma_c001 {worst.gamma_c001:7.4f}")
    print(f"      banked acoustic  {worst.gamma_acoustic_bank:7.4f}"
          f"      recut acoustic   {worst.gamma_acoustic:7.4f}   <- the 1/omega region")
    print(f"      banked min freq  {worst.min_freq_THz_bank:7.4f} THz"
          f"   recut min freq   {worst.min_freq_THz:7.4f} THz")
    print(f"      at the FIXED cutoff the recut gives {worst.gamma_c300:.4f}")
    print("\n   Same code, same CIF, same supercell. The acoustic branch landed")
    print("   differently by a hair in frequency and that hair, divided by omega,")
    print("   became a factor of 11 in gamma_acoustic and 2.4x in gamma itself.")
    print("\n   READ THE MEDIAN ABOVE BEFORE GENERALISING: it is ~1e-12, so 136 of")
    print("   138 re-ran bit-identically. This is not ambient noise - it is a")
    print("   knife-edge. Most crystals do not sit on the edge; the ones whose")
    print("   lowest acoustic mode sits within a hair of zero do, and nothing in")
    print("   the output flags which those are. That is the case for the cutoff:")
    print("   it removes the knife-edge rather than hoping to miss it.")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "80_gamma_reproducibility.csv"), index=False)

    print()
    print("=" * 88)
    print("THE COMPARISON THAT MATTERS")
    print("=" * 88)
    dup = out[out.test.str.startswith("duplicate")].set_index("column")
    print(f"   buggy cutoff (c001) reproduces to : {dup.loc['gamma_c001','max_abs_diff']:.4f}")
    print(f"   fixed cutoff (c300) reproduces to : {dup.loc['gamma_c300','max_abs_diff']:.4f}")
    print(f"   ratio                             : "
          f"{dup.loc['gamma_c001','max_abs_diff']/dup.loc['gamma_c300','max_abs_diff']:.1f}x worse")
    print(f"\nwrote {RESULTS}/80_gamma_reproducibility.csv")


if __name__ == "__main__":
    main()
