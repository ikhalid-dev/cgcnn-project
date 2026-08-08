#!/usr/bin/env python3
"""
STEP 11 - Convert the matbench benchmark into ALIGNN's expected format.
================================================================================

    python scripts/11_prepare_alignn_data.py

WHY THIS SCRIPT EXISTS
-----------------------
Phase 2 asks whether a different, more expressive architecture (ALIGNN - it
adds a line graph of bond angles on top of CGCNN's bond-graph, which is
exactly the geometric information CGCNN's convolution cannot see) does better
than our CGCNN on the SAME data. "Same data" has to mean the same 10,987
matbench crystals AND the same train/val/test split - not just the same
benchmark - or a difference in scores could just as easily be a difference in
which crystals ended up in the test set.

So this script reuses, unchanged, exactly the two functions that already
define "the data" for this project: `load_benchmark()` from
01b_prepare_full_dataset.py (the matbench structures + K_VRH/G_VRH, in the
row order matminer returns them) and `split_indices()` from 02_train.py (the
seeded shuffle-and-slice that produced the CGCNN ensemble's train/val/test
sets). Both are imported via importlib, the same workaround
05_ensemble.py already uses for scripts whose filenames start with a digit.

WHY A CUSTOM SPLIT INSTEAD OF ALIGNN'S OWN split_seed
-------------------------------------------------------
ALIGNN's own get_id_train_val_test (alignn/data.py) shuffles with Python's
stdlib `random.seed(split_seed)`, not numpy's RandomState - a different RNG
algorithm entirely. Passing --split-seed 42 to both models would NOT produce
the same partition; it would just be two different seeds that happen to have
the same numeral. So the split is computed once, here, with our own
split_indices(), and the crystals are physically reordered into
train-then-val-then-test order. ALIGNN is then told (via keep_data_order=True
in scripts/12_train_alignn.py) not to shuffle again - "the order you were
given IS the split," which sidesteps ALIGNN's internal RNG entirely rather
than trying to make two different algorithms agree.

WHY THE TARGET KEYS ARE bulk_modulus_kv / shear_modulus_gv, NOT K_VRH/G_VRH
------------------------------------------------------------------------------
alignn.config.TrainingConfig.target is a pydantic Literal restricted to a
fixed list of known JARVIS-DFT property names (checked directly - passing
"K_VRH" raises a ValidationError listing the allowed values). This is purely
a label ALIGNN uses to look up the right dict key in each dataset_array
entry - it never touches JARVIS-DFT's own data or labels. Reusing
"bulk_modulus_kv"/"shear_modulus_gv" as dict key names is a labelling
convenience to satisfy that validator; the values behind those keys are
still our own matbench K_VRH/G_VRH, unchanged.

OUTPUT
-------
data_full/alignn_data.pkl: {"dataset_array": [...], "n_train", "n_val",
"n_test"}. dataset_array is a list of dicts, ordered train-then-val-then-test,
each {"jid": "mb-00000", "atoms": <jarvis Atoms.to_dict()>,
"bulk_modulus_kv": float, "shear_modulus_gv": float}. Not committed (large,
like graphs.pt) - rebuilds deterministically from matbench in a few minutes.
"""

import os
import sys
import time
import warnings
from importlib import import_module

# torch first - see cgcnn_scratch/data.py for why (MKL/libiomp5 duplicate
# OpenMP runtime segfault if numpy/pandas/pymatgen import first in this env).
import torch  # noqa: F401

import numpy as np
import pandas as pd
import pickle

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Both have a leading digit, so import_module (not `import`) is required -
# same workaround scripts/05_ensemble.py already uses for 03_evaluate.
_prep = import_module("01b_prepare_full_dataset")
_train = import_module("02_train")


def convert_to_jarvis(benchmark):
    """pymatgen Structure -> JARVIS Atoms -> plain dict, one entry per crystal.

    Mirrors 01b's build_graphs(): same per-row try/except (a handful of
    matbench structures may have elements or disorder JARVIS's converter
    rejects, just as some fail CGCNN's featuriser), same mb-{i:05d} id
    convention, and progress printed the same way.
    """
    from jarvis.core.atoms import pmg_to_atoms

    entries, failed = [], []
    start = time.time()
    for i, record in enumerate(benchmark.itertuples(index=False)):
        mb_id = f"mb-{i:05d}"
        try:
            atoms = pmg_to_atoms(record.structure)
            entries.append({
                "jid": mb_id,
                "atoms": atoms.to_dict(),
                "bulk_modulus_kv": float(record.K_VRH),
                "shear_modulus_gv": float(record.G_VRH),
            })
        except Exception as exc:
            failed.append((mb_id, str(exc)[:70]))
            entries.append(None)  # keep a placeholder so indices still line up

        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - start)
            print(f"  {i + 1:5d}/{len(benchmark)} converted "
                 f"({rate:.0f}/s, ~{(len(benchmark) - i - 1) / rate / 60:.0f} min left)")

    print(f"Converted {len(entries) - len(failed)}/{len(benchmark)} structures "
         f"in {(time.time() - start) / 60:.1f} min")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")
    return entries


def main():
    out_path = os.path.join(PROJECT_ROOT, "data_full", "alignn_data.pkl")

    print("=== Preparing matbench data in ALIGNN's format ===\n")
    benchmark = _prep.load_benchmark()

    print("\nConverting every structure to a JARVIS Atoms dict...")
    entries = convert_to_jarvis(benchmark)

    # Same split every CGCNN ensemble member shares (run_pipeline.sh:
    # --train-ratio 0.7 --val-ratio 0.15 --split-seed 42), computed with the
    # exact function that produced it - not a re-derivation that could drift.
    train_idx, val_idx, test_idx = _train.split_indices(
        len(entries), train_ratio=0.7, val_ratio=0.15, seed=42)
    print(f"\nSplit (matching the CGCNN ensemble exactly): "
         f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    # Drop crystals JARVIS's converter rejected, keeping split membership -
    # a placeholder (None) at index i means i is missing from whichever of
    # train/val/test it belonged to; filtering after picking indices, not
    # before, keeps every remaining crystal's split assignment correct.
    def pick(idx_list):
        return [entries[i] for i in idx_list if entries[i] is not None]

    ordered = pick(train_idx) + pick(val_idx) + pick(test_idx)
    n_train, n_val, n_test = len(pick(train_idx)), len(pick(val_idx)), len(pick(test_idx))
    n_dropped = len(entries) - (n_train + n_val + n_test)
    if n_dropped:
        print(f"  ({n_dropped} crystals dropped from the split - failed JARVIS "
             f"conversion, see list above)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"dataset_array": ordered, "n_train": n_train,
                    "n_val": n_val, "n_test": n_test}, fh)
    print(f"\nWrote {out_path} ({n_train} train / {n_val} val / {n_test} test, "
         f"{len(ordered)} total)")


if __name__ == "__main__":
    main()
