#!/usr/bin/env python3
"""
STEP 1 - Build a labelled training set from the raw CIF files.
==============================================================

THE PROBLEM
-----------
`complete-data/` contains 1,213 CIF files and nothing else. A CIF describes
WHERE the atoms are, but it says nothing about the bulk or shear modulus. To
train a supervised model we need (structure, true modulus) PAIRS - and the
true moduli have to come from somewhere else.

THE SOLUTION
------------
Matbench publishes two benchmark datasets of DFT-computed elastic moduli:

    matbench_log_kvrh   10,987 structures + log10(bulk modulus  K_VRH)
    matbench_log_gvrh   10,987 structures + log10(shear modulus G_VRH)

These are exactly the datasets the PINK paper trained on. Both contain the
same materials in the same row order, so row i of one lines up with row i of
the other. Crucially, though, matbench strips the Materials Project IDs - each
entry is just a pymatgen Structure. So we cannot join on `mp-xxxxx`; we have to
decide whether two STRUCTURES are the same crystal.

HOW THE MATCHING WORKS
----------------------
Comparing all 1,213 x 10,987 = 13 million pairs with a full symmetry-aware
comparison would take hours. So we do it in two passes:

    Pass 1 (cheap)  bucket every matbench entry by reduced formula. A crystal
                    can only possibly match another with the same formula, so
                    this instantly narrows 10,987 candidates down to a handful.

    Pass 2 (exact)  within a bucket, use pymatgen's StructureMatcher, which
                    compares crystals up to lattice choice, origin shift and
                    symmetry operations - so the same material written in two
                    different but equivalent settings still matches.

WHAT COMES OUT
--------------
Roughly 278 of the 1,213 CIFs find a match. That is not a bug: matbench's
elastic set covers ~11k of the ~150k materials in the Materials Project, so
most of our CIFs simply have never had their elastic tensor computed. 278
labelled crystals is a small but genuinely usable training set.

Outputs written to `data/`:
    data/labels.csv     material_id, formula, K_VRH, G_VRH  (in GPa)
    data/cifs/          the matched CIF files, copied here
"""

import argparse
import collections
import glob
import os
import shutil
import sys
import warnings

import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

# CIF parsing is noisy about rounded coordinates and partial occupancies.
# None of it is actionable here, so keep the log readable.
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_benchmark():
    """Load both matbench elastic datasets and pair them up row by row.

    Returns a list of (Structure, K_VRH, G_VRH) with moduli in GPa.
    Matbench stores log10 of the moduli, so we undo that here - it is easier to
    reason about "203 GPa" than "2.307" when eyeballing results. The training
    script takes the log again, because a log target trains far better (see the
    note in 02_train.py).
    """
    print("Loading matbench elastic datasets (downloads ~4 MB on first run)...")
    kvrh = load_dataset("matbench_log_kvrh")
    gvrh = load_dataset("matbench_log_gvrh")
    assert len(kvrh) == len(gvrh), "the two datasets should be row-aligned"
    print(f"  loaded {len(kvrh)} benchmark entries")

    return [
        (kvrh.structure[i], 10 ** kvrh["log10(K_VRH)"][i], 10 ** gvrh["log10(G_VRH)"][i])
        for i in range(len(kvrh))
    ]


def load_our_cifs(cif_dir):
    """Parse every CIF in cif_dir into a pymatgen Structure, keyed by mp-id."""
    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not paths:
        sys.exit(f"No CIF files found in {cif_dir}")

    structures, failed = {}, []
    for path in paths:
        material_id = os.path.splitext(os.path.basename(path))[0]
        try:
            structures[material_id] = Structure.from_file(path)
        except Exception as exc:  # a handful of CIFs are malformed
            failed.append((material_id, str(exc)[:60]))

    print(f"Parsed {len(structures)} CIFs from {cif_dir}")
    if failed:
        print(f"  {len(failed)} failed to parse, e.g. {failed[:3]}")
    return structures


def match(our_structures, benchmark):
    """Find which of our structures appear in the benchmark set.

    See the module docstring for why this is two passes.
    """
    # --- Pass 1: bucket the benchmark by reduced formula -------------------
    buckets = collections.defaultdict(list)
    for idx, (structure, _, _) in enumerate(benchmark):
        buckets[structure.composition.reduced_formula].append(idx)
    print(f"Benchmark covers {len(buckets)} distinct formulas")

    candidates = {mid: s for mid, s in our_structures.items()
                  if s.composition.reduced_formula in buckets}
    print(f"Formula-level candidates: {len(candidates)} / {len(our_structures)}")

    # --- Pass 2: exact structural comparison within each bucket -----------
    # primitive_cell=True reduces both crystals to their primitive cell first,
    # so a 2x2x2 supercell still matches the single cell it was built from.
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=False)

    rows = []
    for material_id, structure in candidates.items():
        formula = structure.composition.reduced_formula
        for idx in buckets[formula]:
            bench_structure, k_vrh, g_vrh = benchmark[idx]
            if matcher.fit(structure, bench_structure):
                rows.append({
                    "material_id": material_id,
                    "formula": formula,
                    "K_VRH": k_vrh,
                    "G_VRH": g_vrh,
                })
                break  # first match wins; duplicates in matbench are rare

    print(f"Confirmed structural matches: {len(rows)}")
    return pd.DataFrame(rows).sort_values("material_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data"))
    args = parser.parse_args()

    benchmark = load_benchmark()
    our_structures = load_our_cifs(args.cif_dir)
    labels = match(our_structures, benchmark)

    if labels.empty:
        sys.exit("No matches found - nothing to train on.")

    # --- Write outputs ----------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    labels_path = os.path.join(args.out_dir, "labels.csv")
    labels.to_csv(labels_path, index=False)

    # Copy the matched CIFs into their own folder. Keeping the training set
    # physically separate means the training script never has to think about
    # the 900-odd unlabelled structures sitting next to them.
    cif_out = os.path.join(args.out_dir, "cifs")
    os.makedirs(cif_out, exist_ok=True)
    for material_id in labels.material_id:
        shutil.copy(os.path.join(args.cif_dir, f"{material_id}.cif"), cif_out)

    # CIFData expects atom_init.json to sit in the SAME directory as the CIFs,
    # so place a copy there rather than making the loader hunt for it.
    shutil.copy(os.path.join(args.out_dir, "atom_init.json"), cif_out)

    print(f"\nWrote {len(labels)} labelled structures")
    print(f"  {labels_path}")
    print(f"  {cif_out}/  ({len(labels)} CIFs)")
    print("\nTarget distribution (GPa):")
    print(labels[["K_VRH", "G_VRH"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
