#!/usr/bin/env python3
"""
STEP 1b - Build the FULL training set from matbench, not just the 278 overlap.
=============================================================================

WHY THIS SCRIPT EXISTS
----------------------
`01_prepare_dataset.py` builds a training set by intersecting our 1,213 local
CIFs with matbench's elastic benchmark. Only 278 survive that intersection,
because matbench covers ~11k of the Materials Project's ~150k materials and
most of our CIFs simply have no computed elastic tensor. 278 crystals is not
enough: the model trained on them reaches test MAE ~0.15 log10(GPa) against the
paper's ~0.07, and it overfits hard (train MAE ~0.08).

The fix is to stop treating our 1,213 CIFs as the universe. The labels live in
matbench, so TRAIN ON ALL OF MATBENCH:

    matbench_log_kvrh   10,987 structures + log10(bulk modulus  K_VRH)
    matbench_log_gvrh   10,987 structures + log10(shear modulus G_VRH)

That is a 40x larger training set, and it is exactly the data the PINK paper
used. Our 1,213 CIFs then become what they should have been all along: the
PREDICTION set feeding the kappa_L stage, not the training set.

WHY WE CACHE GRAPHS INSTEAD OF WRITING 11,000 CIFs
--------------------------------------------------
The slow step in this pipeline is not the network, it is turning a crystal into
a graph - a periodic neighbour search for every atom. Round-tripping matbench's
Structure objects through .cif files on disk would mean paying that cost once to
write them, and again on every training run to read them back.

So we convert Structure -> graph directly, once, and pickle the tensors. Both
training runs (bulk and shear) then load the same cache in seconds. The
conversion goes through `cgcnn_scratch.data.structure_to_graph`, the same
function CIFData uses, so cached graphs are bit-for-bit what training on CIFs
would have produced.

PROVENANCE - WHICH PINK CRYSTALS ARE ALSO TRAINING DATA
-------------------------------------------------------
Some of our 1,213 CIFs are in matbench, so they will end up in the training set.
When we later predict moduli for all 1,213, we need to know which predictions
are honest extrapolation and which are the model recalling something it was
fitted on. So this script also re-runs the structure matching from step 1 and
records the mp-id <-> matbench-index mapping in `mp_to_mb.csv`.

Outputs, all under `data_full/`:
    labels.csv      mb_id, formula, n_sites, K_VRH, G_VRH   (moduli in GPa)
    graphs.pt       {"ids": [...], "graphs": [(atom_fea, nbr_fea, nbr_idx), ...]}
    mp_to_mb.csv    mp_id, mb_id, formula  for the CIFs that are in matbench

Run:
    python scripts/01b_prepare_full_dataset.py
"""

import argparse
import collections
import glob
import os
import sys
import time
import warnings

# torch MUST be imported before numpy/pymatgen in this conda env - MKL loads its
# own OpenMP runtime first and the duplicate libiomp5 segfaults torch. Do not
# "tidy" these imports into alphabetical order.
import torch

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

# CIF parsing is noisy about rounded coordinates and partial occupancies.
# None of it is actionable here, so keep the log readable.
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cgcnn_scratch.data import (AtomFeaturiser, GaussianDistance,  # noqa: E402
                                structure_to_graph)


def load_benchmark():
    """Load both matbench elastic datasets and pair them up row by row.

    Returns a DataFrame with a `structure` column plus moduli in GPa. Matbench
    stores log10 of the moduli; we undo that here so labels.csv is readable in
    physical units, and the training script takes the log again.

    The two datasets are row-aligned (same materials, same order), which we
    assert rather than assume - a silent misalignment would pair every crystal
    with another crystal's shear modulus and still train perfectly happily.
    """
    print("Loading matbench elastic datasets (downloads ~100 MB on first run)...")
    kvrh = load_dataset("matbench_log_kvrh")
    gvrh = load_dataset("matbench_log_gvrh")
    assert len(kvrh) == len(gvrh), "the two datasets should be row-aligned"
    print(f"  loaded {len(kvrh)} benchmark entries")

    # Spot-check the alignment on a sample rather than comparing all 11k
    # structures, which would itself take minutes.
    for i in (0, len(kvrh) // 2, len(kvrh) - 1):
        assert (kvrh.structure[i].composition.reduced_formula ==
                gvrh.structure[i].composition.reduced_formula), \
            f"row {i} disagrees between the two datasets - not row-aligned"

    return pd.DataFrame({
        "structure": kvrh.structure,
        "K_VRH": 10 ** kvrh["log10(K_VRH)"],
        "G_VRH": 10 ** gvrh["log10(G_VRH)"],
    })


def build_graphs(benchmark, atom_init_path, max_num_nbr, radius, step):
    """Convert every benchmark Structure into a cached crystal graph.

    This is the expensive part - roughly a periodic neighbour search per
    crystal. Returns (ids, graphs, rows) where `rows` is the labels table,
    restricted to the crystals that actually converted.
    """
    ari = AtomFeaturiser(atom_init_path)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=step)

    ids, graphs, rows, failed = [], [], [], []
    start = time.time()

    for i, record in enumerate(benchmark.itertuples(index=False)):
        structure = record.structure
        mb_id = f"mb-{i:05d}"
        try:
            graph = structure_to_graph(structure, ari, gdf, max_num_nbr, radius)
        except Exception as exc:
            # Almost always an element with no row in atom_init.json (the table
            # covers elements 1-100), or a structure with partial occupancies.
            failed.append((mb_id, str(exc)[:70]))
            continue

        ids.append(mb_id)
        graphs.append(graph)
        rows.append({
            "mb_id": mb_id,
            "formula": structure.composition.reduced_formula,
            "n_sites": len(structure),
            "K_VRH": record.K_VRH,
            "G_VRH": record.G_VRH,
        })

        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - start)
            remaining = (len(benchmark) - i - 1) / rate
            print(f"  {i + 1:5d}/{len(benchmark)} graphs  "
                  f"({rate:.0f}/s, ~{remaining / 60:.0f} min left)")

    print(f"Built {len(graphs)} graphs in {(time.time() - start) / 60:.1f} min")
    if failed:
        print(f"  {len(failed)} structures failed to convert, e.g. {failed[:3]}")
    return ids, graphs, pd.DataFrame(rows)


def match_our_cifs(cif_dir, benchmark):
    """Map our mp-ids onto matbench indices, for provenance.

    Same two-pass approach as step 1: bucket by reduced formula (cheap), then
    confirm with StructureMatcher (exact, symmetry-aware). We only need to know
    WHICH of our CIFs are in the training set, not to copy any labels.
    """
    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    print(f"\nMatching {len(paths)} local CIFs against the benchmark for provenance...")

    # Pass 1: bucket the benchmark by reduced formula.
    buckets = collections.defaultdict(list)
    for i, structure in enumerate(benchmark.structure):
        buckets[structure.composition.reduced_formula].append(i)

    # primitive_cell=True reduces both crystals first, so a 2x2x2 supercell
    # still matches the single cell it was built from.
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=False)

    rows = []
    for path in paths:
        mp_id = os.path.splitext(os.path.basename(path))[0]
        try:
            structure = Structure.from_file(path)
        except Exception:
            continue

        formula = structure.composition.reduced_formula
        for i in buckets.get(formula, []):
            if matcher.fit(structure, benchmark.structure[i]):
                rows.append({"mp_id": mp_id, "mb_id": f"mb-{i:05d}",
                             "formula": formula})
                break  # first match wins; duplicates in matbench are rare

    print(f"  {len(rows)} of our CIFs are present in matbench")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data_full"))
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"),
                        help="our own CIFs, matched only to record provenance")
    parser.add_argument("--atom-init",
                        default=os.path.join(PROJECT_ROOT, "data", "atom_init.json"))
    # These three MUST stay in step with what CIFData defaults to, or the cached
    # graphs will not match graphs built from CIFs at prediction time.
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--radius", type=float, default=8)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--skip-match", action="store_true",
                        help="skip the provenance matching pass (saves ~5 min)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N crystals (smoke testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    benchmark = load_benchmark()
    if args.limit:
        print(f"  --limit {args.limit}: truncating (this is a smoke test, "
              f"do not train on the result)")
        benchmark = benchmark.head(args.limit)

    ids, graphs, labels = build_graphs(
        benchmark, args.atom_init, args.max_num_nbr, args.radius, args.step)
    if not graphs:
        sys.exit("No graphs were built - nothing to train on.")

    # --- Write the graph cache ---------------------------------------------
    # The featurisation settings ride along with the tensors so that a stale
    # cache built with a different radius can be detected instead of silently
    # feeding the model the wrong features.
    cache_path = os.path.join(args.out_dir, "graphs.pt")
    torch.save({"ids": ids, "graphs": graphs,
                "config": {"max_num_nbr": args.max_num_nbr,
                           "radius": args.radius, "step": args.step}},
               cache_path)
    size_mb = os.path.getsize(cache_path) / 1e6

    labels_path = os.path.join(args.out_dir, "labels.csv")
    labels.to_csv(labels_path, index=False)

    if not args.skip_match:
        mapping = match_our_cifs(args.cif_dir, benchmark)
        mapping.to_csv(os.path.join(args.out_dir, "mp_to_mb.csv"), index=False)

    print(f"\nWrote {len(labels)} labelled crystals to {args.out_dir}/")
    print(f"  graphs.pt   {size_mb:.0f} MB")
    print(f"  labels.csv  {len(labels)} rows")
    print("\nTarget distribution (GPa):")
    print(labels[["K_VRH", "G_VRH", "n_sites"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
