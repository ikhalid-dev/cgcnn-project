#!/usr/bin/env python3
"""
STEP 1 - Build the training set from the matbench elastic benchmark.
====================================================================

WHY THE TRAINING SET IS NOT OUR OWN CIFs
----------------------------------------
`complete-data/` holds 1,213 Materials Project CIFs and no labels at all. A CIF
says where the atoms are; it does not say what the modulus is.

It is tempting to label what we can and train on that - but only 278 of the
1,213 have ever had an elastic tensor computed, so that route discards 97% of
the labels that exist. The labels live in matbench, so TRAIN ON ALL OF MATBENCH
and let our 1,213 CIFs be purely a PREDICTION set:

    matbench_log_kvrh   10,987 structures + log10(bulk modulus  K_VRH)
    matbench_log_gvrh   10,987 structures + log10(shear modulus G_VRH)

This is exactly the data the PINK paper used, and it is 40x more than the
overlap-only route would give.

WHY WE CACHE GRAPHS INSTEAD OF WRITING 11,000 CIFs
--------------------------------------------------
The slow step in this pipeline is not the network, it is turning a crystal into
a graph - a periodic neighbour search for every atom. Round-tripping matbench's
Structure objects through .cif files on disk would mean paying that cost once to
write them, and again on every training run to read them back.

So we convert Structure -> graph directly, once, and pickle the tensors. Both
training runs (bulk and shear) then load the same cache in seconds. The
conversion goes through `cgcnn_scratch.data.structure_to_graph`, the same
function the prediction script uses, so the model is never fed features built
by a different code path than the one it was trained on.

PROVENANCE - WHICH PINK CRYSTALS ARE ALSO TRAINING DATA
-------------------------------------------------------
Some of our 1,213 CIFs are in matbench, so they will end up in the training set.
When we later predict moduli for all 1,213, we need to know which predictions
are honest extrapolation and which are the model recalling something it was
fitted on. So this script also matches our CIFs against the benchmark and
records the mp-id <-> matbench-index mapping in `mp_to_mb.csv`.

Outputs, all under `data_full/`:
    labels.csv      mb_id, formula, n_sites, K_VRH, G_VRH   (moduli in GPa)
    graphs.pt       {"ids": [...], "graphs": [(atom_fea, nbr_fea, nbr_idx), ...]}
    mp_to_mb.csv    mp_id, mb_id, formula  for the CIFs that are in matbench

Run:
    python scripts/01b_prepare_full_dataset.py
"""

import argparse       # parses --out-dir / --cif-dir / --limit etc. from the CLI
import collections     # defaultdict, used to bucket structures by formula
import glob            # expands "*.cif" into a list of matching file paths
import os              # path joining/creation and file-size lookups
import sys             # sys.path mutation and sys.exit() on a fatal error
import time            # wall-clock timing for progress/ETA printouts
import warnings        # suppresses noisy-but-harmless CIF parser warnings

# torch MUST be imported before numpy/pymatgen in this conda env - MKL loads its
# own OpenMP runtime first and the duplicate libiomp5 segfaults torch. Do not
# "tidy" these imports into alphabetical order.
import torch

import numpy as np                                    # not used directly below but kept for parity with sibling scripts' import block
import pandas as pd                                    # DataFrame construction, CSV I/O
from matminer.datasets import load_dataset             # fetches named benchmark datasets (downloads + caches them)
from pymatgen.analysis.structure_matcher import StructureMatcher  # symmetry-aware crystal-structure equality test
from pymatgen.core import Structure                    # parses a .cif file into a Structure object

# CIF parsing is noisy about rounded coordinates and partial occupancies.
# None of it is actionable here, so keep the log readable.
warnings.filterwarnings("ignore")   # silences all Python warnings for the rest of this process

# walks up three directories from this file (scripts/cgcnn/ -> scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of the caller's cwd

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
    kvrh = load_dataset("matbench_log_kvrh")   # DataFrame: columns include `structure` and `log10(K_VRH)`
    gvrh = load_dataset("matbench_log_gvrh")   # DataFrame: columns include `structure` and `log10(G_VRH)`
    assert len(kvrh) == len(gvrh), "the two datasets should be row-aligned"
    print(f"  loaded {len(kvrh)} benchmark entries")

    # Spot-check the alignment on a sample rather than comparing all 11k
    # structures, which would itself take minutes.
    for i in (0, len(kvrh) // 2, len(kvrh) - 1):   # first, middle, and last row indices
        assert (kvrh.structure[i].composition.reduced_formula ==
                gvrh.structure[i].composition.reduced_formula), \
            f"row {i} disagrees between the two datasets - not row-aligned"

    return pd.DataFrame({
        "structure": kvrh.structure,     # pymatgen Structure objects, one per crystal
        "K_VRH": 10 ** kvrh["log10(K_VRH)"],   # undo the log10 matbench stores, back to GPa
        "G_VRH": 10 ** gvrh["log10(G_VRH)"],   # same, for shear modulus
    })


def build_graphs(benchmark, atom_init_path, max_num_nbr, radius, step):
    """Convert every benchmark Structure into a cached crystal graph.

    This is the expensive part - roughly a periodic neighbour search per
    crystal. Returns (ids, graphs, rows) where `rows` is the labels table,
    restricted to the crystals that actually converted.
    """
    ari = AtomFeaturiser(atom_init_path)                 # loads the 92-dim per-element feature table
    gdf = GaussianDistance(dmin=0, dmax=radius, step=step)  # builds the Gaussian bond-distance expansion basis

    ids, graphs, rows, failed = [], [], [], []   # accumulators filled by the loop below
    start = time.time()   # wall-clock start, used for the rate/ETA printout

    for i, record in enumerate(benchmark.itertuples(index=False)):   # iterate rows as lightweight namedtuples
        structure = record.structure           # the pymatgen Structure for this row
        mb_id = f"mb-{i:05d}"                  # zero-padded row-index identifier, e.g. "mb-00042"
        try:
            graph = structure_to_graph(structure, ari, gdf, max_num_nbr, radius)  # atom/bond featurisation -> graph tensors
        except Exception as exc:
            # Almost always an element with no row in atom_init.json (the table
            # covers elements 1-100), or a structure with partial occupancies.
            failed.append((mb_id, str(exc)[:70]))   # record id + truncated error message, then skip this crystal
            continue

        ids.append(mb_id)      # keep the id list in the same order as `graphs`
        graphs.append(graph)   # the featurised graph for this crystal
        rows.append({
            "mb_id": mb_id,
            "formula": structure.composition.reduced_formula,   # e.g. "SiO2"
            "n_sites": len(structure),                          # atom count in the unit cell
            "K_VRH": record.K_VRH,
            "G_VRH": record.G_VRH,
        })

        if (i + 1) % 500 == 0:   # print progress every 500 crystals, not every one
            rate = (i + 1) / (time.time() - start)                  # crystals processed per second so far
            remaining = (len(benchmark) - i - 1) / rate              # estimated seconds left at the current rate
            print(f"  {i + 1:5d}/{len(benchmark)} graphs  "
                  f"({rate:.0f}/s, ~{remaining / 60:.0f} min left)")

    print(f"Built {len(graphs)} graphs in {(time.time() - start) / 60:.1f} min")
    if failed:
        print(f"  {len(failed)} structures failed to convert, e.g. {failed[:3]}")   # show only the first 3 as a sample
    return ids, graphs, pd.DataFrame(rows)


def match_our_cifs(cif_dir, benchmark):
    """Map our mp-ids onto matbench indices, for provenance.

    Two passes: bucket by reduced formula (cheap), then confirm with
    StructureMatcher (exact, symmetry-aware). We only need to know
    WHICH of our CIFs are in the training set, not to copy any labels.
    """
    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))   # every local CIF file path, alphabetically ordered
    print(f"\nMatching {len(paths)} local CIFs against the benchmark for provenance...")

    # Pass 1: bucket the benchmark by reduced formula.
    buckets = collections.defaultdict(list)   # formula string -> list of matbench row indices
    for i, structure in enumerate(benchmark.structure):
        buckets[structure.composition.reduced_formula].append(i)   # group row indices sharing the same formula

    # primitive_cell=True reduces both crystals first, so a 2x2x2 supercell
    # still matches the single cell it was built from.
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=False)

    rows = []
    for path in paths:
        mp_id = os.path.splitext(os.path.basename(path))[0]   # filename without extension, e.g. "mp-1234"
        try:
            structure = Structure.from_file(path)   # parse the CIF into a pymatgen Structure
        except Exception:
            continue   # unparsable CIF - skip it, it simply won't get a provenance row

        formula = structure.composition.reduced_formula
        for i in buckets.get(formula, []):   # only compare against benchmark rows with a matching formula
            if matcher.fit(structure, benchmark.structure[i]):   # True if the two structures are the same crystal
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
                        default=os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"))
    # These three MUST match what 04_predict_moduli.py featurises with, or the
    # model would be fed prediction features unlike its training features.
    # They are recorded into graphs.pt so prediction reads them back rather
    # than assuming.
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--radius", type=float, default=8)
    parser.add_argument("--step", type=float, default=0.2)
    parser.add_argument("--skip-match", action="store_true",
                        help="skip the provenance matching pass (saves ~5 min)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N crystals (smoke testing)")
    args = parser.parse_args()   # reads sys.argv, returns a Namespace with the fields above

    os.makedirs(args.out_dir, exist_ok=True)   # create data_full/ if it does not already exist
    benchmark = load_benchmark()               # the full 10,987-row matbench DataFrame
    if args.limit:
        print(f"  --limit {args.limit}: truncating (this is a smoke test, "
              f"do not train on the result)")
        benchmark = benchmark.head(args.limit)   # keep only the first N rows

    ids, graphs, labels = build_graphs(
        benchmark, args.atom_init, args.max_num_nbr, args.radius, args.step)
    if not graphs:
        sys.exit("No graphs were built - nothing to train on.")   # abort with a non-zero exit code

    # --- Write the graph cache ---------------------------------------------
    # The featurisation settings ride along with the tensors so that a stale
    # cache built with a different radius can be detected instead of silently
    # feeding the model the wrong features.
    cache_path = os.path.join(args.out_dir, "graphs.pt")
    torch.save({"ids": ids, "graphs": graphs,
                "config": {"max_num_nbr": args.max_num_nbr,
                           "radius": args.radius, "step": args.step}},
               cache_path)   # serialises the dict (ids, graph tensors, settings) to disk in one file
    size_mb = os.path.getsize(cache_path) / 1e6   # file size in megabytes, for the printout below

    labels_path = os.path.join(args.out_dir, "labels.csv")
    labels.to_csv(labels_path, index=False)   # write the labels table, no pandas row-index column

    if not args.skip_match:
        mapping = match_our_cifs(args.cif_dir, benchmark)   # DataFrame of mp_id/mb_id/formula rows
        mapping.to_csv(os.path.join(args.out_dir, "mp_to_mb.csv"), index=False)

    print(f"\nWrote {len(labels)} labelled crystals to {args.out_dir}/")
    print(f"  graphs.pt   {size_mb:.0f} MB")
    print(f"  labels.csv  {len(labels)} rows")
    print("\nTarget distribution (GPa):")
    print(labels[["K_VRH", "G_VRH", "n_sites"]].describe().round(2).to_string())   # summary stats: count/mean/std/min/max/quartiles


if __name__ == "__main__":
    main()
