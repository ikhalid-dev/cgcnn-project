#!/usr/bin/env python3
"""
STEP 35 - Merge AFLOW soft crystals into the matbench training set.
====================================================================

    python scripts/cgcnn/35_merge_aflow.py

WHAT PROBLEM THIS SOLVES
------------------------
The model is worst exactly where the screen operates. Stratified by true kappa:

    Q1 (lowest kappa)  MAE log10(G) 0.1340   median true G  10.5 GPa
    Q4                 MAE log10(G) 0.0565   median true G  58.5 GPa

Q1 crystals are SOFT, and matbench barely contains any:

                          G < 5   G < 10   G < 20 GPa
    matbench                205      771     2371   (of 10,987)
    AFLOW soft subset        84      321      953

Adding the AFLOW entries enlarges the soft tail by roughly 40%. Loss weighting
(31_train_joint.py's --kappa-weight-alpha) makes the model try harder on the
soft crystals it already has; this gives it more of them. The two are
complementary and are meant to be used together.

THE ONE RULE THAT MATTERS
-------------------------
AFLOW crystals go into TRAINING ONLY. Never validation, never test.

The test set stays exactly the 1,648 matbench crystals selected by
split_seed=42, unchanged from round 1. If it moved, none of the numbers from
six rounds of experiments would be comparable to anything after this, and the
whole baseline table would have to be regenerated. The merged cache therefore
keeps every matbench entry first, in its original order, with AFLOW appended
after - so matbench index i still means the same crystal it always did, and
split_indices() can slice the matbench block exactly as before.

MIXING TWO DFT SOURCES IS NOT FREE
----------------------------------
AFLOW's AEL moduli and matbench's VRH moduli are the same physical quantity
computed by different codes with different settings. They will not agree
exactly. This script measures the disagreement on compounds present in both
and prints it before merging - which is also the only direct estimate we have
of the irreducible label noise in this regime. If that disagreement is large
relative to the model's own error, added data will not help much, and it is
better to know that before spending an hour of GPU time.

OUTPUTS
-------
    data_full/graphs_merged.pt    matbench graphs, then AFLOW graphs
    data_full/labels_merged.csv   same order, with a `source` column
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    "data_dir": "data_full",
    "struct_dir": "data_full/aflow_structures",
    "aflow_csv": "data_full/aflow_soft.csv",
    "out_cache": "data_full/graphs_merged.pt",
    "out_labels": "data_full/labels_merged.csv",

    # Graph construction. These MUST match whatever built the existing
    # graphs.pt, or the AFLOW crystals would be featurised differently from the
    # matbench ones and the model would see two incompatible input conventions.
    # The values are read back from the cache's own config where possible and
    # only used as a fallback.
    "max_num_nbr": 12,
    "radius": 8.0,

    # Drop AFLOW entries whose formula already appears in matbench. Keeping
    # them would duplicate crystals across the train/test boundary - an AFLOW
    # copy of a matbench test crystal sitting in training is leakage.
    "drop_formula_overlap": 1,

    # 0 = process everything; a small number smoke-tests the path.
    "limit": 0,

    # Must match the trainer's split_seed, or the val/test formulas identified
    # below would be the wrong ones and real leakage could slip through.
    "split_seed": 42,
}
# =============================================================================

import argparse   # CLI flag parsing (one --flag per CONFIG key, added below)
import os          # path joining/creation, path existence checks
import sys          # sys.path mutation and sys.exit()/SystemExit on fatal errors
import warnings      # suppresses noisy-but-harmless parser warnings

# torch before numpy/pandas/pymatgen: MKL loads its own OpenMP runtime and the
# duplicate libiomp5 aborts the process otherwise.
import torch
import numpy as np    # log-space arithmetic for the cross-source disagreement check
import pandas as pd   # DataFrame construction, CSV I/O, merge/concat

warnings.filterwarnings("ignore")   # silences all Python warnings for the rest of this process

# walks up three directories from this file (scripts/cgcnn/ -> scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of the caller's cwd

from cgcnn_scratch.data import (  # noqa: E402
    AtomFeaturiser, GaussianDistance, structure_to_graph)


def parse_overrides():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, default in CONFIG.items():   # one --flag per CONFIG entry, generated generically
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default), default=None,
                            help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()   # reads sys.argv into a Namespace, unset flags default to None
    cfg = dict(CONFIG)            # start from a copy of the defaults
    for key, value in vars(args).items():
        if value is not None:      # only overwrite a default if the flag was actually passed
            cfg[key] = value
    return cfg


def reduced_formula(formula):
    """Normalise a composition string so two databases can be compared.

    AFLOW writes 'He8Ne4' where matbench writes 'He2Ne1'-style reduced forms,
    so a raw string comparison finds almost no overlap even when the compounds
    are the same. pymatgen's reduced formula is the common ground.
    """
    from pymatgen.core import Composition   # parses a formula string into element:count pairs
    try:
        return Composition(formula).reduced_formula   # smallest-integer-ratio formula, e.g. "He2Ne1"
    except Exception:
        return None   # unparsable formula string; caller treats None as "no match possible"


def main():
    cfg = parse_overrides()
    data_dir = os.path.join(PROJECT_ROOT, cfg["data_dir"])
    struct_dir = os.path.join(PROJECT_ROOT, cfg["struct_dir"])

    # ---- load what already exists ------------------------------------------
    cache_path = os.path.join(data_dir, "graphs.pt")
    labels_path = os.path.join(data_dir, "labels.csv")
    for p in (cache_path, labels_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")   # abort early with a clear message rather than failing deeper in
    aflow_path = os.path.join(PROJECT_ROOT, cfg["aflow_csv"])
    if not os.path.exists(aflow_path):
        raise SystemExit(f"missing {aflow_path}\n"
                         f"Run scripts/cgcnn/34_fetch_aflow_structures.py first.")

    print("Loading existing matbench cache...")
    blob = torch.load(cache_path, weights_only=False)   # dict with "ids", "graphs", and featuriser "config"
    mb_ids, mb_graphs = list(blob["ids"]), list(blob["graphs"])   # copy out of the loaded dict into plain lists
    mb_labels = pd.read_csv(labels_path)
    print(f"  {len(mb_ids)} matbench graphs, {len(mb_labels)} label rows")

    # Reuse the cache's own featuriser settings wherever it recorded them, so
    # the AFLOW graphs are built identically rather than merely similarly.
    conf = blob.get("config", {}) or {}   # {} if the key is missing or explicitly None/empty
    max_num_nbr = int(conf.get("max_num_nbr", cfg["max_num_nbr"]))   # cache value wins, CONFIG is only the fallback
    radius = float(conf.get("radius", cfg["radius"]))
    print(f"  featuriser: max_num_nbr={max_num_nbr}, radius={radius} "
          f"({'from cache config' if conf else 'from CONFIG fallback'})")

    aflow = pd.read_csv(aflow_path)
    if cfg["limit"]:
        aflow = aflow.head(cfg["limit"])   # smoke-test: keep only the first N rows
    print(f"  {len(aflow)} AFLOW soft candidates")

    # ---- how much do the two DFT sources actually disagree? ----------------
    print("\nCross-checking the two DFT sources on shared compounds...")
    mb_labels["_rf"] = [reduced_formula(f) for f in mb_labels.formula]   # normalised formula, one per matbench row
    aflow["_rf"] = [reduced_formula(f) for f in aflow.compound]           # normalised formula, one per AFLOW row
    shared = aflow.merge(mb_labels.dropna(subset=["_rf"]), on="_rf",       # inner join: only formulas present in both
                         suffixes=("_af", "_mb"))
    if len(shared):
        # Compare in log space - that is the space the model is trained and
        # scored in, so a log discrepancy is directly comparable to its MAE.
        dK = np.log10(shared.ael_bulk_modulus_vrh) - np.log10(shared.K_VRH)    # AFLOW's log10(K) minus matbench's
        dG = np.log10(shared.ael_shear_modulus_vrh) - np.log10(shared.G_VRH)  # same, for shear modulus
        ok = np.isfinite(dK) & np.isfinite(dG)   # excludes any row where a modulus was zero/negative/missing
        print(f"  {ok.sum()} compounds present in both")
        print(f"    log10(K) disagreement  MAE {dK[ok].abs().mean():.4f}"
              f"   bias {dK[ok].mean():+.4f}")
        print(f"    log10(G) disagreement  MAE {dG[ok].abs().mean():.4f}"
              f"   bias {dG[ok].mean():+.4f}")
        print(f"    for scale, the model's own test MAE is ~0.063 (K) / ~0.078 (G)")
        if dG[ok].abs().mean() > 0.10:
            print("    WARNING: the two sources disagree by more than the model's"
                  " own error.\n             Added data will carry that noise"
                  " into training.")
    else:
        print("  no shared compounds found - cannot estimate cross-source noise")

    # ---- drop overlaps so nothing leaks across the split -------------------
    if cfg["drop_formula_overlap"]:
        # Only VAL/TEST overlaps are leakage. An AFLOW crystal sharing a
        # formula with a matbench TRAINING crystal is just a near-duplicate in
        # training, which is harmless - dropping those too threw away 546 of
        # 951 entries for no benefit. Reproduce the split here (it is
        # deterministic given split_seed) to tell the two cases apart.
        rng = np.random.RandomState(cfg["split_seed"])       # seeded PRNG, same seed the trainer uses
        perm = rng.permutation(len(mb_labels))                # shuffled row-index order over the matbench labels
        n_train = int(round(0.70 * len(mb_labels)))           # matches the trainer's 70% train fraction
        n_val = int(round(0.15 * len(mb_labels)))              # matches the trainer's 15% val fraction
        heldout_pos = perm[n_train:n_train + 2 * n_val]      # val + test
        heldout_formulas = set(mb_labels.iloc[heldout_pos]["_rf"].dropna())   # normalised formulas in val+test

        before = len(aflow)
        aflow = aflow[~aflow["_rf"].isin(heldout_formulas)]   # drop any AFLOW row matching a held-out formula
        print(f"\n  dropped {before - len(aflow)} AFLOW entries whose formula "
              f"appears in the matbench VAL/TEST split (leakage)")
        print(f"  kept overlaps with the TRAIN split - harmless near-duplicates")

    # ---- build graphs for the survivors ------------------------------------
    from pymatgen.core import Structure   # parses POSCAR text into a Structure object
    ari = AtomFeaturiser(
        os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"))   # per-element feature table
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)                 # Gaussian bond-distance expansion basis

    print(f"\nBuilding graphs for {len(aflow)} AFLOW crystals...")
    new_ids, new_graphs, new_rows, failed = [], [], [], 0   # accumulators filled by the loop below; failed is a running count
    for i, row in enumerate(aflow.itertuples(), 1):   # 1-based counter for the progress printout
        path = os.path.join(struct_dir, row.poscar_file)
        try:
            # from_str, not from_file: pymatgen dispatches on the FILENAME
            # extension and does not recognise ".poscar", so from_file raises
            # "Unrecognized extension" on every one of these. Naming the format
            # explicitly is both correct and independent of what we call the file.
            with open(path) as fh:
                structure = Structure.from_str(fh.read(), fmt="poscar")   # explicit format bypasses extension sniffing
            graph = structure_to_graph(structure, ari, gdf,
                                       max_num_nbr=max_num_nbr, radius=radius)
        except Exception as exc:
            failed += 1
            # Report the first few rather than swallowing them. A silent
            # except turned "every single graph failed" into a one-line
            # message with no cause attached.
            if failed <= 3:
                print(f"    {row.poscar_file}: {type(exc).__name__}: {exc}")
            continue
        cid = "af-" + row.auid.split(":")[-1][:12]   # short id derived from AFLOW's own auid string
        new_ids.append(cid)
        new_graphs.append(graph)
        new_rows.append({
            "mb_id": cid,
            "formula": row.compound,
            "n_sites": int(row.natoms),
            "K_VRH": float(row.ael_bulk_modulus_vrh),
            "G_VRH": float(row.ael_shear_modulus_vrh),
            "source": "aflow",
        })
        if i % 100 == 0 or i == len(aflow):   # progress line every 100 crystals, plus a final one
            print(f"  {i:5d}/{len(aflow)}  built {len(new_ids):5d}  failed {failed:4d}",
                  flush=True)   # flush so the line appears immediately even when stdout is redirected to a file

    if not new_ids:
        raise SystemExit("no AFLOW graphs built - check the POSCAR files")

    # ---- write the merged cache, matbench FIRST ----------------------------
    # Order is load-bearing: split_indices() in the trainer slices the first
    # len(matbench) entries exactly as it always has, so the test set is
    # bit-identical to every previous round.
    out_cache = os.path.join(PROJECT_ROOT, cfg["out_cache"])
    torch.save({"ids": mb_ids + new_ids,                 # matbench ids first, AFLOW ids appended after
                "graphs": mb_graphs + new_graphs,         # same order as ids, element for element
                "config": {**conf, "max_num_nbr": max_num_nbr, "radius": radius,
                           "n_matbench": len(mb_ids), "n_aflow": len(new_ids)}},
               out_cache)   # serialises the combined dict to disk in one file

    mb_out = mb_labels.drop(columns=["_rf"]).copy()   # drop the helper column added above, not part of the schema
    mb_out["source"] = "matbench"
    merged = pd.concat([mb_out, pd.DataFrame(new_rows)], ignore_index=True)   # stack matbench rows, then AFLOW rows
    out_labels = os.path.join(PROJECT_ROOT, cfg["out_labels"])
    merged.to_csv(out_labels, index=False)

    print()
    print("=" * 74)
    print(f"  {len(mb_ids)} matbench + {len(new_ids)} AFLOW = {len(merged)} crystals")
    print(f"  -> {os.path.relpath(out_cache, PROJECT_ROOT)}")
    print(f"  -> {os.path.relpath(out_labels, PROJECT_ROOT)}")
    print("=" * 74)
    print("  soft-tail coverage after the merge:")
    for thr in (5, 10, 20):
        mb_n = (mb_out.G_VRH < thr).sum()                              # matbench crystals softer than this threshold
        af_n = sum(r["G_VRH"] < thr for r in new_rows)                  # same count among the newly added AFLOW rows
        print(f"    G < {thr:>2} GPa:  matbench {mb_n:5d}  + AFLOW {af_n:5d}"
              f"  = {mb_n + af_n:5d}   ({100 * af_n / max(mb_n, 1):+.0f}%)")
    print("\n  AFLOW rows carry source='aflow'; the trainer must force them all")
    print("  into TRAIN so the 1,648-crystal test set never changes.")


if __name__ == "__main__":
    main()
