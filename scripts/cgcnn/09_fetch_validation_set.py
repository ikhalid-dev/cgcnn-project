#!/usr/bin/env python3
"""
STEP 9 - Fetch PINK's own real-data validation set, and check it against ours.
================================================================================

    python scripts/09_fetch_validation_set.py

WHY THIS SCRIPT EXISTS
----------------------
Every validation this project has run so far compares our model against the
PAPER'S OWN pretrained model (scripts/08_compare_kappa.py) - useful, but it is
still model-vs-model, not model-vs-reality. The paper itself hands us
something better and, until now, unused: Table 1 lists 46 real materials with
EXPERIMENTALLY MEASURED kappa_L, mp-ids included. This script fetches the 45
of those we don't already have, runs our existing pipeline on them unchanged,
and compares our kappa_L against those measured values directly.

THE 46 MATERIALS, TRANSCRIBED FROM THE PAPER
----------------------------------------------
Table 1 of Liu et al., J. Mater. Inf. 2025, 5, 12 (PINK.pdf, pages 10-11 in
this repo). kappa_exp is the room-temperature experimental value in W/m/K, as
tabulated in the paper (references [20,54-57] therein - see the paper for the
underlying experimental sources). kappa_pink, G_paper, vs_paper, and
gamma_paper are the PAPER'S OWN pre-trained model's values for that same
material - its predicted kappa_L, shear modulus, average sound velocity, and
Gruneisen parameter, respectively - also transcribed from Table 1. Having all
four (not just kappa_pink) lets Phase 1 compare our from-scratch model
against real ground truth AND, when the aggregate kappa_L disagrees, actually
tell WHY by checking each intermediate quantity against the paper's own
- rather than only ever seeing one final number with no way to localise a gap.
mp-22922 (AgCl) is already in complete-data/; every other row here is new.

Transcription verified two ways: (1) spot-checked several rows against the
raw PDF text extraction by hand, (2) recomputed summary statistics from every
kappa_exp/kappa_pink pair and confirmed against the paper's own stated numbers
for this exact table (Results section, "Figure 6B"): R^2 = 0.881 matches
exactly; the paper's stated "MAE of 0.526" turns out to be natural-log MAE,
not log10 - log10 MAE here is 0.228, and 0.228 * ln(10) = 0.525, matching to
3 decimals. Both checks passed, so the transcription is treated as correct.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
It does not re-derive the Slack-model physics (that's `slack_physics()` in
07_predict_kappa.py, already verified against pink_predict.py's own
implementation) or the moduli-prediction logic (04_predict_moduli.py). Both
are reused as subprocesses, pointed at this new, small CIF directory, so this
validation runs through the *exact* same code path as every other prediction
in this project - not a parallel implementation that could quietly diverge.

PROVENANCE, AGAIN
------------------
Same discipline as everywhere else in this project: before trusting a
prediction as evidence, check whether the model was trained on that crystal.
None of these 46 materials were in complete-data/ when data_full/graphs.pt was
built, so they cannot appear in matbench via that route - but several of them
(AgCl, NaCl, MgO, Si, ...) are exactly the kind of simple, well-studied binary
compound matbench itself is likely to include independently. So this script
re-runs the same formula-bucket-then-StructureMatcher check
scripts/01b_prepare_full_dataset.py uses, against the live matbench structures,
rather than assuming "not in complete-data" means "not trained on."
"""

import argparse       # CLI flag parsing
import os              # path joining/creation, path existence checks
import shutil           # copies an existing CIF into the working cif_dir
import subprocess         # runs 04_predict_moduli.py / 07_predict_kappa.py as child processes
import sys                 # sys.exit() on a fatal error, sys.executable to relaunch the same python
import warnings             # suppresses noisy-but-harmless parser warnings below

import torch  # noqa: F401  (see cgcnn_scratch/data.py - import order matters)

import numpy as np                                        # not used directly below but kept for parity with sibling scripts' import block
import pandas as pd                                        # DataFrame construction, CSV I/O
from pymatgen.analysis.structure_matcher import StructureMatcher  # symmetry-aware crystal-structure equality test
from pymatgen.core import Structure                        # parses a .cif file into a Structure object

warnings.filterwarnings("ignore")   # silences all Python warnings for the rest of this process

# walks up three directories from this file (scripts/cgcnn/ -> scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (mp_id, formula, kappa_exp, kappa_pink, G_paper, vs_paper, gamma_paper), all
# in SI units matching Table 1's own columns (kappa in W/m/K, G in GPa, vs in
# m/s), transcribed in full from Table 1. kappa_pink, G_paper, vs_paper, and
# gamma_paper are the paper's OWN pre-trained model's values for that same
# material - see the module docstring for how this transcription was
# verified, and scripts/10_validate_table1.py for why all four matter (not
# just kappa_pink): they let Phase 1 isolate exactly which stage of the
# pipeline (moduli, sound velocity, or Gruneisen parameter) is responsible
# for any gap against real experimental values, rather than only seeing an
# aggregate kappa_L number with no way to tell why it differs. The paper
# lists two distinct KBr entries (mp-23251, mp-570891) - kept as two rows,
# since they are two different structures the paper treated separately.
# (one literal tuple per material - the schema is the comment above; not
# annotated row by row, that would repeat the same 7-field shape 46 times)
TABLE1 = [
    ("mp-22922", "AgCl", 1.0, 1.091, 8.801, 1423.597, 1.9),
    ("mp-2172", "AlAs", 98, 47.054, 40.51, 3733.087, 0.66),
    ("mp-2624", "AlSb", 56, 30.62, 28.808, 2959.047, 0.6),
    ("mp-1639", "BN", 760, 727.981, 350.203, 10995.529, 0.7),
    ("mp-1479", "BP", 350, 344.207, 201.292, 9011.073, 0.923),
    ("mp-66", "C", 3000, 1320.871, 547.436, 13613.398, 0.75),
    ("mp-2605", "CaO", 27, 32.552, 63.316, 4863.622, 1.57),
    ("mp-406", "CdTe", 7.5, 10.797, 14.349, 1818.434, 0.52),
    ("mp-2534", "GaAs", 45, 42.381, 45.152, 3291.363, 0.75),
    ("mp-3490", "GaP", 100, 50.725, 48.296, 3846.005, 0.75),
    ("mp-1156", "GaSb", 40, 22.115, 28.033, 2557.748, 0.75),
    ("mp-32", "Ge", 65, 33.219, 47.134, 3362.109, 1.06),
    ("mp-20305", "InAs", 30, 20.454, 23.662, 2355.122, 0.57),
    ("mp-20351", "InP", 93, 25.94, 27.554, 2742.762, 0.6),
    ("mp-20012", "InSb", 20, 14.244, 17.668, 2026.668, 0.56),
    ("mp-23251", "KBr", 3.4, 1.737, 6.156, 1709.446, 1.45),
    ("mp-570891", "KBr", 3.4, 2.502, 8.391, 1885.943, 1.45),
    ("mp-23193", "KCl", 7.1, 2.059, 6.387, 2050.803, 1.45),
    ("mp-22898", "KI", 2.6, 1.585, 5.79, 1546.914, 1.45),
    ("mp-1009009", "LiF", 17.6, 28.999, 58.174, 5236.933, 1.5),
    ("mp-23703", "LiH", 15, 34.63, 39.346, 7537.234, 1.28),
    ("mp-1265", "MgO", 60, 86.06, 123.811, 6564.689, 1.44),
    ("mp-22916", "NaBr", 2.8, 4.897, 14.495, 2392.521, 1.5),
    ("mp-22862", "NaCl", 7.1, 6.531, 16.65, 3123.706, 1.56),
    ("mp-682", "NaF", 18.4, 7.651, 21.929, 3171.392, 1.5),
    ("mp-23268", "NaI", 1.8, 1.521, 6.823, 1547.0, 1.56),
    ("mp-21276", "PbS", 2.9, 4.772, 26.485, 2111.102, 2.0),
    ("mp-2201", "PbSe", 2, 6.189, 22.618, 1876.743, 1.5),
    ("mp-22867", "RbBr", 3.8, 1.974, 6.905, 1651.509, 1.45),
    ("mp-23295", "RbCl", 2.8, 2.244, 7.337, 1854.222, 1.45),
    ("mp-22903", "RbI", 2.3, 1.814, 6.228, 1517.862, 1.41),
    ("mp-149", "Si", 166, 60.869, 55.828, 5480.852, 1.06),
    ("mp-8062", "SiC", 490, 438.312, 222.879, 9107.212, 0.75),
    ("mp-2472", "SrO", 12, 19.845, 47.864, 3468.152, 1.52),
    ("mp-19717", "PbTe", 2.5, 2.711, 17.511, 1674.996, 2.009),
    ("mp-10695", "ZnS", 27, 28.268, 32.768, 3191.504, 0.75),
    ("mp-1190", "ZnSe", 19, 24.432, 31.048, 2763.063, 0.75),
    ("mp-2176", "ZnTe", 18, 15.097, 25.383, 2416.126, 0.97),
    ("mp-661", "AlN", 350, 140.931, 135.549, 7197.895, 0.7),
    ("mp-2542", "BeO", 370, 104.885, 123.309, 7116.682, 0.75),
    ("mp-672", "CdS", 16, 6.79, 16.885, 2166.455, 0.75),
    ("mp-804", "GaN", 210, 79.333, 110.843, 4794.507, 0.7),
    ("mp-2133", "ZnO", 60, 13.479, 33.376, 2790.196, 0.75),
    ("mp-34202", "Bi2Te3", 1.6, 1.196, 10.489, 1339.685, 1.49),
    ("mp-1143", "Al2O3", 30, 33.445, 132.798, 6501.426, 1.34),
    ("mp-753", "ZnSb", 3.5, 2.315, 32.387, 2518.332, 1.681),
]


def fetch_structures(cif_dir, complete_data_dir):
    """Fetch every Table 1 material not already sitting in complete-data/.

    Requires a Materials Project API key discoverable by pymatgen (PMG_MAPI_KEY
    in ~/.pmgrc.yaml, or the MP_API_KEY / PMG_MAPI_KEY env var).
    """
    from mp_api.client import MPRester         # Materials Project REST client
    from pymatgen.core import SETTINGS          # reads pymatgen's own config file (~/.pmgrc.yaml)

    os.makedirs(cif_dir, exist_ok=True)   # create the output directory if missing
    key = SETTINGS.get("PMG_MAPI_KEY") or os.environ.get("MP_API_KEY")   # config file first, env var as fallback
    if not key:
        sys.exit("No Materials Project API key found. Set PMG_MAPI_KEY in "
                 "~/.pmgrc.yaml or export MP_API_KEY.")

    to_fetch = [mp_id for mp_id, *_ in TABLE1                                            # unpack mp_id, ignore the other 6 fields
               if not os.path.exists(os.path.join(complete_data_dir, f"{mp_id}.cif"))]   # keep only ids without an existing CIF
    print(f"{len(TABLE1)} materials in Table 1; {len(to_fetch)} need fetching "
         f"({len(TABLE1) - len(to_fetch)} already in complete-data/)")

    failed = []
    with MPRester(key) as mpr:   # opens (and auto-closes) an authenticated MP API session
        for i, mp_id in enumerate(to_fetch, 1):   # 1-based counter for the progress printout
            out_path = os.path.join(cif_dir, f"{mp_id}.cif")
            if os.path.exists(out_path):
                continue   # already fetched in a previous run of this script
            try:
                structure = mpr.get_structure_by_material_id(mp_id)   # fetch the relaxed structure for this mp-id
            except Exception as exc:
                failed.append((mp_id, str(exc)[:80]))   # record id + truncated error
                print(f"  [{i}/{len(to_fetch)}] SKIPPED {mp_id} - {str(exc)[:80]}")
                continue
            if structure is None:
                # A handful of the paper's mp-ids (published 2025) have since
                # been deprecated/merged into a different ID as MP's own
                # database is periodically consolidated. Skip rather than
                # crash the whole fetch over one stale ID - reported clearly
                # so it's visible in the final comparison, not silently
                # dropped.
                failed.append((mp_id, "no structure returned (likely a "
                                      "deprecated/merged mp-id)"))
                print(f"  [{i}/{len(to_fetch)}] SKIPPED {mp_id} - no structure "
                     f"returned")
                continue
            structure.to(filename=out_path)   # write the fetched structure out as a CIF file
            print(f"  [{i}/{len(to_fetch)}] fetched {mp_id}")

    if failed:
        print(f"\n{len(failed)} material(s) could not be fetched:")
        for mp_id, reason in failed:
            print(f"  {mp_id}: {reason}")

    # Every Table 1 material, fetched or pre-existing, needs to end up in
    # cif_dir so 04_predict_moduli.py can be pointed at one directory.
    for mp_id, *_ in TABLE1:
        dest = os.path.join(cif_dir, f"{mp_id}.cif")
        if not os.path.exists(dest):                                    # not yet fetched into cif_dir this run
            src = os.path.join(complete_data_dir, f"{mp_id}.cif")
            if os.path.exists(src):                                     # it was already sitting in complete-data/
                shutil.copy(src, dest)                                  # copy it across so cif_dir has everything

    return [mp_id for mp_id, _ in failed]   # unpack (mp_id, reason) pairs, keep only the ids


def check_provenance(mp_ids, cif_dir):
    """Which of these materials are ALSO in the matbench training set?

    Same two-pass structure match as 01b_prepare_full_dataset.py: bucket by
    reduced formula (cheap), confirm with StructureMatcher (exact, symmetry-
    aware). Simple, well-studied binaries like these are exactly the kind of
    crystal matbench is likely to contain independently of complete-data/, so
    "not originally in complete-data/" must not be treated as "unseen" without
    actually checking.
    """
    from matminer.datasets import load_dataset   # fetches/loads the named matbench dataset

    print("\nChecking whether these materials are in the matbench training set...")
    kvrh = load_dataset("matbench_log_kvrh")   # the same 10,987-row training benchmark 01b uses
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=False)   # symmetry-aware equality test

    buckets = {}   # reduced formula -> list of matbench row indices sharing it
    for i, structure in enumerate(kvrh.structure):
        buckets.setdefault(structure.composition.reduced_formula, []).append(i)

    provenance = {}   # mp_id -> "in_matbench_training" or "unseen"
    for mp_id in mp_ids:
        cif_path = os.path.join(cif_dir, f"{mp_id}.cif")
        if not os.path.exists(cif_path):
            continue  # fetch failed for this one; excluded upstream already
        structure = Structure.from_file(cif_path)   # parse the local CIF
        formula = structure.composition.reduced_formula
        in_matbench = any(matcher.fit(structure, kvrh.structure[idx])         # True if this structure equals any bucket member
                         for idx in buckets.get(formula, []))
        provenance[mp_id] = "in_matbench_training" if in_matbench else "unseen"

    n_train = sum(v == "in_matbench_training" for v in provenance.values())   # count of True-valued entries
    print(f"  {n_train}/{len(mp_ids)} are in the matbench training set "
         f"(predictions for these are recall, not generalisation)")
    return provenance


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "data", "table1_validation"))
    parser.add_argument("--complete-data", default=os.path.join(PROJECT_ROOT, "complete-data"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))
    parser.add_argument("--skip-fetch", action="store_true",
                        help="CIFs already fetched; just (re)run provenance + pipeline")
    args = parser.parse_args()   # reads sys.argv, returns a Namespace with the fields above

    print("=== Fetching PINK's Table 1 real-data validation set ===\n")
    failed_ids = []
    if not args.skip_fetch:
        failed_ids = fetch_structures(args.cif_dir, args.complete_data)   # list of mp_ids that could not be fetched

    mp_ids = [mp_id for mp_id, *_ in TABLE1 if mp_id not in failed_ids]   # every id we actually have a CIF for
    provenance = check_provenance(mp_ids, args.cif_dir)

    reference = pd.DataFrame(TABLE1, columns=["material_id", "formula", "kappa_exp", "kappa_pink", "G_paper", "vs_paper", "gamma_paper"])
    reference = reference[~reference.material_id.isin(failed_ids)].copy()   # drop rows whose fetch failed
    reference["provenance"] = reference.material_id.map(provenance)         # join the in_matbench_training/unseen label
    # 09_ prefix: this script's own number, so this result can be told apart
    # from any other script's output sitting in the same results/ directory.
    ref_path = os.path.join(args.results_dir, "09_table1_reference.csv")
    reference.to_csv(ref_path, index=False)
    print(f"\nWrote {ref_path} ({len(reference)}/{len(TABLE1)} materials; "
         f"{len(failed_ids)} excluded: {failed_ids})")

    print("\n--- Running our moduli ensemble on these 46 crystals ---")
    moduli_path = os.path.join(args.results_dir, "09_table1_moduli_predictions.csv")   # this run's own intermediate file, read back below
    subprocess.run([sys.executable, "-u",
                   os.path.join(PROJECT_ROOT, "scripts", "04_predict_moduli.py"),
                   "--cif-dir", args.cif_dir,
                   "--k-tag", "K_VRH_full,K_VRH_s1,K_VRH_s2",
                   "--g-tag", "G_VRH_full,G_VRH_s1,G_VRH_s2",
                   "--out", moduli_path],
                  check=True)   # raises CalledProcessError if the child script exits non-zero

    print("\n--- Running the Slack-model physics on those moduli ---")
    kappa_path = os.path.join(args.results_dir, "09_table1_kappa_predictions.csv")
    subprocess.run([sys.executable, "-u",
                   os.path.join(PROJECT_ROOT, "scripts", "07_predict_kappa.py"),
                   "--predictions", moduli_path,
                   "--cif-dir", args.cif_dir,
                   "--out", kappa_path],
                  check=True)

    print(f"\nDone. Next: scripts/10_validate_table1.py compares "
         f"09_table1_kappa_predictions.csv against 09_table1_reference.csv's kappa_exp.")


if __name__ == "__main__":
    main()
