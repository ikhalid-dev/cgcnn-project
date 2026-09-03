#!/usr/bin/env python3
"""
STEP 13 - Screen the GNoME discovery database for low-kappa_L candidates.
================================================================================

    python scripts/13_screen_gnome.py

    # quick smoke test on a small subset first:
    python scripts/13_screen_gnome.py --limit 300

WHAT THIS IS
-------------
The paper's headline result is not the 1,213-crystal comparison this project
has run everywhere else - it's a 377,221-material high-throughput screen of
Google DeepMind's GNoME discovery database, filtered down to candidates with
kappa_L <= 1 W/m/K. This project has never run that screen. This script does:

    GNoME summary CSV --filter (bandgap, stability, no radioactive elements)-->
    candidate structures --(this project's CGCNN ensemble)--> K, G -->
    (Slack model, Monte Carlo uncertainty)--> kappa_L --(threshold <= 1)-->
    our own candidate list --compare against--> the paper's own published
    11,869 candidates (external/AI4Kappa/.../Nature-filtered-low-kappa.csv)

THE SNAPSHOT-DRIFT CAVEAT, STATED ONCE HERE, WORTH REPEATING WHEREVER THIS
RESULT IS QUOTED
--------------------------------------------------------------------------
The GNoME bucket is not a frozen 2023 snapshot - it has grown. This project's
download (2026-08-08) has 554,054 stable materials; the paper's screen used
377,221. There is no way to pull the exact historical snapshot the paper
screened, only today's. So this is "a current GNoME screen using the paper's
stated criteria," not a bit-identical replay - the funnel counts below are
sanity-checked against the paper's own (same order of magnitude, given the
~1.47x larger corpus), never claimed to match exactly.

WHERE THE FILES COME FROM
---------------------------
Two files, fetched once via plain HTTPS from the public GCS bucket (no
gcloud/gsutil needed - anonymously readable):

    https://storage.googleapis.com/gdm_materials_discovery/gnome_data/stable_materials_summary.csv
        (~144 MB) - one row per material: MaterialId, Composition, Reduced
        Formula, Bandgap, Decomposition Energy Per Atom, Elements, etc.
    https://storage.googleapis.com/gdm_materials_discovery/gnome_data/by_composition.zip
        (~450 MB) - one CIF per material. Three redundant zips exist
        (by_composition/by_id/by_reduced_formula, all indexing the same
        554,054 structures); only one is needed.

Both land in gnome_data/ (gitignored - large, and trivially re-downloadable).

HOW STRUCTURES ARE MATCHED TO THE ZIP (CHECKED, NOT ASSUMED)
-----------------------------------------------------------------
by_composition.zip has no MaterialId anywhere in it - entries are named
by_composition/{Composition}.CIF (e.g. "Br2Fe4Ho1Tm3.CIF" - the FULL
composition, not the reduced formula). Verified directly: the zip has exactly
554,054 CIF entries, the summary CSV has exactly 554,054 rows with 554,054
UNIQUE Composition values, and every single zip filename matches a
Composition value in the CSV. So Composition is used as the join key here,
not an assumption that formulas are unique (they are not - Reduced Formula
has a handful of collisions in the paper's own candidate list) but a checked
fact about this specific column.

WHY 33k STRUCTURES ARE READ DIRECTLY FROM THE ZIP, NEVER WRITTEN TO DISK
----------------------------------------------------------------------------
Same reasoning as 01b_prepare_full_dataset.py's graph caching: writing tens
of thousands of individual CIFs to disk just to read them back immediately
would be pure overhead. zipfile.read() hands back the CIF text directly;
pymatgen parses it from a string exactly the same way it would from a file.

REUSE, NOT REIMPLEMENTATION
------------------------------
Every piece of this that already exists elsewhere in this project is
imported, not rewritten: PredictionSet/load_model/predict_ensemble from
04_predict_moduli.py (via importlib - same workaround 05_ensemble.py already
uses for numeric-prefixed filenames), slack_physics/monte_carlo_kappa from
07_predict_kappa.py. Only the GNoME-specific filtering and zip-reading is new
code.
"""

import argparse          # CLI flag parsing
import os                 # path joining/creation, path existence checks
import sys                 # sys.path mutation and sys.exit() on fatal errors
import time                 # wall-clock timing for progress/ETA printouts
import warnings              # suppresses noisy-but-harmless parser warnings
import zipfile                 # reads CIF entries out of by_composition.zip without extracting to disk
from importlib import import_module   # loads a numeric-prefixed module (e.g. "04_predict_moduli") by string name

# torch first - see cgcnn_scratch/data.py for why.
import torch  # noqa: F401

import numpy as np                          # not used directly below but kept for parity with sibling scripts' import block
import pandas as pd                          # DataFrame construction, CSV I/O
from pymatgen.core import Structure          # parses CIF text into a Structure object

warnings.filterwarnings("ignore")   # silences all Python warnings for the rest of this process

# walks up three directories from this file (scripts/cgcnn/ -> scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of the caller's cwd

from cgcnn_scratch.data import AtomFeaturiser, GaussianDistance, structure_to_graph  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # makes this script's own directory importable by module name
_predict = import_module("04_predict_moduli")   # module object for 04_predict_moduli.py (can't `import 04_...` - starts with a digit)
_kappa = import_module("07_predict_kappa")      # module object for 07_predict_kappa.py

# Radioactive / synthetic elements, matching the paper's stated screening
# criterion ("removing radioactive elements"). Tc and Pm are the two lighter
# exceptions; everything from Po (84) onward has no stable isotope.
RADIOACTIVE = {
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def filter_candidates(summary_path, bandgap_lo, bandgap_hi):
    """The paper's own three-stage funnel: bandgap window, thermodynamic
    stability, no radioactive elements. Applied to whatever snapshot is on
    disk - see the module docstring for why this won't exactly match the
    paper's own 377,221 -> 30,199 -> 26,305 funnel, only its order of
    magnitude.
    """
    import ast   # ast.literal_eval turns the CSV's stringified list column back into a real Python list

    print(f"Loading {summary_path}...")
    df = pd.read_csv(summary_path)   # the full GNoME summary table, one row per material
    print(f"  {len(df)} materials in this GNoME snapshot "
         f"(paper's own screen used 377,221 - see module docstring)")

    bandgap_ok = df["Bandgap"].between(bandgap_lo, bandgap_hi)          # boolean mask: bandgap inside the window
    stable = df["Decomposition Energy Per Atom"] <= 0                   # boolean mask: thermodynamically stable or better
    step1 = df[bandgap_ok & stable]                                     # rows passing both masks
    print(f"  after bandgap [{bandgap_lo},{bandgap_hi}] eV + decomposition "
         f"energy <= 0: {len(step1)} (paper: 30,199)")

    elements = step1["Elements"].apply(ast.literal_eval)   # parse each row's element-list string into an actual list
    has_radioactive = elements.apply(lambda els: bool(RADIOACTIVE.intersection(els)))   # True if any element is in RADIOACTIVE
    step2 = step1[~has_radioactive]   # keep only rows with no radioactive element
    print(f"  after removing radioactive elements: {len(step2)} (paper: 26,305)")

    # itertuples() mangles column names containing spaces into positional
    # attributes (_4, _5, ...) - rename up front so downstream row.X access
    # is an explicit, correct name rather than a guessed index.
    return step2.reset_index(drop=True).rename(
        columns={"Reduced Formula": "ReducedFormula"})


def extract_and_featurise(filtered, zip_path, atom_init_path, max_num_nbr, radius, step):
    """Read each candidate's CIF straight out of the zip (never written to
    disk), build its CGCNN graph, and separately compute the structural
    quantities (primitive-cell volume/density/mass) slack_physics() needs -
    the same split responsibility as 04_predict_moduli.py (graph, for the
    model) and 07_predict_kappa.py's structure_quantities() (primitive cell,
    for the physics), just fed from one in-memory Structure instead of two
    separate file reads.
    """
    ari = AtomFeaturiser(atom_init_path)                    # loads the 92-dim per-element feature table
    gdf = GaussianDistance(dmin=0, dmax=radius, step=step)  # builds the Gaussian bond-distance expansion basis
    zf = zipfile.ZipFile(zip_path)                          # opens the zip archive for random-access reads (no extraction)

    graphs, ids, meta, failed = [], [], [], []   # accumulators filled by the loop below
    start = time.time()    # wall-clock start, used for the rate/ETA printout
    n = len(filtered)       # total candidate count, used for progress printouts

    for i, row in enumerate(filtered.itertuples()):   # iterate rows as lightweight namedtuples
        entry = f"by_composition/{row.Composition}.CIF"   # the zip member name for this candidate
        try:
            cif_text = zf.read(entry).decode("utf-8")               # raw CIF bytes -> text, still only in memory
            structure = Structure.from_str(cif_text, fmt="cif")     # parse the CIF text into a Structure
            graph = structure_to_graph(structure, ari, gdf, max_num_nbr, radius)   # featurise into CGCNN graph tensors
            primitive = structure.get_primitive_structure()          # smallest repeating cell, for volume/density below
        except Exception as exc:
            failed.append((row.MaterialId, str(exc)[:70]))   # record id + truncated error, then skip this candidate
            continue

        graphs.append(graph)         # keep in the same order as `ids`
        ids.append(row.MaterialId)   # GNoME's own material identifier
        meta.append({
            "material_id": row.MaterialId,
            "formula": row.ReducedFormula,
            "Number of Atoms": len(primitive),                          # atom count in the primitive cell
            "Volume (A3)": primitive.volume,                            # primitive-cell volume in cubic angstrom
            "Density (g cm-3)": primitive.density,                      # mass / volume, in g/cm^3
            "Atomic mass (amu)": float(primitive.composition.weight),   # total formula-unit mass in atomic mass units
        })

        if (i + 1) % 2000 == 0:   # print progress every 2000 candidates, not every one
            rate = (i + 1) / (time.time() - start)   # candidates processed per second so far
            print(f"  {i + 1:6d}/{n} featurised ({rate:.0f}/s, "
                 f"~{(n - i - 1) / rate / 60:.1f} min left)")

    print(f"Featurised {len(graphs)}/{n} in {(time.time() - start) / 60:.1f} min")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")   # show only the first 3 as a sample
    return _predict.PredictionSet(graphs, ids), pd.DataFrame(meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnome-dir", default=os.path.join(PROJECT_ROOT, "gnome_data"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))
    parser.add_argument("--bandgap-lo", type=float, default=0.1)
    parser.add_argument("--bandgap-hi", type=float, default=3.0)
    parser.add_argument("--kappa-threshold", type=float, default=1.0)
    parser.add_argument("--mc-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-tags", default="K_VRH_full,K_VRH_s1,K_VRH_s2")
    parser.add_argument("--g-tags", default="G_VRH_full,G_VRH_s1,G_VRH_s2")
    parser.add_argument("--limit", type=int, default=None,
                        help="only screen the first N filtered candidates (smoke test)")
    args = parser.parse_args()   # reads sys.argv, returns a Namespace with the fields above

    print("=== Phase 3: screening GNoME for low-kappa_L candidates ===\n")

    summary_path = os.path.join(args.gnome_dir, "stable_materials_summary.csv")
    zip_path = os.path.join(args.gnome_dir, "by_composition.zip")
    if not (os.path.exists(summary_path) and os.path.exists(zip_path)):   # both inputs must already be downloaded
        sys.exit(f"Missing GNoME files in {args.gnome_dir} - fetch both:\n"
                 f"  curl -o {summary_path} https://storage.googleapis.com/"
                 f"gdm_materials_discovery/gnome_data/stable_materials_summary.csv\n"
                 f"  curl -o {zip_path} https://storage.googleapis.com/"
                 f"gdm_materials_discovery/gnome_data/by_composition.zip")

    filtered = filter_candidates(summary_path, args.bandgap_lo, args.bandgap_hi)   # the funnel output DataFrame
    if args.limit:
        filtered = filtered.head(args.limit)   # keep only the first N rows
        print(f"  --limit {args.limit}: smoke-testing on the first "
             f"{len(filtered)} candidates only")

    print("\nFeaturising candidate structures (reading CIFs directly from the "
         "zip, no disk round-trip)...")
    dataset, meta = extract_and_featurise(
        filtered, zip_path,
        atom_init_path=os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"),
        max_num_nbr=12, radius=8, step=0.2)

    print("\nRunning the CGCNN ensemble (bulk and shear modulus)...")
    # loads each named checkpoint and keeps only (model, normalizer) - [:2] drops any extra return values
    k_models = [_predict.load_model(t, args.results_dir)[:2] for t in args.k_tags.split(",")]
    g_models = [_predict.load_model(t, args.results_dir)[:2] for t in args.g_tags.split(",")]
    k_moduli, k_spread = _predict.predict_ensemble(k_models, dataset)   # per-material mean prediction + ensemble spread
    g_moduli, g_spread = _predict.predict_ensemble(g_models, dataset)

    meta["K_VRH_pred"] = meta.material_id.map(k_moduli)                 # join predictions back onto the metadata table by id
    meta["G_VRH_pred"] = meta.material_id.map(g_moduli)
    meta["K_VRH_spread_log10"] = meta.material_id.map(k_spread)
    meta["G_VRH_spread_log10"] = meta.material_id.map(g_spread)

    print("\nRunning the Slack-model physics + Monte Carlo uncertainty...")
    point = _kappa.slack_physics(
        meta["K_VRH_pred"].values, meta["G_VRH_pred"].values,
        meta["Volume (A3)"].values, meta["Density (g cm-3)"].values,
        meta["Atomic mass (amu)"].values, meta["Number of Atoms"].values)   # point-estimate physics: one kappa per material
    meta["Poisson ratio"] = point["poisson"]
    meta["Gruneisen parameter"] = point["gruneisen"]
    meta["Kappa_cal (W m-1 K-1)"] = point["kappa_cal"]

    mc = _kappa.monte_carlo_kappa(meta, n_samples=args.mc_samples, seed=args.seed)   # resamples predictions to get a kappa distribution per material
    meta["Kappa_cal_p05"] = mc["Kappa_cal_p05"]   # 5th percentile of the Monte Carlo kappa distribution
    meta["Kappa_cal_p50"] = mc["Kappa_cal_p50"]   # median
    meta["Kappa_cal_p95"] = mc["Kappa_cal_p95"]   # 95th percentile (the pessimistic bound used for ranking)

    # 13_ prefix: this script's own number, so this result can be told apart
    # from any other script's output sitting in the same results/ directory.
    all_path = os.path.join(args.results_dir, "13_gnome_screen_all.csv")
    meta.to_csv(all_path, index=False)
    print(f"\nWrote {all_path} ({len(meta)} scored candidates, pre-threshold)")

    candidates = meta[meta["Kappa_cal (W m-1 K-1)"] <= args.kappa_threshold].copy()   # rows clearing the kappa threshold
    cand_path = os.path.join(args.results_dir, "13_gnome_screen_candidates.csv")
    candidates.to_csv(cand_path, index=False)
    print(f"kappa_L <= {args.kappa_threshold} W/m/K: {len(candidates)} candidates "
         f"(paper: 11,869, on their smaller/older snapshot)")
    print(f"Wrote {cand_path}")


if __name__ == "__main__":
    main()
