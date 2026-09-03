#!/usr/bin/env python3
"""
STEP 36 - Build the AFLOW gamma dataset for the third head.
============================================================

    python scripts/cgcnn/36_prepare_gamma_dataset.py

WHY A SEPARATE DATASET AND NOT AN EXTRA COLUMN ON MATBENCH
-----------------------------------------------------------
Because matbench cannot evaluate a gamma head, and using it would be circular.

The pipeline's "true kappa" for a matbench crystal is not a measured quantity.
It is Slack(true K, true G) with gamma DERIVED from true K/G through the
empirical Poisson relation. So a model that predicts a different gamma - even
a better one - scores WORSE against that target by construction. The target
already assumes the derivation is correct.

AFLOW breaks the circle. It supplies, per crystal, all four of:

    ael_bulk_modulus_vrh            K
    ael_shear_modulus_vrh           G
    agl_gruneisen                   gamma, tabulated, not derived
    agl_thermal_conductivity_300K   kappa, computed independently

so gamma can be learned against a real label and scored against a real kappa.

THE HEADROOM, MEASURED BEFORE BUILDING ANY OF THIS
---------------------------------------------------
Derived gamma is a poor stand-in for the tabulated one:

    MAE 0.4326, correlation only 0.4205

and substituting the real value into the Slack formula, holding everything
else fixed, improves agreement with AFLOW's own kappa by 25%:

    kappa from DERIVED gamma   MAE log10 0.2395   r 0.9118
    kappa from REAL    gamma   MAE log10 0.1806   r 0.9125

That 0.1806 is the ceiling a perfect gamma head could reach here. It is the
number any result from 37_train_gamma.py must be read against - beating
0.2395 is the bar, and 0.1806 is the limit.

WHAT THIS WRITES
----------------
    data_full/gamma_graphs.pt    graphs, same featuriser as the main cache
    data_full/gamma_labels.csv   K, G, gamma, kappa, and the V and N that the
                                 full Slack formula needs

V and N matter here in a way they did not before. Every earlier script could
drop density and cell volume because they cancel in a predicted/true ratio.
Scoring against AFLOW's ABSOLUTE kappa means they no longer cancel, so the
volume per cell is computed from density and mean atomic mass and carried in
the labels file.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    "agl_csv": "data_full/aflow_agl.csv",             # AFLOW AGL metadata table, written by 33_fetch_aflow_gamma.py
    "struct_dir": "data_full/aflow_structures",        # directory of per-crystal POSCAR files, written by 34_fetch_aflow_structures.py
    "out_cache": "data_full/gamma_graphs.pt",          # this script's own output: serialized crystal graphs
    "out_labels": "data_full/gamma_labels.csv",        # this script's own output: per-crystal numeric labels

    # Featuriser settings. Read back from the main graph cache when possible so
    # gamma-head graphs are built identically to every other graph in the
    # project; these are only the fallback.
    "max_num_nbr": 12,     # max neighbours per atom in the graph, if the main cache's own setting cannot be read
    "radius": 8.0,         # neighbour-search cutoff radius in Angstrom, same fallback role

    # Skip cells larger than this - the periodic neighbour search is the
    # slowest step and a few huge cells would dominate the runtime.
    "max_atoms": 60,       # crystals with more atoms than this are dropped before featurisation

    "limit": 0,   # 0 = everything; small number smoke-tests the path
}
# =============================================================================

import argparse                                    # CLI flag parsing, auto-generated from CONFIG below
import os                                           # path joining/existence checks
import sys                                          # sys.path manipulation to import the project package
import warnings                                     # used to silence noisy third-party warnings

# torch before numpy/pandas/pymatgen or MKL's duplicate libiomp5 aborts.
import torch                                        # graph cache serialization (torch.save/torch.load)
import numpy as np                                  # numerical support (used indirectly via pandas/torch)
import pandas as pd                                 # AGL CSV loading, filtering, and label-table writing

warnings.filterwarnings("ignore")                   # suppress warning noise (e.g. from pymatgen) in the printed log

# __file__ is this script's own path; three dirname() calls climb
# scripts/cgcnn/ -> scripts/ -> the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)                    # make the project root importable so `cgcnn_scratch` resolves below

from cgcnn_scratch.data import (  # noqa: E402
    AtomFeaturiser, GaussianDistance, structure_to_graph)   # shared featurisation helpers, same ones the main dataset uses

AMU_KG = 1.66053906660e-27                          # atomic mass unit in kilograms, for converting atomic mass to SI mass


def parse_overrides():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, default in CONFIG.items():             # auto-generate one --flag per CONFIG entry
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default), default=None,
                            help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()                      # parse sys.argv against the flags just registered
    cfg = dict(CONFIG)                              # start from a copy of the defaults
    for key, value in vars(args).items():           # vars(args) turns the Namespace into a plain dict
        if value is not None:                       # None means "not passed on the CLI" (argparse default), so keep CONFIG's value
            cfg[key] = value                        # otherwise the CLI value overrides the default
    return cfg                                      # the effective, merged configuration for this run


def main():
    cfg = parse_overrides()                         # resolve CONFIG + any CLI overrides
    from pymatgen.core import Composition, Structure    # imported here (not at module scope) so --help works without pymatgen installed

    agl_path = os.path.join(PROJECT_ROOT, cfg["agl_csv"])          # absolute path to the AGL metadata CSV
    struct_dir = os.path.join(PROJECT_ROOT, cfg["struct_dir"])     # absolute path to the POSCAR structures directory
    if not os.path.exists(agl_path):                # fail fast with a clear message rather than a confusing pandas error
        raise SystemExit(f"missing {agl_path} - run 33_fetch_aflow_gamma.py")

    df = pd.read_csv(agl_path)                      # load every AGL row (may include non-numeric/placeholder values)
    numeric = ["agl_gruneisen", "agl_thermal_conductivity_300K",
               "ael_bulk_modulus_vrh", "ael_shear_modulus_vrh",
               "natoms", "density"]
    for col in numeric:                             # coerce each of these columns to numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")   # non-numeric entries become NaN instead of raising

    before = len(df)                                # row count before any filtering, for the printed before/after summary
    df = df.dropna(subset=numeric)                  # drop rows missing any of the required numeric fields
    # Every one of these must be positive: the targets are logs of the first
    # four, and density and atom count divide into the cell volume.
    for col in numeric:
        df = df[df[col] > 0]                        # keep only strictly-positive values for every required column
    df = df[df.natoms <= cfg["max_atoms"]]          # drop cells above the configured atom-count ceiling
    print(f"  {before} AGL entries -> {len(df)} with all four labels positive "
          f"and <= {cfg['max_atoms']} atoms")

    # Only entries whose structure was actually downloaded. The fetch is
    # incomplete (AFLOW started returning 503), so this is the real limit.
    df["key"] = df.auid.str.replace(":", "_", regex=False)   # build the same filesystem-safe key used for POSCAR filenames
    available = {os.path.basename(p)[:-len(".poscar")]        # strip directory and ".poscar" suffix from each filename
                 for p in os.listdir(struct_dir) if p.endswith(".poscar")}   # set of keys that actually have a downloaded structure
    df = df[df.key.isin(available)]                 # keep only rows whose structure file exists on disk
    print(f"  {len(df)} of those have a downloaded structure")
    if cfg["limit"]:                                # 0 is falsy, so this only triggers when a smoke-test limit was set
        df = df.head(cfg["limit"])                  # truncate to the first N rows for a quick test run

    # Reuse the main cache's featuriser settings so these graphs are built
    # identically to every other graph in the project.
    main_cache = os.path.join(PROJECT_ROOT, "data_full", "graphs.pt")   # path to the matbench graph cache built by 01b
    conf = {}                                        # fallback empty config if the main cache is not present
    if os.path.exists(main_cache):
        conf = torch.load(main_cache, weights_only=False).get("config", {}) or {}
        # ^ load only the cache's small "config" dict, not the (much larger) graph list, then fall back to {} if it is missing/empty
    max_num_nbr = int(conf.get("max_num_nbr", cfg["max_num_nbr"]))   # prefer the main cache's setting, else this script's own default
    radius = float(conf.get("radius", cfg["radius"]))                 # same fallback pattern for the cutoff radius
    print(f"  featuriser: max_num_nbr={max_num_nbr}, radius={radius}")

    ari = AtomFeaturiser(os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"))   # per-element feature vector lookup
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)    # Gaussian radial-basis expansion for bond distances

    ids, graphs, rows, failed = [], [], [], 0        # accumulators: crystal ids, graph objects, label rows, and a failure counter
    for i, row in enumerate(df.itertuples(), 1):     # itertuples() is a fast row iterator; enumerate from 1 for human-readable progress counts
        path = os.path.join(struct_dir, f"{row.key}.poscar")   # path to this row's POSCAR file
        try:
            # from_str, not from_file: pymatgen dispatches on the filename
            # extension and rejects ".poscar".
            with open(path) as fh:
                structure = Structure.from_str(fh.read(), fmt="poscar")   # parse the POSCAR text into a pymatgen Structure
            graph = structure_to_graph(structure, ari, gdf,
                                       max_num_nbr=max_num_nbr, radius=radius)   # convert the structure into a CGCNN graph
            mean_amu = Composition(row.compound).weight / Composition(row.compound).num_atoms
            # ^ average atomic mass per atom, in atomic mass units, from the chemical formula
        except Exception as exc:                     # any parsing/featurisation failure for this one crystal
            failed += 1                               # count it and continue rather than aborting the whole run
            if failed <= 3:                           # only print the first few failures, to avoid flooding the log
                print(f"    {row.key}: {type(exc).__name__}: {exc}")
            continue                                  # skip to the next row

        # Cell volume in m^3, from density and total mass. Needed because
        # scoring against an ABSOLUTE kappa means V and N no longer cancel.
        volume_m3 = (row.natoms * mean_amu * AMU_KG) / (row.density * 1e3)
        # ^ total mass (atoms * amu-per-atom, converted to kg) divided by density (g/cm3 converted to kg/m3 via *1e3)

        cid = "ag-" + row.auid.split(":")[-1][:12]   # short, readable crystal id derived from AFLOW's auid
        ids.append(cid)                               # record this crystal's id
        graphs.append(graph)                          # record its graph
        rows.append({                                 # record its label row
            "gid": cid,
            "formula": row.compound,
            "n_sites": int(row.natoms),
            "K_VRH": float(row.ael_bulk_modulus_vrh),      # GPa
            "G_VRH": float(row.ael_shear_modulus_vrh),     # GPa
            "gamma": float(row.agl_gruneisen),             # dimensionless
            "kappa_agl": float(row.agl_thermal_conductivity_300K),  # W/m/K
            "density_g_cm3": float(row.density),
            "volume_m3": volume_m3,
        })
        if i % 200 == 0 or i == len(df):              # print a progress line every 200 crystals, and on the final one
            print(f"  {i:5d}/{len(df)}  built {len(ids):5d}  failed {failed:4d}",
                  flush=True)                          # flush so progress is visible immediately, not buffered

    if not ids:                                       # nothing survived featurisation - nothing useful to save
        raise SystemExit("no graphs built")

    torch.save({"ids": ids, "graphs": graphs,          # serialize ids, graphs, and the featuriser config used to build them
                "config": {"max_num_nbr": max_num_nbr, "radius": radius,
                           "source": "aflow_agl"}},
               os.path.join(PROJECT_ROOT, cfg["out_cache"]))
    out = pd.DataFrame(rows)                           # turn the accumulated label rows into a DataFrame
    out.to_csv(os.path.join(PROJECT_ROOT, cfg["out_labels"]), index=False)   # write the labels CSV, no row-index column

    print()
    print("=" * 74)
    print(f"  {len(out)} crystals -> {cfg['out_cache']} / {cfg['out_labels']}")
    print("=" * 74)
    for col, unit in [("K_VRH", "GPa"), ("G_VRH", "GPa"),
                      ("gamma", ""), ("kappa_agl", "W/m/K")]:
        s = out[col]                                    # this column's Series, for the summary stats below
        print(f"  {col:<12} median {s.median():9.3f} {unit:<7} "
              f"range {s.min():.3g} to {s.max():.3g}")
    print(f"\n  low-kappa coverage: {(out.kappa_agl < 1).sum()} below 1 W/m/K, "
          f"{(out.kappa_agl < 2).sum()} below 2")           # counts of crystals below two low-kappa thresholds


if __name__ == "__main__":                              # only run main() when this file is executed directly, not when imported
    main()
