#!/usr/bin/env python3
"""
STEP 15 - Predict bulk and shear moduli for every PINK crystal, with ALIGNN.
================================================================================

    python scripts/15_alignn_predict_moduli.py

WHAT THIS IS FOR
----------------
The ALIGNN counterpart to scripts/04_predict_moduli.py. Phase 2 trained ALIGNN
on the same matbench split the CGCNN ensemble used and found it wins on the
held-out test set (see docs/method.tex Section 9) - but that comparison never
touched the actual PINK pipeline: the 1,213-CIF prediction set that
scripts/04's output feeds into scripts/07's Slack-model physics. This script
is that missing arrow for ALIGNN:

    CIF  ->  ALIGNN  ->  (K, G)  ->  [scripts/16]  ->  kappa_L

WHY THIS REUSES alignn.pretrained RATHER THAN A HAND-BUILT INFERENCE LOOP
--------------------------------------------------------------------------
scripts/12_train_alignn.py already had to work around this installed ALIGNN
version's DGL dependency (see that script's own docstring) by routing through
neighbor_strategy="pure_torch". The library ships its own inference-side
counterpart for exactly this path: alignn.pretrained.load_pure_torch_model()
reads a directory of {config.json, best_model.pt} - precisely the layout
results/alignn_<target>/ already has - and
alignn.torch_graph_builder.build_pure_torch_graph() builds the (g, lg) pair a
pure-torch ALIGNNAtomWisePure model expects. Reusing these rather than
re-deriving graph construction is the same "reuse the physics/library code,
don't reimplement it" rule this project has followed everywhere else (e.g.
07_predict_kappa.py re-deriving Slack physics only because pink_predict.py's
own version can't do array-valued Monte Carlo - not the case here).

WHY ONE GRAPH SERVES BOTH TARGETS
------------------------------------
Both targets were trained with identical COMMON_ARGS (same cutoff, same
max_neighbors, same atom_features - see colab/alignn_gpu_driver.py). A graph
depends only on the structure and those settings, not on which property is
being predicted, so building it once per crystal and running both models
against it halves the graph-construction work. Checked, not assumed: an
assertion below aborts loudly if the two targets' saved configs ever
disagree on cutoff/max_neighbors/atom_features.

WHY PROVENANCE IS MERGED FROM pink_moduli_predictions.csv, NOT RECOMPUTED
------------------------------------------------------------------------------
scripts/11_prepare_alignn_data.py computed the train/val/test split with this
project's own split_indices() - the exact function scripts/04's
build_provenance() already used for the CGCNN ensemble. Same seed, same
function call, so the split is identical by construction, not by
coincidence. Re-deriving provenance a second, ALIGNN-specific way would risk
the two silently drifting apart; merging the already-verified column from the
CGCNN run's own output is simpler and cannot disagree with it.

OUTPUT
------
    results/alignn_moduli_predictions.csv
        material_id, formula, n_sites, provenance,
        K_VRH_pred, G_VRH_pred        (GPa - what scripts/16 needs)
        K_VRH_dft,  G_VRH_dft         (GPa - blank where no reference exists)
        pugh_ratio
    No spread/uncertainty columns: ALIGNN here is a single trained model, not
    a 3-member ensemble like CGCNN's, so there is nothing to compute a spread
    from - scripts/16 skips the Monte Carlo step for exactly this reason,
    rather than fabricating an uncertainty that isn't there.
"""

import argparse
import glob
import os
import sys
import time
import warnings

# torch first - see cgcnn_scratch/data.py for why (MKL/libiomp5 duplicate
# OpenMP runtime segfault if numpy/pandas/pymatgen import first in this env).
import torch

import numpy as np
import pandas as pd
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_alignn_target(model_dir, device):
    """Load one target's pure-torch ALIGNN model + the graph-building
    settings it was trained with, read back from its own config.json rather
    than assumed - same reasoning as scripts/04's load_model() reading
    architecture back from the checkpoint instead of re-specifying it."""
    from alignn.pretrained import load_pure_torch_model

    if not os.path.exists(os.path.join(model_dir, "best_model.pt")):
        sys.exit(f"No best_model.pt in {model_dir} - train it first with "
                 f"scripts/12_train_alignn.py (or fetch it from Kaggle/Colab).")
    model, config = load_pure_torch_model(model_dir, device=device)

    cutoff = float(config.get("cutoff", 8.0))
    max_neighbors = int(config.get("max_neighbors", 12))
    atom_features = config.get("atom_features")
    if not atom_features:
        n_in = config.get("model", {}).get("atom_input_features", 92)
        atom_features = "atomic_number" if int(n_in) == 1 else "cgcnn"
    return model, {"cutoff": cutoff, "max_neighbors": max_neighbors,
                   "atom_features": atom_features}


def predict_pair(k_model, g_model, atoms, graph_settings, device):
    """Build one graph, run both models against it, return (K, G) in GPa.

    Mirrors alignn.pretrained.get_prediction_pure()'s own body, inlined
    rather than called directly: that function reloads the model from disk
    on every call, which would mean re-reading a 15 MB checkpoint 1,213
    times per target - fine for a single prediction, wasteful in a loop.
    """
    from alignn.torch_graph_builder import build_pure_torch_graph

    g, lg = build_pure_torch_graph(
        atoms=atoms, two_body_cutoff=graph_settings["cutoff"],
        max_neighbors=graph_settings["max_neighbors"],
        atom_features=graph_settings["atom_features"],
        compute_line_graph=True, device=device)
    lat = torch.tensor(atoms.lattice_mat).type(torch.get_default_dtype()).to(device)

    with torch.no_grad():
        k_out = k_model([g, lg, lat])
        g_out = g_model([g, lg, lat])
    k_out = k_out["out"] if isinstance(k_out, dict) else k_out
    g_out = g_out["out"] if isinstance(g_out, dict) else g_out
    k_val = float(k_out.detach().cpu().numpy().flatten()[0])
    g_val = float(g_out.detach().cpu().numpy().flatten()[0])
    return k_val, g_val


def build_predictions(cif_dir, k_model, g_model, graph_settings, device):
    """Parse every CIF in cif_dir, predict K and G with ALIGNN.

    Structurally mirrors scripts/04's build_prediction_graphs() +
    predict_ensemble() combined into one pass, since ALIGNN's graph
    construction (unlike CGCNN's) is cheap enough to do inline per crystal
    rather than pre-building a whole dataset object first.
    """
    from jarvis.core.atoms import pmg_to_atoms

    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not paths:
        sys.exit(f"No CIF files found in {cif_dir}")
    print(f"Predicting on {len(paths)} CIFs from {cif_dir}...")

    rows, failed = [], []
    start = time.time()
    for i, path in enumerate(paths):
        material_id = os.path.splitext(os.path.basename(path))[0]
        try:
            structure = Structure.from_file(path)
            atoms = pmg_to_atoms(structure)
            k_pred, g_pred = predict_pair(k_model, g_model, atoms,
                                          graph_settings, device)
        except Exception as exc:
            # Malformed CIF, an element JARVIS's converter rejects, or a
            # structure too small/large for the trained cutoff to see any
            # neighbours - same failure classes scripts/04 and
            # scripts/11_prepare_alignn_data.py already log rather than crash on.
            failed.append((material_id, str(exc)[:70]))
            continue

        rows.append({"material_id": material_id,
                     "formula": structure.composition.reduced_formula,
                     "n_sites": len(structure),
                     # ALIGNN regresses raw GPa directly (unlike CGCNN, which
                     # trains on log10(GPa) - see 02_train.py) - confirmed by
                     # inspecting results/alignn_*/Test_results.json's
                     # target_out values directly, not assumed from the
                     # matbench_log_kvrh dataset name. A tiny positive floor
                     # guards the downstream log10() in slack_physics()
                     # against a rare negative/near-zero regression output.
                     "K_VRH_pred": max(k_pred, 1e-3),
                     "G_VRH_pred": max(g_pred, 1e-3)})

        if (i + 1) % 250 == 0:
            rate = (i + 1) / (time.time() - start)
            print(f"  {i + 1}/{len(paths)}  ({rate:.1f}/s, "
                 f"~{(len(paths) - i - 1) / rate / 60:.1f} min left)")

    print(f"  predicted {len(rows)} crystals in {(time.time() - start) / 60:.1f} min")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")
    return pd.DataFrame(rows), failed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "alignn"))
    parser.add_argument("--cgcnn-predictions",
                        default=os.path.join(PROJECT_ROOT, "results", "cgcnn",
                                             "pink_moduli_predictions.csv"),
                        help="source of the provenance/DFT-reference columns "
                             "(identical split, computed once for CGCNN)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.results_dir,
                                        "alignn_moduli_predictions.csv")

    # scripts/cgcnn/, not this file's own scripts/alignn/ - see 11's identical note.
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
    from importlib import import_module
    _train = import_module("02_train")  # reuse pick_device(), same as scripts/cgcnn/05/12
    device = _train.pick_device(args.device)
    print(f"=== Predicting bulk and shear moduli for the PINK crystal set (ALIGNN) ===")
    print(f"Device: {device}\n")

    k_dir = os.path.join(args.results_dir, "alignn_bulk_modulus_kv")
    g_dir = os.path.join(args.results_dir, "alignn_shear_modulus_gv")
    print(f"Loading {k_dir} ...")
    k_model, k_settings = load_alignn_target(k_dir, device)
    print(f"Loading {g_dir} ...")
    g_model, g_settings = load_alignn_target(g_dir, device)

    # See the module docstring: one graph per crystal serves both models only
    # if they agree on how that graph is built.
    if k_settings != g_settings:
        sys.exit(f"Bulk and shear ALIGNN models were trained with different "
                 f"graph settings ({k_settings} vs {g_settings}) - can't share "
                 f"one graph between them. Predict them separately instead.")
    print(f"Graph settings (shared): {k_settings}\n")

    df, failed = build_predictions(args.cif_dir, k_model, g_model, k_settings, device)

    # --- Provenance + DFT reference, merged from the CGCNN run's own output
    if os.path.exists(args.cgcnn_predictions):
        cgcnn = pd.read_csv(args.cgcnn_predictions)
        df = df.merge(cgcnn[["material_id", "provenance", "K_VRH_dft", "G_VRH_dft"]],
                     on="material_id", how="left")
        df["provenance"] = df["provenance"].fillna("unknown")
    else:
        print(f"\nWARNING: {args.cgcnn_predictions} not found - "
             f"provenance/DFT-reference columns will be absent. Run "
             f"scripts/04_predict_moduli.py first for the full comparison.")
        df["provenance"] = "unknown"
        df["K_VRH_dft"] = np.nan
        df["G_VRH_dft"] = np.nan

    df["pugh_ratio"] = df.G_VRH_pred / df.K_VRH_pred

    columns = ["material_id", "formula", "n_sites", "provenance",
              "K_VRH_pred", "G_VRH_pred", "pugh_ratio", "K_VRH_dft", "G_VRH_dft"]
    df = df[columns].sort_values("material_id").reset_index(drop=True)

    os.makedirs(args.results_dir, exist_ok=True)
    df.round(4).to_csv(out_path, index=False)

    # --- Report ------------------------------------------------------------
    print(f"\nWrote {len(df)} predictions to {out_path}")
    if failed:
        print(f"  ({len(failed)} CIFs could not be predicted and are absent)")

    print("\nProvenance breakdown:")
    print(df.provenance.value_counts().to_string())

    print("\nPredicted moduli (GPa):")
    print(df[["K_VRH_pred", "G_VRH_pred", "pugh_ratio"]]
          .describe().round(2).to_string())

    have_ref = df.dropna(subset=["K_VRH_dft"])
    if len(have_ref):
        print("\nAgreement with DFT where a reference exists "
             "(MAE in log10 GPa):")
        rows = []
        for split_name, sub in have_ref.groupby("provenance"):
            rows.append({
                "provenance": split_name,
                "n": len(sub),
                "K_MAE_log10": np.abs(np.log10(sub.K_VRH_pred) -
                                      np.log10(sub.K_VRH_dft)).mean(),
                "G_MAE_log10": np.abs(np.log10(sub.G_VRH_pred) -
                                      np.log10(sub.G_VRH_dft)).mean(),
            })
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
