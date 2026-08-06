#!/usr/bin/env python3
"""
STEP 4 - Predict bulk and shear moduli for every PINK crystal.
==============================================================

    python scripts/04_predict_moduli.py

WHAT THIS IS FOR
----------------
This is the hand-off from the machine-learning half of the project to the
physics half. PINK's pipeline is:

    CIF  ->  CGCNN  ->  (K, G)  ->  Slack model  ->  lattice thermal conductivity

Steps 1-3 built and trained the CGCNN. This script runs it over all 1,213 of our
CIFs and writes the (K, G) table that the kappa_L stage consumes.

WHY THE PREDICTIONS ARE TRUSTWORTHY HERE AND WERE NOT BEFORE
------------------------------------------------------------
The first version of this project trained on the 278 CIFs that happen to have
matbench labels, then predicted the other 935 - training and predicting on
nearly the same small pool. Now the model is trained on all 10,987 matbench
crystals, and our 1,213 CIFs are purely a prediction set. That is the right way
round, and it is what the PINK paper does.

278 of the 1,213 are still IN the training data, because they are in matbench.
Those predictions are not evidence of anything - the model was fitted on them.
So every row carries a `provenance` column:

    train / val / test   this crystal is in matbench and the model saw it in
                         that split. `test` rows are genuine held-out evidence;
                         `train` rows are recall, not prediction.
    unseen               not in matbench at all. No DFT reference exists, and
                         this is the model doing the job it was built for.

Rows also carry `K_VRH_dft` / `G_VRH_dft` where a DFT reference exists, so the
error on the 278 can be inspected directly instead of taken on trust.

OUTPUT
------
    results/pink_moduli_predictions.csv
        material_id, formula, n_sites, provenance,
        K_VRH_pred, G_VRH_pred        (GPa - what the kappa_L stage needs)
        K_VRH_dft,  G_VRH_dft         (GPa - blank where no reference exists)
        pugh_ratio                    (G/K, a ductility indicator PINK uses)
"""

import argparse
import glob
import os
import sys
import time
import warnings

# torch first - see the note in 02_train.py about the OpenMP clash.
import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pymatgen.core import Structure

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from cgcnn_scratch.data import (AtomFeaturiser, GaussianDistance,  # noqa: E402
                                Normalizer, collate_pool, structure_to_graph)
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402


class PredictionSet(torch.utils.data.Dataset):
    """Graphs built from a directory of CIFs, with no targets.

    Deliberately mirrors CIFData's output shape - (graph, target, id) - so the
    existing collate_pool works unchanged. The target slot is filled with a
    dummy zero that is never read; only the model's output matters here.
    """

    def __init__(self, graphs, ids):
        self.graphs, self.ids = graphs, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.graphs[idx], torch.Tensor([0.0]), self.ids[idx]


def build_prediction_graphs(cif_dir, atom_init_path, max_num_nbr, radius, step):
    """Parse every CIF in cif_dir and featurise it exactly as training did.

    Goes through the same `structure_to_graph` the training cache was built
    with, so the model sees features in the distribution it was fitted on.
    """
    ari = AtomFeaturiser(atom_init_path)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=step)

    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    if not paths:
        sys.exit(f"No CIF files found in {cif_dir}")
    print(f"Featurising {len(paths)} CIFs from {cif_dir}...")

    graphs, ids, meta, failed = [], [], [], []
    start = time.time()

    for i, path in enumerate(paths):
        material_id = os.path.splitext(os.path.basename(path))[0]
        try:
            structure = Structure.from_file(path)
            graph = structure_to_graph(structure, ari, gdf, max_num_nbr, radius)
        except Exception as exc:
            # Malformed CIF, partial occupancies, or an element outside the
            # 1-100 range atom_init.json covers.
            failed.append((material_id, str(exc)[:70]))
            continue

        graphs.append(graph)
        ids.append(material_id)
        meta.append({"material_id": material_id,
                     "formula": structure.composition.reduced_formula,
                     "n_sites": len(structure)})

        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(paths)}  ({(i + 1) / (time.time() - start):.0f}/s)")

    print(f"  featurised {len(graphs)} crystals in {time.time() - start:.0f}s")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")
    return PredictionSet(graphs, ids), pd.DataFrame(meta), failed


def load_model(tag, results_dir):
    """Rebuild a trained model and its normalizer from the saved checkpoint.

    The architecture is read back from the checkpoint's own saved args rather
    than re-specified here, so a model trained with different widths still
    loads correctly instead of failing on a shape mismatch.
    """
    path = os.path.join(results_dir, f"model_{tag}.pth")
    if not os.path.exists(path):
        sys.exit(f"No checkpoint at {path} - train it first with 02_train.py")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]

    model = CrystalGraphConvNet(
        ckpt["feature_lens"]["orig_atom_fea_len"],
        ckpt["feature_lens"]["nbr_fea_len"],
        atom_fea_len=targs["atom_fea_len"], n_conv=targs["n_conv"],
        h_fea_len=targs["h_fea_len"], n_h=targs["n_h"], classification=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    normalizer = Normalizer(torch.zeros(1))
    normalizer.load_state_dict(ckpt["normalizer"])
    return model, normalizer, ckpt


def predict_log(model, normalizer, dataset, batch_size=64):
    """Run one model over the whole set, returning {id: log10(modulus in GPa)}.

    Stays in log space because that is where ensemble members get averaged -
    see the note in 05_ensemble.py about why averaging in GPa is wrong.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_pool)
    out = {}
    with torch.no_grad():
        for inputs, _, ids in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            pred_log = normalizer.denorm(output.data).view(-1).numpy()
            for material_id, value in zip(ids, pred_log):
                out[material_id] = float(value)
    return out


def predict_ensemble(models, dataset):
    """Average several models' log-space predictions, back-transformed to GPa.

    Returns (moduli in GPa, per-crystal member spread). The spread is a free
    uncertainty estimate: where the members disagree, the ensemble is guessing,
    and a crystal with a large spread deserves less trust in the kappa_L stage.
    """
    per_member = [predict_log(model, normalizer, dataset)
                  for model, normalizer in models]

    moduli, spread = {}, {}
    for material_id in per_member[0]:
        values = [m[material_id] for m in per_member]
        moduli[material_id] = float(10 ** np.mean(values))
        spread[material_id] = float(np.std(values))
    return moduli, spread


def build_provenance(data_full_dir, ckpt, labels):
    """Work out, for each of our mp-ids, whether the model was trained on it.

    Two lookups chained together:
        mp_to_mb.csv   our mp-id  ->  matbench id, from the structure matching
        ckpt["split"] + dataset order   matbench id  ->  train / val / test

    The split in the checkpoint is a list of INDICES into the dataset, and the
    dataset's order is the cache order filtered to labelled rows - so we
    reconstruct that same id list here to turn indices back into ids.
    """
    mapping_path = os.path.join(data_full_dir, "mp_to_mb.csv")
    if not os.path.exists(mapping_path):
        print(f"  no {mapping_path}; every row will be marked 'unknown'")
        return {}, {}

    mapping = pd.read_csv(mapping_path)

    # Rebuild the dataset's id ordering. GraphCacheData keeps the cache's order,
    # restricted to ids that have a label - the same filter clean_labels applied.
    blob = torch.load(os.path.join(data_full_dir, "graphs.pt"),
                      weights_only=False)
    labelled = set(labels.mb_id)
    ordered_ids = [i for i in blob["ids"] if i in labelled]

    mb_to_split = {}
    for split_name, indices in ckpt["split"].items():
        for idx in indices:
            mb_to_split[ordered_ids[idx]] = split_name

    mb_to_moduli = labels.set_index("mb_id")[["K_VRH", "G_VRH"]].to_dict("index")

    provenance, reference = {}, {}
    for row in mapping.itertuples(index=False):
        provenance[row.mp_id] = mb_to_split.get(row.mb_id, "unknown")
        if row.mb_id in mb_to_moduli:
            reference[row.mp_id] = mb_to_moduli[row.mb_id]
    return provenance, reference


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"))
    parser.add_argument("--data-full", default=os.path.join(PROJECT_ROOT, "data_full"))
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results"))
    parser.add_argument("--atom-init",
                        default=os.path.join(PROJECT_ROOT, "data", "atom_init.json"))
    parser.add_argument("--k-tag", default="K_VRH_full",
                        help="checkpoint tag(s) for the bulk modulus model. "
                             "Comma-separate several to average them as an "
                             "ensemble.")
    parser.add_argument("--g-tag", default="G_VRH_full",
                        help="checkpoint tag(s) for the shear modulus model, "
                             "comma-separated for an ensemble")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.results_dir,
                                        "pink_moduli_predictions.csv")

    # --- Load both models --------------------------------------------------
    print("=== Predicting bulk and shear moduli for the PINK crystal set ===\n")
    k_tags = [t.strip() for t in args.k_tag.split(",") if t.strip()]
    g_tags = [t.strip() for t in args.g_tag.split(",") if t.strip()]

    k_loaded = [load_model(t, args.results_dir) for t in k_tags]
    g_loaded = [load_model(t, args.results_dir) for t in g_tags]
    k_models = [(m, n) for m, n, _ in k_loaded]
    g_models = [(m, n) for m, n, _ in g_loaded]
    # Provenance and split checks only need one representative checkpoint each;
    # 05_ensemble.py already refuses to combine members with differing splits.
    k_ckpt, g_ckpt = k_loaded[0][2], g_loaded[0][2]

    for tag, ckpt in list(zip(k_tags, [c for _, _, c in k_loaded])) + \
                     list(zip(g_tags, [c for _, _, c in g_loaded])):
        # A mid-run .partial.pth has no test_mae - it is only written once the
        # run reaches the test set - but it is still a perfectly usable model,
        # so fall back to the validation score rather than refusing to predict.
        if "test_mae" in ckpt:
            print(f"Loaded {tag} (test MAE {ckpt['test_mae']:.4f} log10 GPa)")
        else:
            print(f"Loaded {tag} (INCOMPLETE run, epoch {ckpt.get('epoch', '?')}, "
                  f"val MAE {ckpt['best_val_mae']:.4f} log10 GPa)")
    print()

    # Featurisation settings must match what the models were trained on. The
    # training cache recorded them, so read them back rather than assuming.
    cache_config = torch.load(os.path.join(args.data_full, "graphs.pt"),
                              weights_only=False)["config"]
    print(f"Featurisation from training cache: {cache_config}")

    dataset, meta, failed = build_prediction_graphs(
        args.cif_dir, args.atom_init, cache_config["max_num_nbr"],
        cache_config["radius"], cache_config["step"])

    # --- Predict -----------------------------------------------------------
    print(f"\nRunning the bulk modulus model ({len(k_models)} member(s))...")
    k_pred, k_spread = predict_ensemble(k_models, dataset)
    print(f"Running the shear modulus model ({len(g_models)} member(s))...")
    g_pred, g_spread = predict_ensemble(g_models, dataset)

    # --- Assemble the table ------------------------------------------------
    df = meta.copy()
    df["K_VRH_pred"] = df.material_id.map(k_pred)
    df["G_VRH_pred"] = df.material_id.map(g_pred)
    # Only meaningful with 2+ members; with one it is identically zero.
    if len(k_models) > 1:
        df["K_VRH_spread_log10"] = df.material_id.map(k_spread)
    if len(g_models) > 1:
        df["G_VRH_spread_log10"] = df.material_id.map(g_spread)

    labels = pd.read_csv(os.path.join(args.data_full, "labels.csv"))

    # The provenance column assumes both models were split the same way, which
    # they are: split_indices is seeded and both targets have the same row
    # count. Check it rather than trust it - if they ever diverge, a crystal
    # could be test for K and train for G, and one `provenance` column would be
    # quietly lying about one of them.
    if k_ckpt["split"]["test"] != g_ckpt["split"]["test"]:
        print("\nWARNING: the two models used different splits. The provenance "
              "column describes the BULK model only.")
    provenance, reference = build_provenance(args.data_full, k_ckpt, labels)

    df["provenance"] = df.material_id.map(provenance).fillna("unseen")
    df["K_VRH_dft"] = df.material_id.map(
        lambda m: reference.get(m, {}).get("K_VRH", np.nan))
    df["G_VRH_dft"] = df.material_id.map(
        lambda m: reference.get(m, {}).get("G_VRH", np.nan))

    # Pugh's ratio G/K: below ~0.57 a material is ductile, above it brittle.
    # It costs nothing to carry and the kappa_L discussion uses it.
    df["pugh_ratio"] = df.G_VRH_pred / df.K_VRH_pred

    columns = ["material_id", "formula", "n_sites", "provenance",
               "K_VRH_pred", "G_VRH_pred", "pugh_ratio",
               "K_VRH_dft", "G_VRH_dft"]
    columns += [c for c in ("K_VRH_spread_log10", "G_VRH_spread_log10")
                if c in df.columns]
    df = df[columns]
    df = df.sort_values("material_id").reset_index(drop=True)

    os.makedirs(args.results_dir, exist_ok=True)
    df.round(4).to_csv(out_path, index=False)

    # --- Report ------------------------------------------------------------
    print(f"\nWrote {len(df)} predictions to {out_path}")
    if failed:
        print(f"  ({len(failed)} CIFs could not be featurised and are absent)")

    print("\nProvenance breakdown:")
    print(df.provenance.value_counts().to_string())

    print("\nPredicted moduli (GPa):")
    print(df[["K_VRH_pred", "G_VRH_pred", "pugh_ratio"]]
          .describe().round(2).to_string())

    # Where a DFT reference exists, show the error - split by provenance, since
    # a low error on `train` rows is memorisation and a low error on `test` rows
    # is the number that actually means something.
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
