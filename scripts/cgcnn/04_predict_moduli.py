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

WHY EVERY ROW CARRIES ITS PROVENANCE
------------------------------------
The models are trained on all 10,987 matbench crystals, and these 1,213 CIFs
are purely a prediction set - which is the right way round, and what the PINK
paper does.

But 278 of the 1,213 ARE in matbench, so the model was fitted on them. A
prediction for one of those is recall, not generalisation, and quoting it as
evidence would be circular. So every row carries a `provenance` column:

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

import argparse  # CLI flag parsing for --cif-dir/--k-tag/--g-tag/etc
import glob  # filesystem pattern matching, used to list *.cif files
import os  # path joining and filesystem existence checks
import sys  # sys.exit(...) and sys.path manipulation below
import time  # wall-clock timing for the featurisation progress print
import warnings  # used just below to silence noisy library warnings

# torch first - see the note in 02_train.py about the OpenMP clash.
import torch  # tensors, checkpoint loading, no_grad inference
from torch.utils.data import DataLoader  # batches the PredictionSet for the model

import numpy as np  # mean/std for ensemble averaging, log10 for MAE reporting
import pandas as pd  # DataFrame assembly and CSV I/O for the output table
from pymatgen.core import Structure  # parses each CIF file into a Structure object

warnings.filterwarnings("ignore")  # suppresses pymatgen/torch deprecation noise

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, 3 levels above this file
sys.path.insert(0, PROJECT_ROOT)  # so `import cgcnn_scratch` below resolves regardless of cwd

# imports the atom/bond featurisers, the target normalizer, the DataLoader
# collate function, and the CIF -> graph converter used at training time
from cgcnn_scratch.data import (AtomFeaturiser, GaussianDistance,  # noqa: E402
                                Normalizer, collate_pool, structure_to_graph)
from cgcnn_scratch.model import CrystalGraphConvNet  # noqa: E402 -- the CGCNN architecture class


class PredictionSet(torch.utils.data.Dataset):  # subclasses torch's Dataset so DataLoader can batch it
    """Graphs built from a directory of CIFs, with no targets.

    Deliberately mirrors GraphCacheData's output shape - (graph, target, id) -
    so the existing collate_pool works unchanged. The target slot is filled
    with a dummy zero that is never read; only the model's output matters.
    """

    def __init__(self, graphs, ids):
        self.graphs, self.ids = graphs, ids  # store the pre-built graph list and its parallel id list

    def __len__(self):
        return len(self.ids)  # dataset size = number of successfully featurised crystals

    def __getitem__(self, idx):
        return self.graphs[idx], torch.Tensor([0.0]), self.ids[idx]  # (graph, dummy target, id) tuple collate_pool expects


def build_prediction_graphs(cif_dir, atom_init_path, max_num_nbr, radius, step):
    """Parse every CIF in cif_dir and featurise it exactly as training did.

    Goes through the same `structure_to_graph` the training cache was built
    with, so the model sees features in the distribution it was fitted on.
    """
    ari = AtomFeaturiser(atom_init_path)  # builds the per-element feature vector lookup
    gdf = GaussianDistance(dmin=0, dmax=radius, step=step)  # Gaussian basis expansion for bond distances

    paths = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))  # every .cif path in cif_dir, sorted for determinism
    if not paths:
        sys.exit(f"No CIF files found in {cif_dir}")  # abort with a message instead of silently producing nothing
    print(f"Featurising {len(paths)} CIFs from {cif_dir}...")

    graphs, ids, meta, failed = [], [], [], []  # parallel accumulators filled in by the loop below
    start = time.time()  # wall-clock start, for the rate printed every 250 crystals

    for i, path in enumerate(paths):  # i = 0-based index, path = full filesystem path to one CIF
        material_id = os.path.splitext(os.path.basename(path))[0]  # filename without directory or .cif extension
        try:
            structure = Structure.from_file(path)  # parse the CIF into a pymatgen Structure
            graph = structure_to_graph(structure, ari, gdf, max_num_nbr, radius)  # featurise it into the CGCNN graph format
        except Exception as exc:
            # Malformed CIF, partial occupancies, or an element outside the
            # 1-100 range atom_init.json covers.
            failed.append((material_id, str(exc)[:70]))  # record the id and a truncated error message
            continue  # skip this CIF, move on to the next path

        graphs.append(graph)  # keep the successfully built graph
        ids.append(material_id)  # keep its id, in the same order as graphs
        meta.append({"material_id": material_id,
                     "formula": structure.composition.reduced_formula,  # reduced chemical formula, e.g. "SiO2"
                     "n_sites": len(structure)})  # atom count in the unit cell

        if (i + 1) % 250 == 0:  # progress print every 250 crystals processed
            print(f"  {i + 1}/{len(paths)}  ({(i + 1) / (time.time() - start):.0f}/s)")

    print(f"  featurised {len(graphs)} crystals in {time.time() - start:.0f}s")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")  # show up to 3 example failures
    return PredictionSet(graphs, ids), pd.DataFrame(meta), failed  # graphs as a Dataset, plus a metadata table and failure list


def load_model(tag, results_dir):
    """Rebuild a trained model and its normalizer from the saved checkpoint.

    The architecture is read back from the checkpoint's own saved args rather
    than re-specified here, so a model trained with different widths still
    loads correctly instead of failing on a shape mismatch.
    """
    path = os.path.join(results_dir, f"model_{tag}.pth")  # checkpoint path for this run's tag
    if not os.path.exists(path):
        sys.exit(f"No checkpoint at {path} - train it first with 02_train.py")  # abort with a helpful message

    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # load the checkpoint dict onto CPU regardless of training device
    targs = ckpt["args"]  # the argparse Namespace (as a dict) the model was trained with

    model = CrystalGraphConvNet(
        ckpt["feature_lens"]["orig_atom_fea_len"],  # atom feature vector length, read back from the checkpoint
        ckpt["feature_lens"]["nbr_fea_len"],  # bond/neighbor feature vector length
        atom_fea_len=targs["atom_fea_len"], n_conv=targs["n_conv"],
        h_fea_len=targs["h_fea_len"], n_h=targs["n_h"], classification=False)  # rebuild the exact architecture used at training time
    model.load_state_dict(ckpt["state_dict"])  # load the trained weights into that architecture
    model.eval()  # switch off dropout/batchnorm training behavior for inference

    normalizer = Normalizer(torch.zeros(1))  # placeholder normalizer, immediately overwritten below
    normalizer.load_state_dict(ckpt["normalizer"])  # restore the exact mean/std used to de-normalise predictions
    return model, normalizer, ckpt


def predict_log(model, normalizer, dataset, batch_size=64):
    """Run one model over the whole set, returning {id: log10(modulus in GPa)}.

    Stays in log space because that is where ensemble members get averaged -
    see the note in 05_ensemble.py about why averaging in GPa is wrong.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,  # no shuffling needed, this is inference not training
                        collate_fn=collate_pool)  # batches variable-sized graphs into the model's expected tensor layout
    out = {}  # material_id -> predicted log10(modulus) map, filled in below
    with torch.no_grad():  # disables gradient tracking, since nothing here is trained
        for inputs, _, ids in loader:  # inputs = graph tensors, _ = unused dummy targets, ids = material ids in this batch
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs  # unpack the four tensors collate_pool produced
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)  # forward pass, one prediction per crystal in the batch
            pred_log = normalizer.denorm(output.data).view(-1).numpy()  # undo z-scoring, flatten to 1-D, convert to numpy
            for material_id, value in zip(ids, pred_log):  # pair each id in the batch with its prediction
                out[material_id] = float(value)  # store as a plain Python float
    return out


def predict_ensemble(models, dataset):
    """Average several models' log-space predictions, back-transformed to GPa.

    Returns (moduli in GPa, per-crystal member spread). The spread is a free
    uncertainty estimate: where the members disagree, the ensemble is guessing,
    and a crystal with a large spread deserves less trust in the kappa_L stage.
    """
    per_member = [predict_log(model, normalizer, dataset)  # one {id: log10 value} dict per ensemble member
                  for model, normalizer in models]

    moduli, spread = {}, {}  # final GPa predictions and per-crystal member disagreement
    for material_id in per_member[0]:  # iterate over ids using the first member's keys as the reference set
        values = [m[material_id] for m in per_member]  # this crystal's log10 prediction from every member
        moduli[material_id] = float(10 ** np.mean(values))  # average in log space, then convert back to GPa
        spread[material_id] = float(np.std(values))  # standard deviation across members, in log10 units
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
    mapping_path = os.path.join(data_full_dir, "mp_to_mb.csv")  # path to the mp-id -> matbench-id lookup table
    if not os.path.exists(mapping_path):
        print(f"  no {mapping_path}; every row will be marked 'unknown'")
        return {}, {}  # empty provenance/reference maps if the mapping file is missing

    mapping = pd.read_csv(mapping_path)  # load the mp-id <-> matbench-id pairs

    # Rebuild the dataset's id ordering. GraphCacheData keeps the cache's order,
    # restricted to ids that have a label - the same filter clean_labels applied.
    blob = torch.load(os.path.join(data_full_dir, "graphs.pt"),
                      weights_only=False)  # the cached graph dict, including its original id ordering
    labelled = set(labels.mb_id)  # matbench ids that actually have a K/G label
    ordered_ids = [i for i in blob["ids"] if i in labelled]  # cache order, filtered down to labelled ids only

    mb_to_split = {}  # matbench id -> "train"/"val"/"test"
    for split_name, indices in ckpt["split"].items():  # ckpt["split"] maps split name to a list of dataset indices
        for idx in indices:
            mb_to_split[ordered_ids[idx]] = split_name  # index -> id -> split name

    mb_to_moduli = labels.set_index("mb_id")[["K_VRH", "G_VRH"]].to_dict("index")  # matbench id -> {"K_VRH":.., "G_VRH":..}

    provenance, reference = {}, {}
    for row in mapping.itertuples(index=False):  # iterate the mp-id/matbench-id pairs as namedtuples
        provenance[row.mp_id] = mb_to_split.get(row.mb_id, "unknown")  # "unknown" if this matbench id was never in a split
        if row.mb_id in mb_to_moduli:
            reference[row.mp_id] = mb_to_moduli[row.mb_id]  # attach the DFT reference values when one exists
    return provenance, reference


def main():
    parser = argparse.ArgumentParser(  # builds the CLI argument parser
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # reuses the module docstring as --help text, unwrapped
    parser.add_argument("--cif-dir", default=os.path.join(PROJECT_ROOT, "complete-data"))  # directory of CIFs to predict on
    parser.add_argument("--data-full", default=os.path.join(PROJECT_ROOT, "data_full"))  # where graphs.pt/labels.csv/mp_to_mb.csv live
    parser.add_argument("--results-dir", default=os.path.join(PROJECT_ROOT, "results", "cgcnn"))  # where checkpoints live and output is written
    parser.add_argument("--atom-init",
                        default=os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"))  # elemental feature table path
    parser.add_argument("--k-tag", default="K_VRH_full",
                        help="checkpoint tag(s) for the bulk modulus model. "
                             "Comma-separate several to average them as an "
                             "ensemble.")  # one or more comma-separated bulk-modulus checkpoint tags
    parser.add_argument("--g-tag", default="G_VRH_full",
                        help="checkpoint tag(s) for the shear modulus model, "
                             "comma-separated for an ensemble")  # same, for the shear-modulus model
    parser.add_argument("--out", default=None)  # optional explicit output path override
    args = parser.parse_args()  # parse sys.argv into the args namespace

    out_path = args.out or os.path.join(args.results_dir,
                                        "pink_moduli_predictions.csv")  # fall back to the default filename if --out was not given

    # --- Load both models --------------------------------------------------
    print("=== Predicting bulk and shear moduli for the PINK crystal set ===\n")
    k_tags = [t.strip() for t in args.k_tag.split(",") if t.strip()]  # split "--k-tag" on commas, dropping blanks
    g_tags = [t.strip() for t in args.g_tag.split(",") if t.strip()]  # same for the shear-modulus tag list

    k_loaded = [load_model(t, args.results_dir) for t in k_tags]  # (model, normalizer, ckpt) triples, one per K tag
    g_loaded = [load_model(t, args.results_dir) for t in g_tags]  # same for G tags
    k_models = [(m, n) for m, n, _ in k_loaded]  # drop the checkpoint dict, keep (model, normalizer) pairs
    g_models = [(m, n) for m, n, _ in g_loaded]
    # Provenance and split checks only need one representative checkpoint each;
    # 05_ensemble.py already refuses to combine members with differing splits.
    k_ckpt, g_ckpt = k_loaded[0][2], g_loaded[0][2]  # first member's checkpoint dict, used below for splits/provenance

    for tag, ckpt in list(zip(k_tags, [c for _, _, c in k_loaded])) + \
                     list(zip(g_tags, [c for _, _, c in g_loaded])):  # every (tag, checkpoint) pair across both models
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
                              weights_only=False)["config"]  # the dict of featuriser settings saved at cache-build time
    print(f"Featurisation from training cache: {cache_config}")

    dataset, meta, failed = build_prediction_graphs(
        args.cif_dir, args.atom_init, cache_config["max_num_nbr"],
        cache_config["radius"], cache_config["step"])  # featurise every CIF using those exact settings

    # --- Predict -----------------------------------------------------------
    print(f"\nRunning the bulk modulus model ({len(k_models)} member(s))...")
    k_pred, k_spread = predict_ensemble(k_models, dataset)  # {id: K_VRH in GPa}, {id: log10 spread across members}
    print(f"Running the shear modulus model ({len(g_models)} member(s))...")
    g_pred, g_spread = predict_ensemble(g_models, dataset)  # same, for shear modulus

    # --- Assemble the table ------------------------------------------------
    df = meta.copy()  # start from the (material_id, formula, n_sites) metadata table
    df["K_VRH_pred"] = df.material_id.map(k_pred)  # attach each row's predicted bulk modulus
    df["G_VRH_pred"] = df.material_id.map(g_pred)  # attach each row's predicted shear modulus
    # Only meaningful with 2+ members; with one it is identically zero.
    if len(k_models) > 1:
        df["K_VRH_spread_log10"] = df.material_id.map(k_spread)  # add the disagreement column only if there was an ensemble
    if len(g_models) > 1:
        df["G_VRH_spread_log10"] = df.material_id.map(g_spread)

    labels = pd.read_csv(os.path.join(args.data_full, "labels.csv"))  # matbench K/G labels, for the DFT-reference lookups below

    # The provenance column assumes both models were split the same way, which
    # they are: split_indices is seeded and both targets have the same row
    # count. Check it rather than trust it - if they ever diverge, a crystal
    # could be test for K and train for G, and one `provenance` column would be
    # quietly lying about one of them.
    if k_ckpt["split"]["test"] != g_ckpt["split"]["test"]:  # compares the two models' test-index lists for equality
        print("\nWARNING: the two models used different splits. The provenance "
              "column describes the BULK model only.")
    provenance, reference = build_provenance(args.data_full, k_ckpt, labels)  # {mp_id: split name}, {mp_id: {"K_VRH":..,"G_VRH":..}}

    df["provenance"] = df.material_id.map(provenance).fillna("unseen")  # ids absent from the map were outside matbench entirely
    df["K_VRH_dft"] = df.material_id.map(
        lambda m: reference.get(m, {}).get("K_VRH", np.nan))  # DFT bulk modulus where one exists, else NaN
    df["G_VRH_dft"] = df.material_id.map(
        lambda m: reference.get(m, {}).get("G_VRH", np.nan))  # same, for shear modulus

    # Pugh's ratio G/K: below ~0.57 a material is ductile, above it brittle.
    # It costs nothing to carry and the kappa_L discussion uses it.
    df["pugh_ratio"] = df.G_VRH_pred / df.K_VRH_pred  # elementwise division across the whole column

    columns = ["material_id", "formula", "n_sites", "provenance",
               "K_VRH_pred", "G_VRH_pred", "pugh_ratio",
               "K_VRH_dft", "G_VRH_dft"]  # fixed column order for the output CSV
    columns += [c for c in ("K_VRH_spread_log10", "G_VRH_spread_log10")
                if c in df.columns]  # append the spread columns only if they were actually created
    df = df[columns]  # reorder/select columns into that fixed layout
    df = df.sort_values("material_id").reset_index(drop=True)  # deterministic row order, fresh 0..n-1 index

    os.makedirs(args.results_dir, exist_ok=True)  # ensure the output directory exists before writing
    df.round(4).to_csv(out_path, index=False)  # write the CSV, 4 decimal places, no pandas index column

    # --- Report ------------------------------------------------------------
    print(f"\nWrote {len(df)} predictions to {out_path}")
    if failed:
        print(f"  ({len(failed)} CIFs could not be featurised and are absent)")

    print("\nProvenance breakdown:")
    print(df.provenance.value_counts().to_string())  # count of rows per provenance category

    print("\nPredicted moduli (GPa):")
    print(df[["K_VRH_pred", "G_VRH_pred", "pugh_ratio"]]
          .describe().round(2).to_string())  # mean/std/min/max/quartiles for these columns

    # Where a DFT reference exists, show the error - split by provenance, since
    # a low error on `train` rows is memorisation and a low error on `test` rows
    # is the number that actually means something.
    have_ref = df.dropna(subset=["K_VRH_dft"])  # rows where a DFT bulk-modulus reference is available
    if len(have_ref):
        print("\nAgreement with DFT where a reference exists "
              "(MAE in log10 GPa):")
        rows = []
        for split_name, sub in have_ref.groupby("provenance"):  # one group of rows per provenance value
            rows.append({
                "provenance": split_name,
                "n": len(sub),  # number of rows in this provenance group
                "K_MAE_log10": np.abs(np.log10(sub.K_VRH_pred) -
                                      np.log10(sub.K_VRH_dft)).mean(),  # mean absolute error, log10 GPa, bulk modulus
                "G_MAE_log10": np.abs(np.log10(sub.G_VRH_pred) -
                                      np.log10(sub.G_VRH_dft)).mean(),  # same, shear modulus
            })
        print(pd.DataFrame(rows).round(4).to_string(index=False))  # one printed row per provenance group, no pandas index


if __name__ == "__main__":  # only run main() when executed as a script, not when imported
    main()
