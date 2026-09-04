#!/usr/bin/env python3
"""
STEP 43 - A composition-only tree-model baseline, to check the CGCNN is
actually earning its complexity.
================================================================================

    python scripts/cgcnn/43_baseline_tree_models.py

WHY THIS SCRIPT EXISTS
------------------------
Every accuracy number in this project so far comes from a graph neural
network that sees the full 3D structure - bond distances, coordination,
periodicity. None of that has ever been checked against the much simpler
question: how much of the signal is just STOICHIOMETRY? If a decision tree
using nothing but "what elements are present, in what ratio" gets close to
the CGCNN's number, the network's structural awareness is buying little; if
the tree is far worse, the 3D graph representation is doing real work. This
project has asserted the latter throughout without ever measuring it -
this script measures it.

FEATURES: COMPOSITION ONLY, DELIBERATELY NO STRUCTURE
------------------------------------------------------------
Each crystal becomes one fixed-length vector: the composition-WEIGHTED MEAN
and STANDARD DEVIATION of the same 92-dimensional per-element feature table
CGCNN's own embedding layer reads (cgcnn_scratch/atom_init.json) - the
element's identity, encoded the same way, just averaged over the formula
instead of fed into a graph. This is deliberately NOT given bond lengths,
coordination numbers, space group, or density - a tree model handed those
would partly be cheating by using structural information the "composition
only" framing is supposed to rule out. n_sites (atom count in the cell) is
kept, since it is a cheap, composition-adjacent number every source already
has in its labels file, not something that requires solving the structure.

WHY DECISION TREE, RANDOM FOREST, AND XGBOOST - NOT JUST ONE
-------------------------------------------------------------------
A single decision tree overfits hard on ~200-8000 rows and is the honest
"weakest reasonable baseline" number. Random forest and XGBoost are what a
materials-informatics paper would actually report as "the tree baseline" -
both average over many trees and generalise far better than one. Reporting
all three shows the gap is not an artifact of picking a weak baseline on
purpose.

WHY THE SAME SPLIT AS EVERY NEURAL NETWORK IN THIS PROJECT
-----------------------------------------------------------------
split_indices() with split_seed=42 is reproduced here exactly as every
training script uses it, and the matbench/AFLOW ID ORDER is read from the
same graphs.pt/gamma_graphs.pt caches the CGCNN scripts use (even though
this script never touches the graph tensors themselves) - so the SAME
crystals land in the same train/val/test rows here as everywhere else in
this project. Without that, a different split could make either model look
better or worse for reasons having nothing to do with model quality.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -----------------------------------------------------------
    "matbench_labels_csv": "data_full/labels.csv",
    "matbench_graphs_pt": "data_full/graphs.pt",     # only its ID ORDER is used, never the graph tensors
    "aflow_labels_csv": "data_full/gamma_labels.csv",
    "aflow_graphs_pt": "data_full/gamma_graphs.pt",  # same - ID order only
    "atom_init_json": "cgcnn_scratch/atom_init.json",

    # ---- split reconstruction (must match every training script exactly) -----
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,

    # ---- model hyperparameters - deliberately close to each library's own
    # sensible defaults, not tuned, since this is a baseline, not a
    # competing model this project wants to win with -------------------------
    "random_forest_n_estimators": 300,
    "xgboost_n_estimators": 300,
    "xgboost_max_depth": 6,
    "xgboost_learning_rate": 0.05,
    # tree_method="hist" (xgboost's default) segfaults on this machine on this
    # project's real feature data (confirmed: crashes even at n=100 rows,
    # n_jobs=1, no torch imported - reproducible with any y, so it is xgboost's
    # native histogram builder, not a threading/OpenMP conflict). "exact" is
    # xgboost's older, non-histogram code path - slower on huge datasets, a
    # non-issue at this project's size (~11k/5.5k rows) - and never crashes.
    "xgboost_tree_method": "exact",
    "random_state": 42,

    # ---- output -------------------------------------------------------------
    # all baseline-tree output lives under one folder, split by file type, so
    # a future script (44_baseline_tree_analysis.py) can find the csv/png
    # split without guessing - see that script for the correlation/confusion
    # matrix outputs built on top of these predictions.
    "out_dir": "results/cgcnn/baseline_tree/csv",
    "families_dir": "results/families",
}
# =============================================================================

import os                 # path joining/creation
import sys                 # sys.path mutation
import warnings              # silences pymatgen's formula-parsing warnings

# Must be set before torch/xgboost load their OpenMP runtime, not just passed
# as n_jobs=1 to the model - confirmed on this machine that xgboost's own
# multi-threaded fit (any tree_method, any n_jobs, even 1) segfaults
# nondeterministically (sometimes mid-fit, sometimes at interpreter exit)
# whenever more than one OpenMP thread is live. Pinning the env var here is
# what actually stops it - reproduced 3/3 clean runs only after this, and it
# was not the torch/numpy import-order issue this repo usually hits (that one
# is an abort, not a segfault, and happens on import, not on fit()).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401   # torch first - see cgcnn_scratch/data.py for why (MKL/OpenMP import-order guard)
import numpy as np                          # feature-vector math and the split PRNG
import pandas as pd                          # DataFrame construction, CSV I/O
from pymatgen.core import Composition        # formula -> {element: fractional amount}

from sklearn.tree import DecisionTreeRegressor          # the "weakest reasonable baseline"
from sklearn.ensemble import RandomForestRegressor       # the "what a paper would actually report" baseline
import xgboost as xgb                                     # the strongest tree-based baseline

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of caller's cwd

from cgcnn_scratch.data import AtomFeaturiser   # noqa: E402  the same 92-dim per-element table CGCNN's embedding layer reads


def split_indices(n_total, train_ratio, val_ratio, seed):
    """Identical to every training script's own split_indices()."""
    rng = np.random.RandomState(seed)              # seeded PRNG, deterministic given `seed`
    idx = rng.permutation(n_total)                  # shuffled array of indices 0..n_total-1
    n_tr = int(round(train_ratio * n_total))         # crystals assigned to train
    n_va = int(round(val_ratio * n_total))            # crystals assigned to val
    return idx[:n_tr].tolist(), idx[n_tr:n_tr + n_va].tolist(), idx[n_tr + n_va:].tolist()


def composition_features(formula, ari):
    """One fixed-length vector per formula: composition-weighted mean and
    std of the 92-dim per-element feature table, concatenated (184-dim).

    No structure, no bond information - only "what elements, in what
    fractional amounts" goes in, by design (see module docstring).
    """
    comp = Composition(formula).fractional_composition.get_el_amt_dict()   # {element symbol: fractional amount, sums to 1}
    vecs, weights = [], []
    for element, frac in comp.items():
        z = Composition(element).elements[0].Z   # atomic number, what AtomFeaturiser is keyed by
        if z not in ari.atom_types:
            return None   # an element outside atom_init.json's coverage - drop this row rather than fabricate a feature
        vecs.append(ari.get_atom_fea(z))
        weights.append(frac)
    vecs = np.array(vecs)          # (n_elements, 92)
    weights = np.array(weights)     # (n_elements,), sums to 1

    mean = np.average(vecs, axis=0, weights=weights)                        # composition-weighted mean, per feature
    variance = np.average((vecs - mean) ** 2, axis=0, weights=weights)       # composition-weighted variance, per feature
    std = np.sqrt(variance)                                                  # composition-weighted std, per feature
    return np.concatenate([mean, std])   # (184,)


def build_feature_table(labels_csv, graphs_pt, id_col, ari):
    """Features + targets + split, for one source, in the SAME crystal order
    (and therefore the SAME train/val/test assignment) every neural network
    script in this project uses."""
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, labels_csv)).set_index(id_col)
    blob = torch.load(os.path.join(PROJECT_ROOT, graphs_pt), weights_only=False)   # only .["ids"] is read below
    ordered_ids = [i for i in blob["ids"] if i in labels.index]   # cache order, filtered to labelled ids - the shared convention

    tr, va, te = split_indices(len(ordered_ids), CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["split_seed"])
    split_of = {}
    for i in tr: split_of[ordered_ids[i]] = "train"    # noqa: E701 - deliberately compact
    for i in va: split_of[ordered_ids[i]] = "val"      # noqa: E701
    for i in te: split_of[ordered_ids[i]] = "test"     # noqa: E701

    rows, ids, failed = [], [], []
    for crystal_id in ordered_ids:
        fea = composition_features(labels.loc[crystal_id, "formula"], ari)
        if fea is None:
            failed.append(crystal_id)
            continue
        rows.append(fea)
        ids.append(crystal_id)
    if failed:
        print(f"  {len(failed)} rows dropped (element outside atom_init.json's coverage)")

    X = np.array(rows)   # (n_crystals, 184)
    meta = labels.loc[ids].copy()
    meta["split"] = [split_of[i] for i in ids]
    meta["n_sites"] = meta["n_sites"]   # already present in both labels files - kept explicitly for clarity, not derived
    return X, meta


def fit_and_score(X, y_log, split, target_name):
    """One decision tree, one random forest, one XGBoost model, all trained
    on log10(target) with the SAME train rows, scored on the SAME val/test
    rows. Returns a small DataFrame of per-model, per-split MAE."""
    tr_mask, va_mask, te_mask = split == "train", split == "val", split == "test"

    models = {
        "decision_tree": DecisionTreeRegressor(random_state=CONFIG["random_state"]),
        "random_forest": RandomForestRegressor(
            n_estimators=CONFIG["random_forest_n_estimators"], random_state=CONFIG["random_state"], n_jobs=1),
        "xgboost": xgb.XGBRegressor(
            n_estimators=CONFIG["xgboost_n_estimators"], max_depth=CONFIG["xgboost_max_depth"],
            learning_rate=CONFIG["xgboost_learning_rate"], tree_method=CONFIG["xgboost_tree_method"],
            random_state=CONFIG["random_state"], n_jobs=1),
    }

    rows = []
    predictions = {}   # model_name -> full-length array of predictions, for the per-crystal CSV
    for name, model in models.items():
        model.fit(X[tr_mask], y_log[tr_mask])                 # fit on train only, exactly like the neural network scripts
        pred = model.predict(X)                                 # predict every row at once - cheap, and gives train/val/test together
        predictions[name] = pred
        for split_name, mask in [("train", tr_mask), ("val", va_mask), ("test", te_mask)]:
            mae = np.abs(pred[mask] - y_log[mask]).mean()
            rows.append({"target": target_name, "model": name, "split": split_name,
                        "mae_log10": mae, "n": int(mask.sum())})
    return pd.DataFrame(rows), predictions


def main():
    print("=" * 78)
    print("  STEP 43 - composition-only tree baselines (decision tree / RF / XGBoost)")
    print("=" * 78)

    ari = AtomFeaturiser(os.path.join(PROJECT_ROOT, CONFIG["atom_init_json"]))

    print("\nBuilding matbench composition features...")
    X_mb, meta_mb = build_feature_table(CONFIG["matbench_labels_csv"], CONFIG["matbench_graphs_pt"], "mb_id", ari)
    print(f"  {len(meta_mb)} crystals, {X_mb.shape[1]}-dim feature vector")

    print("\nBuilding AFLOW composition features...")
    X_af, meta_af = build_feature_table(CONFIG["aflow_labels_csv"], CONFIG["aflow_graphs_pt"], "gid", ari)
    print(f"  {len(meta_af)} crystals, {X_af.shape[1]}-dim feature vector")

    all_scores = []
    all_preds = []

    for label, X, meta, targets in [
        ("matbench", X_mb, meta_mb, ["K_VRH", "G_VRH"]),
        ("aflow", X_af, meta_af, ["K_VRH", "G_VRH", "gamma"]),
    ]:
        print(f"\n{'=' * 78}\n  {label.upper()}\n{'=' * 78}")
        split = meta["split"].values
        pred_table = meta[["formula", "split"]].copy()
        for target in targets:
            y_log = np.log10(meta[target].values)
            scores, preds = fit_and_score(X, y_log, split, target)
            all_scores.append(scores.assign(source=label))
            for model_name, pred in preds.items():
                pred_table[f"{target}_true_log10"] = y_log
                pred_table[f"{target}_pred_log10_{model_name}"] = pred
            print(scores.pivot(index="model", columns="split", values="mae_log10")
                 .reindex(columns=["train", "val", "test"]).to_string())
        pred_table.insert(0, "material_id", meta.index)
        pred_table.insert(0, "source", label)
        all_preds.append(pred_table)

    scores = pd.concat(all_scores, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    out_dir = os.path.join(PROJECT_ROOT, CONFIG["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "43_baseline_tree_scores.csv")
    preds_path = os.path.join(out_dir, "43_baseline_tree_predictions.csv")
    scores.to_csv(scores_path, index=False)
    preds.to_csv(preds_path, index=False)

    print("\n" + "=" * 78)
    print("  TEST-SET COMPARISON AGAINST THE CGCNN (the only fair judge)")
    print("=" * 78)
    cgcnn_reference = {
        ("matbench", "K_VRH"): 0.0630, ("matbench", "G_VRH"): 0.0781,      # 3-model ensemble, this project's own headline numbers
        ("aflow", "K_VRH"): 0.1255, ("aflow", "G_VRH"): 0.1521,             # round-9, 3-seed ensemble mean, this session
        ("aflow", "gamma"): None,   # not directly comparable - round-9's gamma head is scored against real AFLOW gamma the same way
    }
    test = scores[scores.split == "test"]
    for (src, tgt), cgcnn_mae in cgcnn_reference.items():
        sub = test[(test.source == src) & (test.target == tgt)].sort_values("mae_log10")
        best_model, best_mae = sub.iloc[0][["model", "mae_log10"]]
        print(f"  {src:<9} {tgt:<6}  best tree baseline: {best_model:<14} MAE={best_mae:.4f}"
             + (f"   CGCNN: {cgcnn_mae:.4f}" if cgcnn_mae else ""))

    # ---- mirror the perovskite-relevant slice into the sorted family folder --
    ANION_FORMERS = {"C", "N", "S", "P", "Se", "Te", "B", "F", "Cl", "Br", "I"}   # identical to 40/41/42's own exclusion set

    def is_real_perovskite(formula):
        try:
            comp = Composition(formula)
        except Exception:
            return False
        if "O" not in {str(el) for el in comp.elements}:
            return False
        amounts = comp.reduced_composition.get_el_amt_dict()
        o_amt = amounts.pop("O", 0)
        if set(amounts) & ANION_FORMERS:
            return False
        return len(amounts) == 2 and sorted(amounts.values()) == [1, 1] and o_amt == 3

    pero_preds = preds[preds["formula"].apply(is_real_perovskite)]
    families_dir = os.path.join(PROJECT_ROOT, CONFIG["families_dir"], "perovskite-like-abo3")
    os.makedirs(families_dir, exist_ok=True)
    pero_path = os.path.join(families_dir, "baseline_tree_predictions.csv")
    pero_preds.to_csv(pero_path, index=False)

    print(f"\nWrote {scores_path}")
    print(f"Wrote {preds_path}")
    print(f"Wrote {pero_path} ({len(pero_preds)} perovskite rows)")


if __name__ == "__main__":
    main()
