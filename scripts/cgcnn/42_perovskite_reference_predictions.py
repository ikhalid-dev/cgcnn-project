#!/usr/bin/env python3
"""
STEP 42 - One reference table: true vs predicted K/G/gamma/kappa for
perovskites, both the ones we have labels for and the ones we don't.
================================================================================

    python scripts/cgcnn/42_perovskite_reference_predictions.py

WHY THIS SCRIPT EXISTS
------------------------
Two different questions about the model's perovskite performance keep
coming up, and they need two different kinds of data:

    "is the model's prediction actually right?"      -> needs a TRUE value
                                                          to compare against -
                                                          only matbench (300
                                                          perovskites, K/G
                                                          only) and AFLOW
                                                          (120, K/G/gamma/
                                                          kappa) have one.
    "what does the model say about a NEW perovskite?"  -> no true value
                                                          exists - JARVIS-DFT's
                                                          1,467 structurally
                                                          verified perovskites
                                                          (41_screen_jarvis_
                                                          perovskites.py) are
                                                          exactly this: real
                                                          structures, no
                                                          label this project
                                                          trusts (JARVIS's own
                                                          K/G convention was
                                                          already found to
                                                          disagree with
                                                          matbench worse than
                                                          AFLOW's - materials-
                                                          dft-databases memory).

Both belong in one table, not two, with a `has_ground_truth` flag - so
"how good is the model" and "what does it predict for something new" can be
answered from the same file without cross-referencing two different scripts'
output.

WHAT "PREDICTED" MEANS FOR EACH SOURCE
-------------------------------------------
matbench: the existing 3-model K/G ensemble's own predictions
(results/cgcnn/predictions_K_VRH_ens.csv / predictions_G_VRH_ens.csv,
already computed by 05_ensemble.py - reused, not rerun). No gamma/kappa
column for matbench rows: matbench has no measured gamma, and this
project's whole point (see the README's "Fixing the K/G correlation
problem" section) is that deriving one from K/G is a poor substitute -
reporting a derived number here would misrepresent it as comparable to
AFLOW's real one.

AFLOW and JARVIS: round 9's model (scripts/cgcnn/37_train_gamma.py),
3 seeds ensembled, run fresh here for AFLOW (matbench's ensemble script
never touches AFLOW) and read back from 41_screen_jarvis_perovskites.py's
own saved output for JARVIS (identical model, no need to re-run inference
that already happened).

WHY AFLOW'S FULL SET, NOT JUST ITS TEST SPLIT
-------------------------------------------------
Earlier checks in this project (session history) only ever scored AFLOW's
14 TEST-split perovskites, since that split is the only fair judge of
generalisation. This table includes train/val too, clearly flagged by
`split` - useful as a reference ("what does the model say about a
perovskite it trained on") but the train-split numbers should never be
quoted as an accuracy claim; only `split == "test"` rows are a fair test.

A THIRD, GENUINELY EXTERNAL CHECK: JARVIS'S OWN K/G
--------------------------------------------------------
Explicitly requested after the fact: matbench's and AFLOW's test splits are
still THIS project's own reserved holdout, carved out of data used to build
the model in the first place. JARVIS's 1,467 verified perovskites were never
in that split at all - not train, not val, not test - because JARVIS was
never used for training anything. 567 of them happen to have JARVIS's own
reported bulk_modulus_kv/shear_modulus_gv, fetched here and used as
`true_K_GPa`/`true_G_GPa` for those rows specifically (has_ground_truth
flips to True only for these 567; the other 900 verified perovskites still
have no trusted label and stay NaN).

This is NOT a clean accuracy number the way the matbench/AFLOW test splits
are, and is reported as such: this project already measured JARVIS's own
elastic-modulus convention disagreeing with matbench's WORSE than AFLOW's
does (0.127/0.155 log10 K/G vs AFLOW's 0.094/0.109 - materials-dft-
databases memory) - so error against JARVIS's values here is real model
error PLUS that convention gap, not one cleanly separated from the other.
Worth having anyway, because it is the only fully-external check available:
genuinely unseen structures, genuinely unseen labels, zero overlap with
anything this model has ever been fit or selected against.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- known-perovskite sources (have a true value) -------------------------
    "matbench_labels_csv": "data_full/labels.csv",
    "matbench_k_pred_csv": "results/cgcnn/predictions_K_VRH_ens.csv",
    "matbench_g_pred_csv": "results/cgcnn/predictions_G_VRH_ens.csv",
    "aflow_labels_csv": "data_full/gamma_labels.csv",
    "aflow_graphs_pt": "data_full/gamma_graphs.pt",

    # ---- external-check source (never in train/val/test at all) ---------------
    "jarvis_dataset": "dft_3d",   # jarvis.db.figshare dataset name, for its own reported K/G
    "jarvis_screen_csv": "results/cgcnn/41_jarvis_perovskite_screen_all.csv",

    # ---- the round-9 gamma model, run fresh for AFLOW (identical to 39/41) ----
    "results_dir": "results/cgcnn",
    "r9_tags": "r9_full_s42,r9_full_s1,r9_full_s2",

    # ---- split reconstruction (must match every training script exactly) -----
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "split_seed": 42,

    # ---- output -----------------------------------------------------------
    "families_dir": "results/families",
}
# =============================================================================

import os                 # path joining/creation
import sys                 # sys.path mutation
import warnings              # silences pymatgen's formula-parsing warnings
from importlib import import_module   # loads numeric-prefixed sibling scripts by string name

import torch  # noqa: F401   # torch first - see cgcnn_scratch/data.py for why (MKL/OpenMP import-order guard)
import numpy as np                          # the split-reconstruction PRNG
import pandas as pd                          # every table this script reads and writes
from pymatgen.core import Composition        # formula parsing for the perovskite/anion-former check

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of caller's cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # makes sibling numeric-prefixed scripts importable by name

from cgcnn_scratch.joint import JointGraphCacheData   # noqa: E402  the (graph, target, id) dataset AFLOW's gamma cache uses
_screen39 = import_module("39_screen_gnome_gamma")     # load_gamma_model / predict_gamma_model
_sort40 = import_module("40_sort_by_family")           # classify_family / ANION_FORMERS, reused rather than re-defined

# Same exclusion set as 40/41 - see those scripts' docstrings for the Te and
# per-polymorph (KAsO3-style) false positives this catches.
ANION_FORMERS = _sort40.ANION_FORMERS


def is_real_perovskite(formula):
    try:
        comp = Composition(formula)
    except Exception:
        return False
    elements = {str(el) for el in comp.elements}
    if "O" not in elements:
        return False
    amounts = comp.reduced_composition.get_el_amt_dict()
    o_amt = amounts.pop("O", 0)
    if set(amounts) & ANION_FORMERS:
        return False
    return len(amounts) == 2 and sorted(amounts.values()) == [1, 1] and o_amt == 3


def split_indices(n_total, train_ratio, val_ratio, seed):
    rng = np.random.RandomState(seed)              # seeded PRNG, deterministic given `seed`
    idx = rng.permutation(n_total)                  # shuffled array of indices 0..n_total-1
    n_tr = int(round(train_ratio * n_total))         # crystals assigned to train
    n_va = int(round(val_ratio * n_total))            # crystals assigned to val
    return idx[:n_tr].tolist(), idx[n_tr:n_tr + n_va].tolist(), idx[n_tr + n_va:].tolist()


def matbench_perovskite_table():
    """True vs predicted K/G for every matbench perovskite, every split."""
    labels = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["matbench_labels_csv"]))
    is_pero = labels["formula"].apply(is_real_perovskite)
    pero_ids = set(labels.loc[is_pero, "mb_id"])
    print(f"  {len(pero_ids)} matbench perovskites")

    k_pred = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["matbench_k_pred_csv"]))
    g_pred = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["matbench_g_pred_csv"]))
    k_pred = k_pred[k_pred.material_id.isin(pero_ids)][["material_id", "split", "true_GPa", "pred_GPa"]]
    g_pred = g_pred[g_pred.material_id.isin(pero_ids)][["material_id", "pred_GPa", "true_GPa"]]

    out = k_pred.rename(columns={"true_GPa": "true_K_GPa", "pred_GPa": "pred_K_GPa"})
    out = out.merge(g_pred.rename(columns={"true_GPa": "true_G_GPa", "pred_GPa": "pred_G_GPa"}),
                    on="material_id", how="left")
    out = out.merge(labels[["mb_id", "formula"]].rename(columns={"mb_id": "material_id"}),
                    on="material_id", how="left")
    out["source"] = "matbench"
    out["has_ground_truth"] = True
    for col in ("true_gamma", "pred_gamma", "true_kappa", "pred_kappa"):
        out[col] = np.nan   # matbench has no real gamma/kappa - explicit NaN, not a derived stand-in (see module docstring)
    return out


def aflow_perovskite_table():
    """True vs predicted K/G/gamma/kappa for every AFLOW perovskite, every
    split - runs the round-9 ensemble fresh, since nothing else in this
    project has saved AFLOW's OWN per-crystal round-9 predictions to a file
    before (only GNoME's and JARVIS's)."""
    meta = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["aflow_labels_csv"])).set_index("gid")
    log_K = np.log10(meta.K_VRH.values)
    log_G = np.log10(meta.G_VRH.values)
    log_gamma = np.log10(meta.gamma.values)
    cols = [log_G, log_K - log_G, log_gamma]
    targets = {i: tuple(c[k] for c in cols) for k, i in enumerate(meta.index)}
    dataset = JointGraphCacheData(os.path.join(PROJECT_ROOT, CONFIG["aflow_graphs_pt"]), targets)

    tr, va, te = split_indices(len(dataset), CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["split_seed"])
    split_of = {}
    for i in tr: split_of[dataset.ids[i]] = "train"     # noqa: E701 - deliberately compact
    for i in va: split_of[dataset.ids[i]] = "val"       # noqa: E701
    for i in te: split_of[dataset.ids[i]] = "test"      # noqa: E701

    is_pero = meta["formula"].apply(is_real_perovskite)
    pero_ids = [i for i in dataset.ids if is_pero.get(i, False)]
    print(f"  {len(pero_ids)} AFLOW perovskites "
         f"(train {sum(split_of[i]=='train' for i in pero_ids)} / "
         f"val {sum(split_of[i]=='val' for i in pero_ids)} / "
         f"test {sum(split_of[i]=='test' for i in pero_ids)})")

    # predict_gamma_model iterates whatever Dataset it is handed in full, so
    # restrict to the perovskite rows here rather than passing the whole
    # 5,563-crystal AFLOW set through the model 3x for a ~117-row answer.
    # JointGraphCacheData.graphs is a dict keyed BY ID (not a positional
    # list - checked directly in cgcnn_scratch/joint.py before relying on
    # it), so each perovskite's graph is a plain dict lookup, no index math.
    _predict = import_module("04_predict_moduli")   # PredictionSet - the (graph, dummy target, id) Dataset wrapper
    pero_graphs = [dataset.graphs[i] for i in pero_ids]
    pero_dataset = _predict.PredictionSet(pero_graphs, pero_ids)
    sample_shapes = (pero_graphs[0][0].shape[-1], pero_graphs[0][1].shape[-1])

    results_dir = os.path.join(PROJECT_ROOT, CONFIG["results_dir"])
    per_seed = []
    for tag in CONFIG["r9_tags"].split(","):
        model, norm, _ = _screen39.load_gamma_model(tag.strip(), results_dir, sample_shapes)
        preds = _screen39.predict_gamma_model(model, norm, pero_dataset)
        per_seed.append(preds)

    from importlib import import_module as _im
    _gamma = _im("37_train_gamma")

    log_kappa_seeds = np.zeros((len(pero_ids), 3))
    log_K_seeds = np.zeros((len(pero_ids), 3))
    log_G_seeds = np.zeros((len(pero_ids), 3))
    gamma_seeds = np.zeros((len(pero_ids), 3))
    vol = meta.loc[pero_ids, "volume_m3"].values
    nst = meta.loc[pero_ids, "n_sites"].values
    dens = meta.loc[pero_ids, "density_g_cm3"].values
    for s, preds in enumerate(per_seed):
        row = np.array([preds[i] for i in pero_ids])
        lg = row[:, 0]; lk = row[:, 0] + row[:, 1]; gam = 10.0 ** row[:, 2]
        kap = _gamma.kappa_full(lk, lg, gam, vol, nst, dens)
        with np.errstate(all="ignore"):
            log_kappa_seeds[:, s] = np.log10(np.where(kap > 0, kap, np.nan))
        log_K_seeds[:, s] = lk; log_G_seeds[:, s] = lg; gamma_seeds[:, s] = gam

    out = pd.DataFrame({
        "material_id": pero_ids,
        "formula": meta.loc[pero_ids, "formula"].values,
        "split": [split_of[i] for i in pero_ids],
        "true_K_GPa": meta.loc[pero_ids, "K_VRH"].values,
        "pred_K_GPa": 10.0 ** np.nanmean(log_K_seeds, axis=1),
        "true_G_GPa": meta.loc[pero_ids, "G_VRH"].values,
        "pred_G_GPa": 10.0 ** np.nanmean(log_G_seeds, axis=1),
        "true_gamma": meta.loc[pero_ids, "gamma"].values,
        "pred_gamma": np.nanmean(gamma_seeds, axis=1),
        "true_kappa": meta.loc[pero_ids, "kappa_agl"].values,
        "pred_kappa": 10.0 ** np.nanmean(log_kappa_seeds, axis=1),
    })
    out["source"] = "aflow"
    out["has_ground_truth"] = True
    return out


def jarvis_reference_table():
    """The round-9 model's own prediction, read back from
    41_screen_jarvis_perovskites.py's already-computed output (no need to
    re-run inference that already happened) - PLUS, where JARVIS itself
    reports one, its own bulk_modulus_kv/shear_modulus_gv as a genuinely
    external true value. See the module docstring's "third, genuinely
    external check" section for why this is real ground truth but not a
    clean accuracy number the way matbench/AFLOW's test splits are."""
    df = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["jarvis_screen_csv"]))
    df = df[df["verified_perovskite"]].copy()   # only the structurally-confirmed ones, see 41's own docstring for why

    from jarvis.db.figshare import data as jarvis_data   # imported here, not at module top, to keep the import-order guard clean
    ids = set(df["material_id"])
    true_k, true_g = {}, {}
    for entry in jarvis_data(CONFIG.get("jarvis_dataset", "dft_3d")):
        if entry["jid"] not in ids:
            continue
        k, g = entry.get("bulk_modulus_kv"), entry.get("shear_modulus_gv")
        if k not in (None, "na", "") and g not in (None, "na", ""):
            true_k[entry["jid"]] = float(k)
            true_g[entry["jid"]] = float(g)
    print(f"  {len(true_k)}/{len(df)} have JARVIS's own reported K/G (used as an external true value below)")

    out = pd.DataFrame({
        "material_id": df["material_id"], "formula": df["formula"], "split": "n/a (not a training crystal)",
        "true_K_GPa": df["material_id"].map(true_k), "pred_K_GPa": df["K_r9_pred"],
        "true_G_GPa": df["material_id"].map(true_g), "pred_G_GPa": df["G_r9_pred"],
        "true_gamma": np.nan, "pred_gamma": df["gamma_r9_pred"],   # JARVIS has no lattice gamma at all - see materials-dft-databases memory
        "true_kappa": np.nan, "pred_kappa": df["Kappa_r9_gamma"],  # same for lattice kappa - nkappa/pkappa are electronic, not this
    })
    out["source"] = "jarvis"
    out["has_ground_truth"] = out["material_id"].isin(true_k)   # True only for the 567 with a real external K/G, not all 1,467
    print(f"  {len(out)} JARVIS reference perovskites total ({out['has_ground_truth'].sum()} with an external label)")
    return out


def main():
    print("=" * 78)
    print("  STEP 42 - perovskite reference table: true vs predicted, all sources")
    print("=" * 78)

    print("\nMatbench (K/G only - no real gamma/kappa label exists)...")
    mb = matbench_perovskite_table()

    print("\nAFLOW (K/G/gamma/kappa, round-9 model run fresh)...")
    af = aflow_perovskite_table()

    print("\nJARVIS-DFT (prediction only, no trusted label)...")
    jv = jarvis_reference_table()

    cols = ["source", "material_id", "formula", "split", "has_ground_truth",
           "true_K_GPa", "pred_K_GPa", "true_G_GPa", "pred_G_GPa",
           "true_gamma", "pred_gamma", "true_kappa", "pred_kappa"]
    combined = pd.concat([mb[cols], af[cols], jv[cols]], ignore_index=True)

    out_dir = os.path.join(PROJECT_ROOT, CONFIG["families_dir"], "perovskite-like-abo3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "true_vs_predicted.csv")
    combined.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(combined.groupby(["source", "split"], dropna=False).size().to_string())

    print("\nTEST-SPLIT ACCURACY (the only fair judge; train/val rows are reference, not an accuracy claim):")
    for src in ("matbench", "aflow"):
        sub = combined[(combined.source == src) & (combined.split == "test")]
        if not len(sub):
            continue
        mae_k = np.abs(np.log10(sub.pred_K_GPa) - np.log10(sub.true_K_GPa)).mean()
        mae_g = np.abs(np.log10(sub.pred_G_GPa) - np.log10(sub.true_G_GPa)).mean()
        line = f"  {src:<10} n={len(sub):3d}  MAE log10(K)={mae_k:.4f}  log10(G)={mae_g:.4f}"
        if src == "aflow":
            mae_kap = np.abs(np.log10(sub.pred_kappa) - np.log10(sub.true_kappa)).mean()
            line += f"  log10(kappa)={mae_kap:.4f}"
        print(line)

    print("\nEXTERNAL CHECK - JARVIS's own K/G, on structures NEVER in train/val/test at all")
    print("(includes real model error PLUS JARVIS's known worse-than-AFLOW convention gap - not a clean number, see docstring):")
    ext = combined[(combined.source == "jarvis") & combined.has_ground_truth]
    if len(ext):
        mae_k = np.abs(np.log10(ext.pred_K_GPa) - np.log10(ext.true_K_GPa)).mean()
        mae_g = np.abs(np.log10(ext.pred_G_GPa) - np.log10(ext.true_G_GPa)).mean()
        bias_k = (np.log10(ext.pred_K_GPa) - np.log10(ext.true_K_GPa)).mean()   # signed, not absolute - shows systematic over/under-prediction
        bias_g = (np.log10(ext.pred_G_GPa) - np.log10(ext.true_G_GPa)).mean()
        print(f"  jarvis     n={len(ext):3d}  MAE log10(K)={mae_k:.4f}  log10(G)={mae_g:.4f}  "
             f"(signed bias K={bias_k:+.4f} G={bias_g:+.4f})")

    print(f"\nWrote {out_path} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
