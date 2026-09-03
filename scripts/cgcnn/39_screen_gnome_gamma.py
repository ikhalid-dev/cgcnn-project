#!/usr/bin/env python3
"""
STEP 39 - Re-score the GNoME screen with the TRAINED gamma head, not the
derived one.
================================================================================

    python scripts/cgcnn/39_screen_gnome_gamma.py

    # quick smoke test on a small subset first:
    python scripts/cgcnn/39_screen_gnome_gamma.py --limit 300

WHY THIS SCRIPT EXISTS
-----------------------
13_screen_gnome.py (and everything upstream of it: 04_predict_moduli.py,
07_predict_kappa.py) computes the Gruneisen parameter the Slack model needs
by DERIVING it from the predicted K/G ratio through an empirical Poisson-ratio
formula (slack_physics() in 07_predict_kappa.py). That derivation was
measured directly against AFLOW's own tabulated gamma:

    gamma:  derived   MAE 0.3551   corr +0.334
            PREDICTED MAE 0.139-0.149   corr +0.53-0.55   (round 9, 3 seeds)

Round 9 (37_train_gamma.py) trained a 3-head model - log10(G), log10(K/G),
log10(gamma) - on the full 5,563-crystal AFLOW set specifically so gamma
would not have to be derived. That head has been sitting unused: nothing
upstream of the GNoME screen was ever rewired to call it, because the screen
predates round 9. This script closes that gap - not by grafting the gamma
head onto the EXISTING matbench-trained K/G predictions (which would mix
AFLOW's elastic-modulus convention with matbench's Voigt-Reuss-Hill
convention inside one kappa number - exactly the kind of convention-mixing
that cost real accuracy in rounds 7/8), but by running the round-9
checkpoint's OWN K, G, and gamma heads together, self-consistently, on every
GNoME candidate. Every number in the output comes from one model family, one
convention, no mixing.

WHAT THIS DOES NOT CHANGE
---------------------------
The upstream filter (bandgap window, thermodynamic stability, no radioactive
elements) is IDENTICAL to 13_screen_gnome.py - reused via import_module, not
reimplemented, so the two screens are directly comparable on the same
candidate pool. Oxide detection/classification is reused from
15_filter_oxides.py the same way.

FEATURISATION IS RE-DONE, ON PURPOSE
--------------------------------------
13_screen_gnome.py builds its CGCNN graphs in memory and never writes them to
disk (see its own docstring for why raw CIFs are never cached). This script
needs those same 33k crystals featurised again for the round-9 model to score
- CIF parsing does not depend on which model reads the result afterward, so
the graphs ARE cached here (gnome_data/39_gnome_graphs.pt) so a second run
(a different --r9-tags selection, a different --limit) does not repeat the
~13 minute featurisation step.

HOW THE THREE-SEED ENSEMBLE'S UNCERTAINTY IS HANDLED
-------------------------------------------------------
The existing Monte Carlo kappa (07_predict_kappa.py's monte_carlo_kappa) was
built for K and G coming from two SEPARATELY trained model families, so it
treats their log-space errors as two independent Gaussians. That assumption
is wrong here: the round-9 model is ONE shared-trunk network per seed, so a
seed's K, G and gamma outputs are all correlated with each other (the same
trunk mistake shows up in all three heads at once - that correlation is the
entire point of the joint architecture, see cgcnn_scratch/joint.py). The
statistically honest way to propagate that is to compute the FULL kappa from
each seed's own (K, G, gamma) triple separately - 3 real kappa values per
crystal, not 3 independent per-modulus draws recombined - then treat the
spread of those 3 real numbers (in log10 space) as the ensemble's own
uncertainty and Monte-Carlo-sample around THAT. This never assumes K, G and
gamma are independent, because it never separates them.

WHAT IS REPORTED
------------------
Every candidate gets both numbers side by side: Kappa_cal_derived_matbench
(the existing 13_screen_gnome.py pipeline's value, unchanged, joined in for
comparison) and Kappa_r9_gamma (this script's new value). They will not
match exactly - different model family, different training convention - the
point of printing both is to make that visible, not to hide it.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs --------------------------------------------------------------
    # Same two GNoME files 13_screen_gnome.py reads - not re-downloaded here.
    "gnome_dir": "gnome_data",
    # Where trained checkpoints live and where this script's own outputs land.
    "results_dir": "results/cgcnn",
    # 13_screen_gnome.py's own scored table, joined in purely so the old
    # derived-gamma kappa sits next to the new one for comparison.
    "old_screen_csv": "results/cgcnn/13_gnome_screen_all.csv",

    # ---- candidate filter (IDENTICAL to 13_screen_gnome.py) ------------------
    "bandgap_lo": 0.1,   # eV, lower edge of the semiconductor window
    "bandgap_hi": 3.0,   # eV, upper edge
    "kappa_threshold": 1.0,   # W/m/K, the paper's own low-kappa cutoff

    # ---- the round-9 gamma model ----------------------------------------------
    # Three independently-seeded checkpoints from 37_train_gamma.py's round-9
    # run (full 5,563-crystal AFLOW set) - see model_37_r9_full_s*.pth in
    # results/cgcnn. Averaging across seeds is the same ensembling discipline
    # used everywhere else in this project (round 5 onward).
    "r9_tags": "r9_full_s42,r9_full_s1,r9_full_s2",

    # ---- Monte Carlo uncertainty ------------------------------------------
    "mc_samples": 2000,   # draws per crystal, matches 07_predict_kappa.py's default
    "seed": 0,            # RNG seed, for reproducible percentile bands

    # ---- featurisation (MUST match what 36_prepare_gamma_dataset.py used to
    # build the AFLOW graphs the round-9 model was trained on, or the model
    # sees features outside the distribution it was fitted on) -------------
    "max_num_nbr": 12,
    "radius": 8,
    "step": 0.2,

    # ---- caching --------------------------------------------------------------
    # Graphs are expensive to build (~13 min for 33k crystals) and do not
    # depend on which model reads them, so they are cached once here rather
    # than in 13_screen_gnome.py (which only ever needed them transiently).
    "graph_cache": "gnome_data/39_gnome_graphs.pt",

    "limit": None,   # only score the first N filtered candidates (smoke test)
}
# =============================================================================

import argparse          # CLI flag parsing, auto-generated from CONFIG below
import os                 # path joining/creation, existence checks
import sys                 # sys.path mutation and sys.exit() on fatal errors
import time                 # wall-clock timing for progress/ETA printouts
import warnings              # suppresses noisy-but-harmless parser warnings
from importlib import import_module   # loads numeric-prefixed sibling scripts by string name

# torch first - see cgcnn_scratch/data.py for why (MKL/OpenMP import-order guard).
import torch  # noqa: F401
from torch.utils.data import DataLoader   # batches the cached graphs for model inference

import numpy as np                          # array math for the ensemble/Monte Carlo steps
import pandas as pd                          # DataFrame construction, CSV I/O

warnings.filterwarnings("ignore")   # silences pymatgen/pandas warnings for the whole run

# walks up three directories from this file (scripts/cgcnn/ -> scripts/ -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of caller's cwd

from cgcnn_scratch.data import collate_pool                 # noqa: E402  batches variable-sized graphs into model-ready tensors
from cgcnn_scratch.joint import JointCrystalGraphConvNet, VectorNormalizer   # noqa: E402  the round-9 model class + its per-column normalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # makes sibling numeric-prefixed scripts importable by name
_screen = import_module("13_screen_gnome")     # module object: filter_candidates(), extract_and_featurise()
_oxide = import_module("15_filter_oxides")     # module object: has_oxygen(), classify_oxide()
_gamma = import_module("37_train_gamma")       # module object: kappa_full(), derived_gamma()


def load_gamma_model(tag, results_dir, sample_shapes):
    """Rebuild one round-9 checkpoint's architecture + weights + normalizer.

    Mirrors 04_predict_moduli.py's load_model(), but for the 3-head joint
    model instead of the single-head CrystalGraphConvNet, and reading the
    "39_"-numbered checkpoint filename convention 37_train_gamma.py writes.
    """
    path = os.path.join(results_dir, f"model_37_{tag}.pth")   # 37's own save-path convention (see that script's docstring)
    if not os.path.exists(path):
        sys.exit(f"No checkpoint at {path} - train it first with 37_train_gamma.py "
                 f"(or fetch it from Kaggle, see kaggle/run_joint_kernel.py)")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)   # load onto CPU regardless of the training device
    cfg = ckpt["config"]   # the full CONFIG dict 37_train_gamma.py trained this checkpoint with

    orig_atom_fea_len, nbr_fea_len = sample_shapes   # feature widths read off one of THIS run's own graphs
    model = JointCrystalGraphConvNet(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=cfg["atom_fea_len"], n_conv=cfg["n_conv"],
        h_fea_len=cfg["h_fea_len"], n_shared_fc=cfg["n_shared_fc"],
        n_head_fc=cfg["n_head_fc"], head_fea_len=cfg["head_fea_len"],
        dropout=cfg["dropout"], n_heads=cfg["n_heads"])   # rebuild the exact trained architecture
    model.load_state_dict(ckpt["state_dict"])   # load the trained weights into that architecture
    model.eval()   # disable dropout for inference

    normalizer = VectorNormalizer(torch.zeros(1, cfg["n_heads"]))   # placeholder stats, immediately overwritten below
    normalizer.load_state_dict(ckpt["normalizer"])   # restore the exact per-column mean/std used at training time
    return model, normalizer, cfg


def predict_gamma_model(model, normalizer, dataset, batch_size=64):
    """Run one round-9 checkpoint over every cached graph.

    Returns {material_id: (log10_G, log10_ratio, log10_gamma)}, all in
    physical (de-normalised) units - the exact three quantities score() in
    37_train_gamma.py works with, just for GNoME candidates instead of
    AFLOW's own held-out test set.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,   # no shuffling needed for inference
                        collate_fn=collate_pool)   # same batching function training used
    out = {}   # material_id -> (log10_G, log10_ratio, log10_gamma) tuple
    with torch.no_grad():   # no gradients needed, nothing is being trained
        for inputs, _, ids in loader:   # inputs = graph tensors, _ = unused dummy target, ids = this batch's material ids
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs   # unpack the 4-tuple collate_pool produced
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)   # forward pass: (B, 3) normalised predictions
            physical = normalizer.denorm(output.data).numpy()   # undo z-scoring, back to log10 physical units, as a numpy array
            for material_id, row in zip(ids, physical):   # pair each id in the batch with its 3-element prediction row
                out[material_id] = (float(row[0]), float(row[1]), float(row[2]))   # (log10_G, log10_ratio, log10_gamma)
    return out


def ensemble_and_score(per_seed_preds, meta):
    """Combine the 3 seeds' predictions into one point estimate + MC interval.

    per_seed_preds : list of {material_id: (log10_G, log10_ratio, log10_gamma)}
        one dict per seed, same material_id keys throughout.
    meta : DataFrame indexed by material_id, columns Number of Atoms / Volume
        (A3) / Density (g cm-3) - the structural constants kappa_full() needs.

    Returns a DataFrame indexed by material_id with the ensemble-mean K, G,
    gamma, kappa point estimate, and the MC p05/p50/p95 kappa band.
    """
    ids = list(per_seed_preds[0].keys())   # material ids, taken from the first seed (all three share the same keys)
    n_seeds = len(per_seed_preds)   # number of ensemble members (3, per CONFIG["r9_tags"])

    vol_m3 = (meta.loc[ids, "Volume (A3)"].values) * 1e-30   # cubic angstrom -> cubic metre, per candidate
    n_sites = meta.loc[ids, "Number of Atoms"].values   # atom count in the primitive cell, per candidate
    dens = meta.loc[ids, "Density (g cm-3)"].values   # g/cm^3, kappa_full() converts internally

    log_kappa_per_seed = np.zeros((len(ids), n_seeds))   # (n_crystals, n_seeds) - filled in the loop below
    log_G_per_seed = np.zeros((len(ids), n_seeds))       # same shape, log10(G) per seed
    log_K_per_seed = np.zeros((len(ids), n_seeds))       # same shape, log10(K) per seed
    gamma_per_seed = np.zeros((len(ids), n_seeds))       # same shape, real (non-log) gamma per seed

    for s, preds in enumerate(per_seed_preds):   # s = seed index, preds = that seed's {id: (G,ratio,gamma)} dict
        row = np.array([preds[i] for i in ids])   # (n_crystals, 3): columns log10_G, log10_ratio, log10_gamma, in `ids` order
        log_G = row[:, 0]                          # this seed's log10(G) for every crystal
        log_K = row[:, 0] + row[:, 1]               # log10(K) = log10(G) + log10(K/G), exact by construction
        gamma = 10.0 ** row[:, 2]                    # undo log10 on the gamma head's raw output
        kappa = _gamma.kappa_full(log_K, log_G, gamma, vol_m3, n_sites, dens)   # this seed's own full-formula kappa, W/m/K
        with np.errstate(all="ignore"):   # some rows can be non-physical (kappa<=0); suppress the warning, keep the NaN
            log_kappa_per_seed[:, s] = np.log10(np.where(kappa > 0, kappa, np.nan))   # log10(kappa), NaN where invalid
        log_G_per_seed[:, s] = log_G   # stash for the reported ensemble-mean G
        log_K_per_seed[:, s] = log_K   # stash for the reported ensemble-mean K
        gamma_per_seed[:, s] = gamma   # stash for the reported ensemble-mean gamma

    mean_log_kappa = np.nanmean(log_kappa_per_seed, axis=1)   # ensemble point estimate, per crystal, in log10 space
    std_log_kappa = np.nanstd(log_kappa_per_seed, axis=1)     # seed disagreement - the ensemble's own uncertainty measure
    # ^ this is the key departure from 07_predict_kappa.py's monte_carlo_kappa(): the spread is measured on the
    #   already-combined physical quantity (kappa itself, per seed), so it needs no independence assumption
    #   between K, G and gamma - see the module docstring for why that assumption would be wrong here.

    rng = np.random.RandomState(CONFIG["seed"])   # seeded PRNG, deterministic given CONFIG["seed"]
    z = rng.standard_normal((len(ids), CONFIG["mc_samples"]))   # (n_crystals, n_samples) standard-normal draws
    mc_log_kappa = mean_log_kappa[:, None] + std_log_kappa[:, None] * z   # per-crystal Normal(mean, std) draws, broadcast

    result = pd.DataFrame({
        "material_id": ids,
        "K_r9_pred": 10.0 ** np.nanmean(log_K_per_seed, axis=1),      # ensemble-mean K, GPa (geometric mean, log-space average)
        "G_r9_pred": 10.0 ** np.nanmean(log_G_per_seed, axis=1),      # ensemble-mean G, GPa
        "gamma_r9_pred": np.nanmean(gamma_per_seed, axis=1),          # ensemble-mean gamma (arithmetic mean, real units)
        "K_r9_spread_log10": np.nanstd(log_K_per_seed, axis=1),       # seed disagreement on K, log10 units
        "G_r9_spread_log10": np.nanstd(log_G_per_seed, axis=1),       # seed disagreement on G, log10 units
        "Kappa_r9_gamma": 10.0 ** mean_log_kappa,                      # the headline point estimate, W/m/K
        "Kappa_r9_p05": 10.0 ** np.nanpercentile(mc_log_kappa, 5, axis=1),    # pessimistic (high) bound at 5th percentile...
        "Kappa_r9_p50": 10.0 ** np.nanpercentile(mc_log_kappa, 50, axis=1),   # ...note: LOW kappa is "good", so ranking
        "Kappa_r9_p95": 10.0 ** np.nanpercentile(mc_log_kappa, 95, axis=1),   # by p95 (not p05) is the pessimistic choice
    }).set_index("material_id")   # index by material_id so a later .join() on the meta table lines up by id, not position
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)   # --help shows the module docstring verbatim
    for key, default in CONFIG.items():   # register one CLI flag per CONFIG entry, so CONFIG stays the single source of truth
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default) if default is not None else str,
                            default=None, help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()   # parse sys.argv against the registered flags
    for key, value in vars(args).items():   # walk each parsed argument
        if value is not None:               # None means the flag was not passed on the CLI
            CONFIG[key] = value              # otherwise the CLI value overrides the CONFIG default
    if CONFIG["limit"] is not None:
        CONFIG["limit"] = int(CONFIG["limit"])   # argparse's generic type() above can't know this one is an int when default is None

    gnome_dir = os.path.join(PROJECT_ROOT, CONFIG["gnome_dir"])   # absolute path to gnome_data/
    results_dir = os.path.join(PROJECT_ROOT, CONFIG["results_dir"])   # absolute path to results/cgcnn/
    cache_path = os.path.join(PROJECT_ROOT, CONFIG["graph_cache"])   # absolute path to this script's own graph cache

    print("=" * 78)
    print("  STEP 39 - GNoME screen with the TRAINED gamma head (round 9)")
    print("=" * 78)

    # ---- filter candidates, IDENTICAL criteria to 13_screen_gnome.py --------
    summary_path = os.path.join(gnome_dir, "stable_materials_summary.csv")   # GNoME's own summary table
    zip_path = os.path.join(gnome_dir, "by_composition.zip")                 # GNoME's own CIF archive
    filtered = _screen.filter_candidates(summary_path, CONFIG["bandgap_lo"], CONFIG["bandgap_hi"])   # reused, unchanged funnel
    if CONFIG["limit"]:
        filtered = filtered.head(CONFIG["limit"])   # keep only the first N rows, for a smoke test
        print(f"  --limit {CONFIG['limit']}: smoke-testing on the first {len(filtered)} candidates only")

    # ---- featurise (cached across runs, see module docstring) ---------------
    if os.path.exists(cache_path) and not CONFIG["limit"]:
        print(f"\nLoading cached graphs from {cache_path} (delete it to force a rebuild)...")
        blob = torch.load(cache_path, weights_only=False)   # {"ids": [...], "graphs": [...], "meta": DataFrame}
        dataset = _screen._predict.PredictionSet(blob["graphs"], blob["ids"])   # rebuild the same lightweight Dataset wrapper
        meta = blob["meta"]   # the cached metadata table (material_id, formula, Number of Atoms, Volume, Density, mass)
        print(f"  {len(dataset)} cached crystals")
    else:
        print("\nFeaturising candidate structures (reading CIFs directly from the zip)...")
        dataset, meta = _screen.extract_and_featurise(
            filtered, zip_path,
            atom_init_path=os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json"),
            max_num_nbr=CONFIG["max_num_nbr"], radius=CONFIG["radius"], step=CONFIG["step"])
        if not CONFIG["limit"]:   # only cache a full run - a --limit smoke test cache would silently truncate later full runs
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)   # ensure gnome_data/ exists (it always does, but cheap to guard)
            torch.save({"ids": dataset.ids, "graphs": dataset.graphs, "meta": meta}, cache_path)   # persist for next time
            print(f"  cached to {cache_path}")

    # Drop rows with a corrupted upstream material_id BEFORE they become a
    # dict/DataFrame key. Some GNoME rows carry a literal "NaN" string or a
    # garbled 30+ digit number as MaterialId (real upstream defect, verified
    # against the raw stable_materials_summary.csv - see the
    # materials-dft-databases memory). Python treats each NaN float as a
    # DISTINCT dict key, but pandas treats every NaN in an index as equal to
    # every other NaN - so leaving these in makes predict_gamma_model()'s
    # per-seed dict grow one entry per corrupted row while meta.loc[ids] on
    # the same NaN key returns every corrupted meta row at once, and the two
    # counts silently diverge. Cheaper to drop them once, here, than to make
    # every downstream keyed lookup NaN-safe.
    bad_id = meta["material_id"].isna() | (meta["material_id"].astype(str).str.len() > 15)   # same corruption test used in 13/15's id_unverifiable flag
    if bad_id.any():
        print(f"  dropping {int(bad_id.sum())} candidates with a corrupted upstream "
             f"material_id (literal 'NaN' or a garbled long number)")
        keep_ids = set(meta.loc[~bad_id, "material_id"])   # the subset of ids that are safe to use as keys
        keep_idx = [i for i, mid in enumerate(dataset.ids) if mid in keep_ids]   # positions in dataset.ids/graphs to retain
        dataset = _screen._predict.PredictionSet(
            [dataset.graphs[i] for i in keep_idx], [dataset.ids[i] for i in keep_idx])   # rebuild the Dataset over the kept positions only
        meta = meta[~bad_id].reset_index(drop=True)   # matching drop on the metadata table

    meta = meta.set_index("material_id")   # index by id so .loc[ids] lookups in ensemble_and_score() work directly

    # ---- run the round-9 ensemble --------------------------------------------
    print("\nRunning the round-9 gamma-head ensemble "
         f"({CONFIG['r9_tags']})...")
    sample_shapes = (dataset.graphs[0][0].shape[-1], dataset.graphs[0][1].shape[-1])   # (atom_fea_len, nbr_fea_len) from one real graph
    per_seed_preds = []   # one {material_id: (log10_G, log10_ratio, log10_gamma)} dict per seed
    for tag in CONFIG["r9_tags"].split(","):   # e.g. "r9_full_s42", "r9_full_s1", "r9_full_s2"
        model, normalizer, cfg = load_gamma_model(tag.strip(), results_dir, sample_shapes)   # rebuild this seed's trained model
        preds = predict_gamma_model(model, normalizer, dataset)   # run it over every cached graph
        per_seed_preds.append(preds)   # keep for the cross-seed ensemble step
        print(f"  {tag.strip():<14} scored {len(preds)} crystals")

    print("\nCombining seeds into an ensemble point estimate + Monte Carlo band...")
    scored = ensemble_and_score(per_seed_preds, meta)   # the new pipeline's own K/G/gamma/kappa table, indexed by material_id

    # ---- join everything into one table --------------------------------------
    out = meta.join(scored, how="inner")   # meta's structural columns + this script's new K/G/gamma/kappa columns
    out = out.reset_index()   # material_id back to a normal column, for CSV writing and downstream joins

    # Bring in the OLD (matbench-K/G, derived-gamma) pipeline's kappa for
    # direct comparison, purely as a joined-in reference column - never used
    # to filter or threshold anything in THIS script.
    if os.path.exists(os.path.join(PROJECT_ROOT, CONFIG["old_screen_csv"])):
        old = pd.read_csv(os.path.join(PROJECT_ROOT, CONFIG["old_screen_csv"]),
                          usecols=["material_id", "Kappa_cal (W m-1 K-1)"],
                          dtype={"material_id": str})   # only the one column needed for comparison, id kept as string
        old = old.rename(columns={"Kappa_cal (W m-1 K-1)": "Kappa_cal_derived_matbench"})   # explicit name, not a bare reuse
        out = out.merge(old, on="material_id", how="left")   # left join: keep every round-9-scored row even if 13 lacked it

    # Same corrupted-ID hygiene flag as 13/15's outputs (see the
    # script-style-config-block-heavy-comments memory's 2026-09-02 update) -
    # some upstream GNoME rows carry a literal "NaN" string or a garbled
    # 30+ digit number as MaterialId; flag rather than silently drop them.
    out["id_unverifiable"] = out["material_id"].isna() | (out["material_id"].astype(str).str.len() > 15)

    # ---- oxide detection, reused from 15_filter_oxides.py --------------------
    out["has_oxygen"] = out["formula"].apply(_oxide.has_oxygen)   # True/False per candidate
    out.loc[out["has_oxygen"], "stoichiometry_pattern"] = out.loc[out["has_oxygen"], "formula"].apply(_oxide.classify_oxide)   # only classify actual oxides

    out = out.sort_values("Kappa_r9_p95").reset_index(drop=True)   # pessimistic-bound ranking, same convention as the DFT-candidate CSVs

    all_path = os.path.join(results_dir, "39_gnome_screen_all_gamma.csv")
    out.to_csv(all_path, index=False)
    print(f"\nWrote {all_path} ({len(out)} scored candidates, pre-threshold)")

    candidates = out[out["Kappa_r9_gamma"] <= CONFIG["kappa_threshold"]].copy()   # rows clearing the kappa threshold on the NEW pipeline
    cand_path = os.path.join(results_dir, "39_gnome_screen_candidates_gamma.csv")
    candidates.to_csv(cand_path, index=False)
    print(f"kappa_L <= {CONFIG['kappa_threshold']} W/m/K (round-9 gamma): "
         f"{len(candidates)} candidates")
    print(f"Wrote {cand_path}")

    oxide_low_kappa = candidates[candidates["has_oxygen"]].copy()   # of THOSE candidates, the ones that are also oxides
    oxide_path = os.path.join(results_dir, "39_gnome_oxide_low_kappa_candidates_gamma.csv")
    oxide_low_kappa.to_csv(oxide_path, index=False)
    print(f"  of which {len(oxide_low_kappa)} are oxides")
    print(f"Wrote {oxide_path}")

    # ---- headline comparison against the old (derived-gamma) screen ---------
    if "Kappa_cal_derived_matbench" in out.columns:
        old_candidates_path = os.path.join(results_dir, "13_gnome_screen_candidates.csv")
        if os.path.exists(old_candidates_path):
            old_ids = set(pd.read_csv(old_candidates_path, dtype={"material_id": str})["material_id"])   # old pipeline's candidate set
            new_ids = set(candidates["material_id"])   # this script's candidate set
            both = old_ids & new_ids   # candidates both pipelines agree clear the threshold
            print("\n" + "=" * 78)
            print("  COMPARISON: derived-gamma/matbench screen vs trained-gamma/AFLOW screen")
            print("=" * 78)
            print(f"  old (derived, matbench):  {len(old_ids)} candidates")
            print(f"  new (predicted, AFLOW) :  {len(new_ids)} candidates")
            print(f"  in both                :  {len(both)}  "
                 f"({100 * len(both) / max(1, len(old_ids | new_ids)):.1f}% jaccard overlap)")
            valid = out.dropna(subset=["Kappa_cal_derived_matbench", "Kappa_r9_gamma"])   # rows scored by both pipelines
            valid = valid[valid["Kappa_cal_derived_matbench"] > 0]   # guard log10 below against a zero/negative value
            corr = np.corrcoef(np.log10(valid["Kappa_cal_derived_matbench"]),
                               np.log10(valid["Kappa_r9_gamma"]))[0, 1]   # Pearson r between the two pipelines' log10(kappa)
            print(f"  log10(kappa) correlation, old vs new, over {len(valid)} shared rows: r={corr:.3f}")


if __name__ == "__main__":
    main()
