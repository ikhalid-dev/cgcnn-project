#!/usr/bin/env python3
"""
Shared prediction engine behind predict.py (CLI) and predict_app.py (web app).
================================================================================

No CLI or UI code lives here on purpose - both entry points call the exact
same two functions, so they can never quietly disagree with each other.

WHAT THIS DOES, AND WHAT IT DOESN'T
------------------------------------
Nothing here is a new model or a new physics formula. Every prediction comes
from calling the already-trained, already-validated pipeline this project
built earlier: scripts/cgcnn/04_predict_moduli.py's CGCNN ensemble,
scripts/alignn/15_alignn_predict_moduli.py's ALIGNN model, and
scripts/cgcnn/07_predict_kappa.py's Slack-model physics - imported and called
directly (same importlib pattern this project already uses everywhere for
cross-directory reuse, e.g. 15/16 reusing 01b/02/07/08), not re-implemented.

The one genuinely new thing is confidence_tier(): PREDICT_NEW_CIFS.txt's
Section 5 already explains how to manually judge a prediction (agreement
between CGCNN and ALIGNN, the CGCNN ensemble's own spread, provenance).
confidence_tier() just automates that reading instead of asking a human to
eyeball 2-3 CSVs side by side every time. It's a heuristic, not a calibrated
statistic - see its own docstring for exactly what it checks and why those
thresholds.

WHY MODEL LOADING IS SEPARATE FROM PREDICTION
------------------------------------------------
load_models() reads 5 checkpoints off disk (3 CGCNN + 2 ALIGNN) - the slow
part, a few seconds. predict_batch() takes an already-loaded ModelBundle and
only does the fast part (featurise + forward pass) - so predict.py (one-shot
CLI) can call both once, and predict_app.py (a long-lived server process)
can call load_models() once at startup (wrapped in st.cache_resource) and
predict_batch() on every upload without reloading weights each time.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from importlib import import_module

# torch first - see cgcnn_scratch/data.py for why (MKL/libiomp5 duplicate
# OpenMP runtime segfault if numpy/pandas import first in this env). Every
# entry point that imports this module inherits the same requirement: torch
# must be the first thing imported in the PROCESS, not just in this file.
import torch  # noqa: F401

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "alignn"))

_moduli = import_module("04_predict_moduli")
_kappa = import_module("07_predict_kappa")
_alignn_moduli = import_module("15_alignn_predict_moduli")
_train = import_module("02_train")  # pick_device()

RESULTS_CGCNN = os.path.join(PROJECT_ROOT, "results", "cgcnn")
RESULTS_ALIGNN = os.path.join(PROJECT_ROOT, "results", "alignn")
DATA_FULL = os.path.join(PROJECT_ROOT, "data_full")
ATOM_INIT = os.path.join(PROJECT_ROOT, "cgcnn_scratch", "atom_init.json")

K_TAGS = ["K_VRH_full", "K_VRH_s1", "K_VRH_s2"]
G_TAGS = ["G_VRH_full", "G_VRH_s1", "G_VRH_s2"]


@dataclass
class ModelBundle:
    """Everything predict_batch() needs, loaded once by load_models()."""
    k_models: list         # [(model, normalizer), ...] the CGCNN K ensemble
    g_models: list         # same, for G
    k_ckpt: dict           # one representative checkpoint, for provenance lookup
    cache_config: dict     # featurisation settings the CGCNN models were trained with
    labels: pd.DataFrame   # matbench labels, for provenance/DFT-reference lookup
    alignn_k_model: object
    alignn_g_model: object
    alignn_settings: dict
    alignn_device: object


def load_models(device="auto"):
    """Load the 3-seed CGCNN ensemble (K and G) and both ALIGNN models.

    Tags are the project's own established ensemble (see
    PREDICT_NEW_CIFS.txt's Section 2) - not configurable here, because
    07_predict_kappa.py's uncertainty propagation requires the ensemble
    spread columns a single-model run doesn't produce.
    """
    k_loaded = [_moduli.load_model(t, RESULTS_CGCNN) for t in K_TAGS]
    g_loaded = [_moduli.load_model(t, RESULTS_CGCNN) for t in G_TAGS]

    cache_config = torch.load(os.path.join(DATA_FULL, "graphs.pt"),
                              weights_only=False)["config"]
    labels = pd.read_csv(os.path.join(DATA_FULL, "labels.csv"))

    torch_device = _train.pick_device(device)
    k_dir = os.path.join(RESULTS_ALIGNN, "alignn_bulk_modulus_kv")
    g_dir = os.path.join(RESULTS_ALIGNN, "alignn_shear_modulus_gv")
    alignn_k_model, k_settings = _alignn_moduli.load_alignn_target(k_dir, torch_device)
    alignn_g_model, g_settings = _alignn_moduli.load_alignn_target(g_dir, torch_device)
    if k_settings != g_settings:
        raise RuntimeError(f"ALIGNN bulk/shear graph settings disagree: "
                           f"{k_settings} vs {g_settings}")

    return ModelBundle(
        k_models=[(m, n) for m, n, _ in k_loaded],
        g_models=[(m, n) for m, n, _ in g_loaded],
        k_ckpt=k_loaded[0][2],
        cache_config=cache_config,
        labels=labels,
        alignn_k_model=alignn_k_model,
        alignn_g_model=alignn_g_model,
        alignn_settings=k_settings,
        alignn_device=torch_device,
    )


def confidence_tier(row):
    """How much to trust one crystal's prediction - a heuristic, not a
    calibrated statistic. Reads columns predict_batch() already computed.

    unverified   ALIGNN (or CGCNN) produced no usable number for this
                 crystal at all - there's nothing to cross-check against.
    low          the two models disagree by more than 50% on kappa_L, OR
                 either model's structural inputs looked implausible
                 (Gruneisen parameter outside roughly 0-5, a sane range for
                 ordinary solids).
    medium       20-50% disagreement, OR the CGCNN ensemble disagrees with
                 ITSELF by more than the noise floor documented in
                 PREDICT_NEW_CIFS.txt (>0.15 log10, ~40% relative spread
                 across the 3 seeds) even if ALIGNN happens to land close.
    high         <=20% disagreement between CGCNN and ALIGNN's kappa_L, and
                 the CGCNN ensemble agrees with itself. This threshold is
                 not arbitrary - PREDICT_NEW_CIFS.txt's own Section 5b
                 already establishes ~15-20% cross-model agreement as what
                 this pipeline looks like when it's working correctly, based
                 on the project's own measured ~15% accuracy floor.

    If `provenance` isn't "unseen", a real DFT reference already exists for
    this crystal (K_VRH_dft/G_VRH_dft columns) - report that directly
    alongside the tier rather than folding it in; real ground truth is
    strictly more informative than a heuristic and shouldn't be averaged
    away into one label.
    """
    kc, ka = row.get("Kappa_cal_cgcnn"), row.get("Kappa_cal_alignn")
    if (pd.isna(kc) or pd.isna(ka) or kc is None or ka is None
            or kc <= 0 or ka <= 0):
        return "unverified", float("nan")

    for g in (row.get("Gruneisen_cgcnn"), row.get("Gruneisen_alignn")):
        if g is not None and not pd.isna(g) and not (0 < g < 5):
            return "low", float("nan")

    disagreement_pct = (10 ** abs(np.log10(kc) - np.log10(ka)) - 1) * 100

    if disagreement_pct > 50:
        tier = "low"
    elif disagreement_pct > 20:
        tier = "medium"
    else:
        tier = "high"

    ensemble_spread = max(row.get("K_VRH_spread_log10", 0) or 0,
                          row.get("G_VRH_spread_log10", 0) or 0)
    if tier == "high" and ensemble_spread > 0.15:
        tier = "medium"

    return tier, disagreement_pct


def predict_batch(cif_dir, models):
    """Run the full CGCNN + ALIGNN + kappa_L pipeline over every .cif in
    cif_dir. Returns (DataFrame, failures) - failures is
    {"cgcnn": [(id, reason), ...], "alignn": [(id, reason), ...]}, kept
    separate rather than merged because the two models can fail on
    different crystals for different reasons.
    """
    dataset, meta, failed_cgcnn = _moduli.build_prediction_graphs(
        cif_dir, ATOM_INIT, models.cache_config["max_num_nbr"],
        models.cache_config["radius"], models.cache_config["step"])

    k_pred, k_spread = _moduli.predict_ensemble(models.k_models, dataset)
    g_pred, g_spread = _moduli.predict_ensemble(models.g_models, dataset)

    df = meta.copy()
    df["K_VRH_pred"] = df.material_id.map(k_pred)
    df["G_VRH_pred"] = df.material_id.map(g_pred)
    df["K_VRH_spread_log10"] = df.material_id.map(k_spread)
    df["G_VRH_spread_log10"] = df.material_id.map(g_spread)

    provenance, reference = _moduli.build_provenance(DATA_FULL, models.k_ckpt, models.labels)
    df["provenance"] = df.material_id.map(provenance).fillna("unseen")
    df["K_VRH_dft"] = df.material_id.map(lambda m: reference.get(m, {}).get("K_VRH", np.nan))
    df["G_VRH_dft"] = df.material_id.map(lambda m: reference.get(m, {}).get("G_VRH", np.nan))

    alignn_df, failed_alignn = _alignn_moduli.build_predictions(
        cif_dir, models.alignn_k_model, models.alignn_g_model,
        models.alignn_settings, models.alignn_device)
    alignn_df = alignn_df.rename(columns={"K_VRH_pred": "K_VRH_alignn",
                                          "G_VRH_pred": "G_VRH_alignn"})
    df = df.merge(alignn_df[["material_id", "K_VRH_alignn", "G_VRH_alignn"]],
                  on="material_id", how="left")

    # Structural quantities depend only on the CIF, not on which model
    # predicted K/G - computed once, shared by both models' physics below.
    structure = _kappa.structure_quantities(cif_dir, df.material_id)
    df = df.merge(structure, on="material_id", how="inner")

    point_cgcnn = _kappa.slack_physics(
        df["K_VRH_pred"].values, df["G_VRH_pred"].values,
        df["Volume (A3)"].values, df["Density (g cm-3)"].values,
        df["Atomic mass (amu)"].values, df["Number of Atoms"].values)
    df["Kappa_cal_cgcnn"] = point_cgcnn["kappa_cal"]
    df["Gruneisen_cgcnn"] = point_cgcnn["gruneisen"]

    # Only crystals ALIGNN actually produced a number for get its physics -
    # slack_physics() is shape-agnostic but NaN in means NaN out regardless.
    has_alignn = df["K_VRH_alignn"].notna()
    df["Kappa_cal_alignn"] = np.nan
    df["Gruneisen_alignn"] = np.nan
    if has_alignn.any():
        sub = df[has_alignn]
        point_alignn = _kappa.slack_physics(
            sub["K_VRH_alignn"].values, sub["G_VRH_alignn"].values,
            sub["Volume (A3)"].values, sub["Density (g cm-3)"].values,
            sub["Atomic mass (amu)"].values, sub["Number of Atoms"].values)
        df.loc[has_alignn, "Kappa_cal_alignn"] = point_alignn["kappa_cal"]
        df.loc[has_alignn, "Gruneisen_alignn"] = point_alignn["gruneisen"]

    # Monte Carlo interval from the CGCNN ensemble's own K/G spread - ALIGNN
    # has no equivalent (a single trained model, not an ensemble; same
    # reasoning scripts/alignn/16_alignn_predict_kappa.py documents for why
    # it skips this step too).
    mc = _kappa.monte_carlo_kappa(df, n_samples=2000, seed=0)
    df["Kappa_cal_cgcnn_p05"] = mc["Kappa_cal_p05"]
    df["Kappa_cal_cgcnn_p95"] = mc["Kappa_cal_p95"]

    tiers = df.apply(confidence_tier, axis=1, result_type="expand")
    df["confidence"], df["model_disagreement_pct"] = tiers[0], tiers[1]

    df = df.sort_values("Kappa_cal_cgcnn").reset_index(drop=True)
    failures = {"cgcnn": failed_cgcnn, "alignn": failed_alignn}
    return df, failures
