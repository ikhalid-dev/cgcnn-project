#!/usr/bin/env python3
"""
STEP 27 - Run ALIGNN on the small-cell DFT shortlist, as a second opinion.
================================================================================

    python scripts/cgcnn/27_alignn_dft_shortlist.py

WHAT THIS IS
-------------
25_calibration_check.py found the CGCNN ensemble's uncertainty is badly
overconfident, and 26's recalibration showed almost none of the DFT
shortlist's confidence labels survive. The natural follow-up question this
answers: does ALIGNN - a second, independently-trained architecture that
sees bond angles as well as bond lengths, and already measured as more
accurate on the matbench test set (docs/method.tex, Section 9) - AGREE with
CGCNN on these specific 267 small-cell, low-kappa candidates? Two models
sharing nothing but training data landing on similar K/G/kappa_L values for
a candidate is real evidence beyond either model's own (miscalibrated)
uncertainty interval; landing far apart is a warning sign no single
ensemble-spread number would have caught.

WHY THIS IS NEW INFERENCE, NOT A CSV MERGE
------------------------------------------------
ALIGNN was previously only ever run on the 1,213-crystal PINK prediction
set (scripts/alignn/15_alignn_predict_moduli.py) - a completely different,
smaller collection of Materials Project CIFs. It has never seen any of the
33,323 GNoME screen candidates before, so there is no existing ALIGNN
prediction file to merge from for this shortlist. This script reads each
shortlist candidate's CIF directly from the GNoME zip (same
Composition-keyed lookup 13_screen_gnome.py uses - checked, not assumed:
265 of 267 material_ids resolve to a CIF in the archive; the 2 that don't
are the same 2 rows already flagged as unmatched in
16_spacegroups_oxides.py's space-group join, a pre-existing upstream gap,
not a new one), builds the ALIGNN graph, and predicts K and G with the
already-trained models in results/alignn/.

REUSE, NOT REIMPLEMENTATION
--------------------------------
load_alignn_target() and predict_pair() are imported unchanged from
scripts/alignn/15_alignn_predict_moduli.py (via importlib, the same
numeric-prefixed-filename workaround used throughout this project).
slack_physics() is imported unchanged from scripts/cgcnn/07_predict_kappa.py
to turn ALIGNN's (K, G) into its own Kappa_cal, reusing this shortlist's
ALREADY-COMPUTED structural quantities (Volume, Density, Atomic mass,
Number of Atoms) rather than re-deriving them from the CIF a second time -
those depend only on the structure, not on which model predicted the moduli.
No Monte Carlo interval is computed for ALIGNN's Kappa_cal: ALIGNN here is
one trained model per target, not a 3-member ensemble, so - same reasoning
as 15_alignn_predict_moduli.py's own module docstring - there is no spread
to propagate honestly.

Overwrites gnome_small_cell_dft_ranking.csv in place with three new columns
(ALIGNN_K_VRH_pred, ALIGNN_G_VRH_pred, ALIGNN_Kappa_cal). Re-run
24_small_cell_oxides_for_dft.py afterward to carry these columns through
into its 24-row oxide subset unchanged - no code change needed there, since
it only filters and copies columns.
"""

import os  # path joining and filesystem path handling
import sys  # sys.path manipulation, needed to import sibling scripts by number-prefixed filename
import time  # wall-clock timing for the inference-rate printout
import warnings  # used below to silence noisy third-party deprecation warnings
import zipfile  # reading CIF files directly out of the GNoME archive without extracting it
from importlib import import_module  # imports a module whose filename starts with a digit (not a valid identifier for a normal `import`)

# torch first - see cgcnn_scratch/data.py for why.
import torch  # noqa: F401

import numpy as np  # log10/corrcoef for the CGCNN-vs-ALIGNN agreement stats, and NaN placeholders for failed predictions
import pandas as pd  # CSV read/write and the DataFrame merge/groupby operations below
from pymatgen.core import Structure  # parses the CIF text pulled from the zip into a Structure object

warnings.filterwarnings("ignore")  # suppresses warning spam from pymatgen/ALIGNN during inference

# walk up three directories from this file (scripts/cgcnn/27_...py -> scripts/cgcnn -> scripts -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")
ALIGNN_RESULTS = os.path.join(PROJECT_ROOT, "results", "alignn")
GNOME_DIR = os.path.join(PROJECT_ROOT, "gnome_data")
# 22_gnome_small_cell_dft_ranking.csv is written by 22_best_dft_candidate.py; this script both
# reads it AND overwrites it in place with three extra ALIGNN_* columns further down.
SHORTLIST_CSV = os.path.join(RESULTS, "22_gnome_small_cell_dft_ranking.csv")

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "alignn"))  # make scripts/alignn/ importable
_alignn_predict = import_module("15_alignn_predict_moduli")  # the module providing load_alignn_target() and predict_pair()
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))  # make scripts/cgcnn/ importable (for the two imports below)
_kappa = import_module("07_predict_kappa")  # the module providing slack_physics()
_train = import_module("02_train")  # the module providing pick_device()


def main():
    print("=== Running ALIGNN on the small-cell DFT shortlist ===\n")
    shortlist = pd.read_csv(SHORTLIST_CSV)  # load the 22-produced shortlist to enrich with ALIGNN predictions
    print(f"Loaded {len(shortlist)} candidates from {SHORTLIST_CSV}")

    # GNoME's own summary file maps MaterialId -> Composition, which is the key the CIF zip is indexed by
    summary = pd.read_csv(os.path.join(GNOME_DIR, "stable_materials_summary.csv"),
                          usecols=["MaterialId", "Composition"])
    merged = shortlist.merge(summary, left_on="material_id", right_on="MaterialId", how="left")  # left join: keep every shortlist row even if no GNoME match is found
    missing = merged["Composition"].isna().sum()  # count of shortlist rows with no matching Composition after the join
    if missing:
        print(f"  {missing}/{len(merged)} have no matching GNoME summary row "
             f"(same pre-existing gap noted in 16_spacegroups_oxides.py) - "
             f"ALIGNN columns left blank for those")

    device = _train.pick_device("auto")  # cuda if available, else cpu/mps - reuses 02_train.py's own device-selection logic
    print(f"Device: {device}")
    k_dir = os.path.join(ALIGNN_RESULTS, "alignn_bulk_modulus_kv")
    g_dir = os.path.join(ALIGNN_RESULTS, "alignn_shear_modulus_gv")
    k_model, k_settings = _alignn_predict.load_alignn_target(k_dir, device)  # load the already-trained ALIGNN bulk-modulus model + its config
    g_model, g_settings = _alignn_predict.load_alignn_target(g_dir, device)  # load the already-trained ALIGNN shear-modulus model + its config
    assert k_settings == g_settings, "K/G ALIGNN configs disagree - see 15's own assertion"
    print(f"Loaded ALIGNN K/G models from {k_dir}, {g_dir}")

    zf = zipfile.ZipFile(os.path.join(GNOME_DIR, "by_composition.zip"))  # open the CIF archive for random-access reads by filename
    from jarvis.core.atoms import pmg_to_atoms  # converts a pymatgen Structure into the jarvis Atoms object ALIGNN expects

    k_preds, g_preds, failed = [], [], []  # accumulators: predicted K, predicted G, and (material_id, error) pairs for anything that fails
    start = time.time()  # wall-clock start, used for the periodic rate printout below
    for i, row in enumerate(merged.itertuples()):  # itertuples() is faster than iterrows() for read-only row access
        if pd.isna(row.Composition):  # this row had no GNoME match, so there is no CIF to look up
            k_preds.append(np.nan)
            g_preds.append(np.nan)
            continue
        try:
            cif_text = zf.read(f"by_composition/{row.Composition}.CIF").decode("utf-8")  # read the CIF bytes out of the zip and decode as text
            structure = Structure.from_str(cif_text, fmt="cif")  # parse the CIF text into a pymatgen Structure
            atoms = pmg_to_atoms(structure)  # convert to the Atoms representation ALIGNN's predict_pair() expects
            k_val, g_val = _alignn_predict.predict_pair(k_model, g_model, atoms, k_settings, device)  # run both trained models on this one structure
        except Exception as exc:  # any failure (bad CIF, graph-build error, etc.) is recorded, not fatal to the whole run
            failed.append((row.material_id, str(exc)[:70]))  # keep only the first 70 chars of the error message
            k_preds.append(np.nan)
            g_preds.append(np.nan)
            continue
        k_preds.append(max(k_val, 1e-3))  # floor at a small positive value - a zero/negative modulus would break the log-space kappa formula
        g_preds.append(max(g_val, 1e-3))

        if (i + 1) % 50 == 0:  # print a progress line every 50 candidates
            rate = (i + 1) / (time.time() - start)  # candidates processed per second so far
            print(f"  {i + 1}/{len(merged)} ({rate:.1f}/s)")

    print(f"Predicted {len(merged) - len(failed) - missing}/{len(merged)} in "
         f"{(time.time() - start):.1f}s")
    if failed:
        print(f"  {len(failed)} failed during ALIGNN inference, e.g. {failed[:3]}")  # show only the first 3 failures, not the whole list

    shortlist["ALIGNN_K_VRH_pred"] = k_preds  # attach the new column - list order matches merged's row order, which matches shortlist's
    shortlist["ALIGNN_G_VRH_pred"] = g_preds

    point = _kappa.slack_physics(
        shortlist["ALIGNN_K_VRH_pred"].values, shortlist["ALIGNN_G_VRH_pred"].values,
        shortlist["Volume (A3)"].values, shortlist["Density (g cm-3)"].values,
        shortlist["Atomic mass (amu)"].values, shortlist["Number of Atoms"].values)  # reuse 07's Slack-model formula with ALIGNN's moduli, this shortlist's own structural columns
    shortlist["ALIGNN_Kappa_cal"] = point["kappa_cal"]  # pull the scalar kappa field out of slack_physics()'s returned dict

    # Cross-model disagreement, independent of either model's own (currently
    # miscalibrated - see 25_calibration_check.py) uncertainty interval: two
    # architectures sharing nothing but training data landing far apart is
    # its own red flag, and whether they even AGREE on clearing the
    # threshold at all is the single most actionable check available before
    # spending real DFT time on a candidate.
    both_kappa = shortlist[["Kappa_cal (W m-1 K-1)", "ALIGNN_Kappa_cal"]]  # the two models' kappa columns side by side
    shortlist["cross_model_ratio"] = both_kappa.max(axis=1) / both_kappa.min(axis=1)  # row-wise max/min -> always >= 1, how far apart the two predictions are
    shortlist["both_models_clear_threshold"] = (
        (shortlist["Kappa_cal (W m-1 K-1)"] <= 1.0) & (shortlist["ALIGNN_Kappa_cal"] <= 1.0))  # True only if BOTH models predict kappa <= 1 W/m/K

    shortlist.to_csv(SHORTLIST_CSV, index=False)  # overwrite the shortlist in place with the five new columns added above
    print(f"\nWrote {SHORTLIST_CSV} ({len(shortlist)} rows, +ALIGNN_K_VRH_pred, "
         f"ALIGNN_G_VRH_pred, ALIGNN_Kappa_cal, cross_model_ratio, "
         f"both_models_clear_threshold)")

    disagree = shortlist[shortlist["both_models_clear_threshold"] == False]  # noqa: E712
    disagree = disagree.dropna(subset=["ALIGNN_Kappa_cal"])  # drop rows where ALIGNN never produced a prediction at all (missing/failed), not a real disagreement
    if len(disagree):
        print(f"\n{len(disagree)} candidates where CGCNN and ALIGNN disagree on whether "
             f"the threshold is even cleared at all (CGCNN says yes, ALIGNN says no, "
             f"or vice versa) - a concrete red flag independent of either model's own "
             f"uncertainty interval:")
        for _, row in disagree.sort_values("cross_model_ratio").iterrows():  # worst (largest ratio) disagreements last
            print(f"  {row.formula:16s} CGCNN={row['Kappa_cal (W m-1 K-1)']:.3f}  "
                 f"ALIGNN={row.ALIGNN_Kappa_cal:.3f}  ratio={row.cross_model_ratio:.2f}x")

    both = shortlist.dropna(subset=["ALIGNN_Kappa_cal"])  # rows where both models produced a usable prediction
    log_cgcnn = np.log10(both["Kappa_cal (W m-1 K-1)"])  # log10 kappa, CGCNN
    log_alignn = np.log10(both["ALIGNN_Kappa_cal"])  # log10 kappa, ALIGNN
    r = np.corrcoef(log_cgcnn, log_alignn)[0, 1]  # Pearson correlation between the two models' log-kappa predictions
    mae = (log_cgcnn - log_alignn).abs().mean()  # mean absolute difference in log10 kappa between the two models
    print(f"\nCGCNN vs. ALIGNN agreement on Kappa_cal, this shortlist "
         f"(n={len(both)}): Pearson r={r:.3f}, MAE={mae:.3f} log10")


if __name__ == "__main__":  # only run main() when executed as a script, not when imported as a module
    main()
