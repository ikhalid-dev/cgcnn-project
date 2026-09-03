#!/usr/bin/env python3
"""
STEP 28 - Screen ALL small-cell candidates for low-kappa_L, with ALIGNN.
================================================================================

    python scripts/cgcnn/28_alignn_small_cell_filter.py

WHAT THIS IS
-------------
27_alignn_dft_shortlist.py ran ALIGNN only on the 267 candidates CGCNN had
ALREADY flagged as low-kappa - a cross-check of CGCNN's own picks, not an
independent screen. This script runs ALIGNN on the FULL small-cell
population instead - all 1,215 candidates from 19_small_cells.py, low-kappa
or not by CGCNN's own scoring - and filters by ALIGNN's OWN
kappa_L <= 1 W/m/K threshold. That is a genuinely different question:
27 asked "does ALIGNN agree with CGCNN's picks", this asks "what would
ALIGNN itself have picked", which can surface candidates CGCNN missed
entirely (predicted kappa_L > 1, so never made the original 267-row list)
as well as reject ones CGCNN got right.

REUSE, NOT REIMPLEMENTATION
--------------------------------
Same ALIGNN loading/inference/physics path as 27_alignn_dft_shortlist.py
(load_alignn_target(), predict_pair() from scripts/alignn/15_alignn_predict_moduli.py;
slack_physics() from scripts/cgcnn/07_predict_kappa.py) - reused via
importlib exactly the same way, just applied to a larger, differently-scoped
population. No new inference logic.

OUTPUTS
--------
    gnome_small_cells_alignn.csv           all 1,215, ALIGNN columns added
    gnome_small_cells_alignn_low_kappa.csv ALIGNN_Kappa_cal <= 1 subset (general)
    gnome_small_cells_alignn_oxides.csv    oxide-containing rows of the above

Also reports the four-way breakdown against CGCNN's own 267-candidate
pick: agree (both say low-kappa), CGCNN-only (CGCNN said yes, ALIGNN says
no - the same disagreement direction 27 already found), and ALIGNN-only
(ALIGNN says yes, CGCNN said no - candidates the CGCNN-first pipeline never
surfaced at all, only visible by screening independently rather than only
cross-checking).
"""

import os  # path joining and filesystem path handling
import sys  # sys.path manipulation, needed to import sibling scripts by number-prefixed filename
import time  # wall-clock timing for the inference-rate printout
import warnings  # used below to silence noisy third-party warnings
import zipfile  # reading CIF files directly out of the GNoME archive without extracting it
from importlib import import_module  # imports a module whose filename starts with a digit (not a valid identifier for a normal `import`)

# torch first - see cgcnn_scratch/data.py for why.
import torch  # noqa: F401

import numpy as np  # NaN placeholders for failed predictions
import pandas as pd  # CSV read/write and the DataFrame filtering operations below
from pymatgen.core import Composition, Structure  # Composition for the oxide check, Structure for parsing CIF text

warnings.filterwarnings("ignore")  # suppresses warning spam from pymatgen/ALIGNN during inference

# walk up three directories from this file (scripts/cgcnn/28_...py -> scripts/cgcnn -> scripts -> project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")
ALIGNN_RESULTS = os.path.join(PROJECT_ROOT, "results", "alignn")
GNOME_DIR = os.path.join(PROJECT_ROOT, "gnome_data")
SMALL_CELLS_CSV = os.path.join(RESULTS, "19_gnome_screen_small_cells.csv")  # written by 19_small_cells.py
OUT_ALL_CSV = os.path.join(RESULTS, "28_gnome_small_cells_alignn.csv")  # read by 29_alignn_agreement_dft_candidates.py
OUT_LOW_KAPPA_CSV = os.path.join(RESULTS, "28_gnome_small_cells_alignn_low_kappa.csv")
OUT_OXIDES_CSV = os.path.join(RESULTS, "28_gnome_small_cells_alignn_oxides.csv")
KAPPA_THRESHOLD = 1.0  # W/m/K cutoff used for ALIGNN's own low-kappa flag

sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "alignn"))  # make scripts/alignn/ importable
_alignn_predict = import_module("15_alignn_predict_moduli")  # the module providing load_alignn_target() and predict_pair()
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))  # make scripts/cgcnn/ importable (for the two imports below)
_kappa = import_module("07_predict_kappa")  # the module providing slack_physics()
_train = import_module("02_train")  # the module providing pick_device()


def has_oxygen(formula):
    """Same pymatgen-based check as 15_filter_oxides.py - not a substring
    search, see that script's docstring for why."""
    try:
        return "O" in {str(el) for el in Composition(formula).elements}  # parse the formula into elements, then check membership by element SYMBOL, not by substring of the formula string
    except Exception:
        return False  # a formula pymatgen cannot parse is treated as "not an oxide" rather than crashing the whole run


def main():
    print("=== Screening ALL small-cell candidates for low-kappa_L, with ALIGNN ===\n")
    df = pd.read_csv(SMALL_CELLS_CSV)  # load the full small-cell population (all 1,215, not just CGCNN's low-kappa picks)
    print(f"Loaded {len(df)} small-cell (<10 atom) candidates from {SMALL_CELLS_CSV}")
    print(f"  {int(df.is_low_kappa_candidate.sum())} already confirmed low-kappa by CGCNN")

    # GNoME's own summary file maps MaterialId -> Composition, which is the key the CIF zip is indexed by
    summary = pd.read_csv(os.path.join(GNOME_DIR, "stable_materials_summary.csv"),
                          usecols=["MaterialId", "Composition"])
    merged = df.merge(summary, left_on="material_id", right_on="MaterialId", how="left")  # left join: keep every row even if no GNoME match is found
    missing = merged["Composition"].isna().sum()  # count of rows with no matching Composition after the join
    if missing:
        print(f"  {missing}/{len(merged)} have no matching GNoME summary row - "
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

        if (i + 1) % 200 == 0:  # print a progress line every 200 candidates (larger population than 27, so a coarser interval)
            rate = (i + 1) / (time.time() - start)  # candidates processed per second so far
            print(f"  {i + 1}/{len(merged)} ({rate:.1f}/s, "
                 f"~{(len(merged) - i - 1) / rate:.0f}s left)")  # estimated seconds remaining at the current rate

    n_ok = len(merged) - len(failed) - missing  # rows that produced a real prediction (not missing a CIF, not a failed inference)
    print(f"Predicted {n_ok}/{len(merged)} in {(time.time() - start):.1f}s")
    if failed:
        print(f"  {len(failed)} failed during ALIGNN inference, e.g. {failed[:3]}")  # show only the first 3 failures, not the whole list

    df["ALIGNN_K_VRH_pred"] = k_preds  # attach the new column - list order matches merged's row order, which matches df's
    df["ALIGNN_G_VRH_pred"] = g_preds

    point = _kappa.slack_physics(
        df["ALIGNN_K_VRH_pred"].values, df["ALIGNN_G_VRH_pred"].values,
        df["Volume (A3)"].values, df["Density (g cm-3)"].values,
        df["Atomic mass (amu)"].values, df["Number of Atoms"].values)  # reuse 07's Slack-model formula with ALIGNN's moduli, this table's own structural columns
    df["ALIGNN_Kappa_cal"] = point["kappa_cal"]  # pull the scalar kappa field out of slack_physics()'s returned dict
    df["ALIGNN_is_low_kappa_candidate"] = df["ALIGNN_Kappa_cal"] <= KAPPA_THRESHOLD  # ALIGNN's own low-kappa flag, independent of CGCNN's

    # predict_pair() floors a raw negative/near-zero modulus output to 1e-3
    # GPa (15_alignn_predict_moduli.py's own guard against log10(<=0) further
    # downstream) - a real value this project's data never actually produces
    # (min G_VRH_pred across the whole 33,323-candidate CGCNN screen is
    # 0.57 GPa, two orders of magnitude above this floor), so landing on it
    # means ALIGNN's raw regression output was degenerate for that specific
    # structure - an extrapolation failure dressed up as a suspiciously good
    # number, not a genuine independent low-kappa confirmation. Flagged
    # explicitly rather than silently trusted.
    FLOOR = 1e-3
    df["alignn_prediction_reliable"] = ~((df["ALIGNN_K_VRH_pred"] <= FLOOR * 1.1) |
                                         (df["ALIGNN_G_VRH_pred"] <= FLOOR * 1.1))  # False if either modulus landed on (or just above) the floor value
    n_floored = int((~df["alignn_prediction_reliable"] & df["ALIGNN_Kappa_cal"].notna()).sum())  # count of rows flagged unreliable that DID get a kappa value
    if n_floored:
        print(f"\n  {n_floored} candidate(s) had ALIGNN hit the regression floor "
             f"(degenerate prediction, not a real low-kappa confirmation) - "
             f"flagged via alignn_prediction_reliable=False, e.g.: "
             f"{df.loc[~df['alignn_prediction_reliable'] & df['ALIGNN_Kappa_cal'].notna(), 'formula'].tolist()[:5]}")  # show up to 5 example formulas

    df.to_csv(OUT_ALL_CSV, index=False)  # write the full population with ALIGNN columns attached, no pandas row-index column
    print(f"\nWrote {OUT_ALL_CSV} ({len(df)} rows)")

    valid = df.dropna(subset=["ALIGNN_Kappa_cal"])  # drop rows ALIGNN never scored at all (missing CIF or failed inference)
    valid = valid[valid["alignn_prediction_reliable"]]  # further drop rows flagged as a degenerate/floored prediction
    cgcnn_yes = valid["is_low_kappa_candidate"]  # CGCNN's own low-kappa flag (computed upstream in 19_small_cells.py)
    alignn_yes = valid["ALIGNN_is_low_kappa_candidate"]  # ALIGNN's low-kappa flag, computed just above
    both = valid[cgcnn_yes & alignn_yes]  # both models say low-kappa
    cgcnn_only = valid[cgcnn_yes & ~alignn_yes]  # CGCNN says yes, ALIGNN says no
    alignn_only = valid[~cgcnn_yes & alignn_yes]  # ALIGNN says yes, CGCNN said no - a genuinely new find
    neither = valid[~cgcnn_yes & ~alignn_yes]  # both models agree this candidate is NOT low-kappa

    print(f"\n{'=' * 60}")
    print(f"Agreement breakdown ({len(valid)} candidates scored by both models)")
    print(f"{'=' * 60}")
    print(f"  Both agree low-kappa       : {len(both):4d}")
    print(f"  CGCNN yes, ALIGNN no       : {len(cgcnn_only):4d}  (CGCNN picks ALIGNN rejects)")
    print(f"  ALIGNN yes, CGCNN no       : {len(alignn_only):4d}  (new finds CGCNN missed entirely)")
    print(f"  Neither                   : {len(neither):4d}")

    low_kappa = df[(df["ALIGNN_is_low_kappa_candidate"] == True) &  # noqa: E712
                  df["alignn_prediction_reliable"]].copy()  # ALIGNN-confirmed low-kappa AND not a degenerate/floored prediction
    low_kappa = low_kappa.sort_values("ALIGNN_Kappa_cal").reset_index(drop=True)  # ascending kappa - lowest (most promising) first, index renumbered from 0
    low_kappa.to_csv(OUT_LOW_KAPPA_CSV, index=False)
    print(f"\nWrote {OUT_LOW_KAPPA_CSV} ({len(low_kappa)} rows, ALIGNN-confirmed "
         f"low-kappa small cells, sorted by ALIGNN_Kappa_cal ascending)")

    oxides = low_kappa[low_kappa["formula"].apply(has_oxygen)].copy()  # apply the oxide check to every formula in the low-kappa subset
    oxides.to_csv(OUT_OXIDES_CSV, index=False)
    print(f"Wrote {OUT_OXIDES_CSV} ({len(oxides)} rows, "
         f"{len(oxides) / len(low_kappa) * 100:.1f}% of the ALIGNN low-kappa small cells)")

    if len(alignn_only):
        print(f"\nTop 5 ALIGNN-only finds (CGCNN never flagged these at all):")
        top_new = alignn_only.sort_values("ALIGNN_Kappa_cal").head(5)  # the 5 lowest-kappa ALIGNN-only finds
        for _, row in top_new.iterrows():
            print(f"  {row.formula:16s} CGCNN={row['Kappa_cal (W m-1 K-1)']:.3f} "
                 f"(missed) ALIGNN={row.ALIGNN_Kappa_cal:.3f}  "
                 f"{int(row['Number of Atoms'])} atoms")


if __name__ == "__main__":  # only run main() when executed as a script, not when imported as a module
    main()
