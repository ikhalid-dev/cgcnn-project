#!/usr/bin/env python3
"""
STEP 45 - Run the composition-only TREE baseline over the whole GNoME screen,
and annotate every downstream candidate CSV with its predictions.
================================================================================

    python scripts/cgcnn/45_screen_gnome_baseline_tree.py

    # smoke test on the first 500 GNoME rows, and skip touching the CSVs:
    python scripts/cgcnn/45_screen_gnome_baseline_tree.py --limit 500 --annotate 0

WHY THIS SCRIPT EXISTS
------------------------
43_baseline_tree_models.py established that a composition-only tree model is
NOT a straw man on this project's data - on the AFLOW set it beats the
round-9 CGCNN on every target (test MAE, log10):

    K_VRH   tree 0.0592 (xgboost)   vs  CGCNN 0.1255
    G_VRH   tree 0.1010 (RF)        vs  CGCNN 0.1521
    gamma   tree 0.0149 (RF)        vs  CGCNN (trained, not directly comparable)

That was measured on AFLOW's own 835-crystal held-out test set. It says
nothing about GNoME, which is the set this project actually screens: 33,118
filtered candidates, none of them in any training set, most of them
stoichiometries no DFT elastic calculation has ever touched. A model that
wins on held-out AFLOW can still fall apart there, and the only way to find
out is to run it and compare the two screens candidate-by-candidate.

WHY NO CIF PARSING HAPPENS HERE
---------------------------------
39_screen_gnome_gamma.py spent ~13 minutes featurising 33k CIFs into CGCNN
graphs, then cached them. This script needs NONE of that: the tree baseline's
only input is the chemical formula (see 43's docstring for why structure is
deliberately withheld), and the Slack formula's structural constants -
Number of Atoms, Volume (A3), Density (g cm-3) - are already columns in
39's own output table. So the entire GNoME pool is scored by reading one CSV
and re-featurising 33k formulas, which takes seconds. The candidate FILTER
(bandgap window, stability, no radioactives) is therefore inherited exactly,
by construction, rather than re-applied: every row 39 scored is a row this
script scores, and no other row is.

WHY THE MODELS ARE RE-TRAINED HERE INSTEAD OF LOADED
------------------------------------------------------
43 never persisted its fitted estimators - it wrote scores and per-crystal
predictions, not pickles. Re-fitting is cheap (3,894 training rows, 184
features, a few seconds) and strictly safer than a pickle: the split, the
featuriser and the hyperparameters are all imported from 43 itself, so the
models scoring GNoME here are provably the same models 43 scored on AFLOW's
test set. This script re-prints those test MAEs from its own freshly-fitted
models as a check that nothing has drifted.

WHY AFLOW AND NOT MATBENCH
----------------------------
Same reason 39 uses the round-9 checkpoint: the Slack formula needs gamma,
and gamma labels exist ONLY in AFLOW. Taking K and G from a matbench-trained
model and gamma from an AFLOW-trained one mixes two elastic-modulus
conventions inside a single kappa - the exact convention-mixing that cost
real accuracy in rounds 7/8. All three tree models here are fitted on the
same AFLOW rows, so the screen is self-consistent.

WHY BOTH RANDOM FOREST AND XGBOOST ARE REPORTED
-------------------------------------------------
On AFLOW's test set they are within 0.003 log10 of each other on all three
targets (RF wins G and gamma, xgboost wins K), which is far inside the noise
of a 835-crystal test set. Rather than pick one and hide the other, both are
carried through to kappa as complete, self-consistent families - never a
K from one and a G from the other, since the Slack formula's anharmonic term
depends on the RATIO K/G and a cross-family ratio would inherit two unrelated
models' errors (see cgcnn_scratch/joint.py for the measurement that makes
this concrete). Random forest is the HEADLINE (`Kappa_baseline`) because it
wins two targets of three and is what a materials-informatics paper would
report as "the tree baseline"; the xgboost columns sit beside it so the
choice is visible rather than buried.

WHAT GETS ANNOTATED
---------------------
The same predictions are joined by material_id into every downstream
candidate table that already exists, in place, so the tree baseline's number
sits next to the CGCNN's number in the files actually being read:

    results/cgcnn/39_gnome_screen_all_gamma.csv
    results/cgcnn/39_gnome_screen_candidates_gamma.csv
    results/cgcnn/39_gnome_oxide_low_kappa_candidates_gamma.csv
    ~/Desktop/gnome_screen_candidates_gamma.csv
    ~/Desktop/gnome_oxide_low_kappa_candidates_gamma.csv
    ~/Desktop/dft_candidates.csv
    ~/Desktop/dft_candidates_oxides.csv

The two dft_candidates files have no Volume/Density columns of their own -
they are a ranked shortlist, not a scored table - so their kappa is joined
in from this script's own GNoME scoring by material_id (verified 218/218 and
20/20 present) rather than recomputed from columns they do not have.

Annotation is IDEMPOTENT: any column this script previously added is dropped
before the join, so re-running never stacks duplicate `_x`/`_y` columns, and
the row count is asserted unchanged before anything is written.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs --------------------------------------------------------------
    # 39's own scored table IS the candidate pool: reading it inherits the
    # bandgap/stability/radioactive filter exactly, and supplies the three
    # structural constants the Slack formula needs (see module docstring).
    "gnome_scored_csv": "results/cgcnn/39_gnome_screen_all_gamma.csv",
    "atom_init_json": "cgcnn_scratch/atom_init.json",

    # ---- threshold (IDENTICAL to 13/39, so candidate counts are comparable) --
    "kappa_threshold": 1.0,   # W/m/K, the paper's own low-kappa cutoff

    # ---- output ---------------------------------------------------------------
    "results_dir": "results/cgcnn",
    "out_prefix": "45_",

    # ---- annotation -----------------------------------------------------------
    "annotate": 1,     # 0 disables in-place CSV annotation (scoring still runs)
    "desktop_dir": "/Users/mac/Desktop",

    "limit": None,     # only score the first N GNoME rows (smoke test)
}
# =============================================================================

import argparse          # CLI flag parsing, auto-generated from CONFIG below
import os                 # path joining/creation, existence checks
import sys                 # sys.path mutation and sys.exit() on fatal errors
import warnings              # suppresses noisy-but-harmless pymatgen warnings
from importlib import import_module   # loads numeric-prefixed sibling scripts by string name

# Must be set before torch/xgboost load their OpenMP runtime - see 43's own
# comment and the cgcnn-pink-environment memory: with more than one OpenMP
# thread live, xgboost segfaults nondeterministically on this machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: F401   # torch first - see cgcnn_scratch/data.py (MKL/OpenMP import-order guard)
import numpy as np                          # feature-vector math, Slack arithmetic
import pandas as pd                          # DataFrame construction, CSV I/O

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)   # makes `cgcnn_scratch` importable regardless of caller's cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # makes sibling numeric-prefixed scripts importable

from cgcnn_scratch.data import AtomFeaturiser   # noqa: E402  the same 92-dim per-element table CGCNN's embedding layer reads

_tree = import_module("43_baseline_tree_models")   # module object: CONFIG, build_feature_table(), composition_features()
_gamma = import_module("37_train_gamma")            # module object: kappa_full() - the full Slack formula, no cancelling


# Columns this script owns. Listed explicitly (rather than matched by prefix)
# so the idempotency drop below can never delete a column some other script
# happens to have named similarly.
BASELINE_COLUMNS = [
    "K_baseline_rf", "G_baseline_rf", "gamma_baseline_rf", "Kappa_baseline_rf",
    "K_baseline_xgb", "G_baseline_xgb", "gamma_baseline_xgb", "Kappa_baseline_xgb",
    "Kappa_baseline",              # headline = the random-forest family (see module docstring)
    "Kappa_baseline_vs_r9_ratio",  # baseline kappa / round-9 CGCNN kappa, per crystal
]


def fit_baseline_models(ari):
    """Fit RF + XGBoost on the AFLOW TRAIN split for K_VRH, G_VRH and gamma.

    Everything that defines "which rows, which features, which
    hyperparameters" is imported from 43_baseline_tree_models rather than
    restated, so these are provably the same models that script scored - the
    printed test MAEs below are the check that this is still true.

    Returns (fitted_models, test_mae) where fitted_models is
    {target: {model_name: estimator}} and test_mae is {(target, model): mae}.
    """
    from sklearn.ensemble import RandomForestRegressor   # imported here, after the OMP env pin above
    import xgboost as xgb                                 # same - see the env comment at the top

    print("\nRebuilding the AFLOW composition feature table (same rows, same split as step 43)...")
    X, meta = _tree.build_feature_table(
        _tree.CONFIG["aflow_labels_csv"], _tree.CONFIG["aflow_graphs_pt"], "gid", ari)
    assert X.shape[1] == 184, f"expected a 184-dim composition vector, got {X.shape[1]}"
    split = meta["split"].values
    tr_mask, te_mask = split == "train", split == "test"
    print(f"  {len(meta)} AFLOW crystals: {tr_mask.sum()} train / {te_mask.sum()} test, {X.shape[1]}-dim features")

    fitted, test_mae = {}, {}
    for target in ["K_VRH", "G_VRH", "gamma"]:
        y_log = np.log10(meta[target].values)   # every target is fitted in log10 space, exactly as 43 does
        models = {
            "rf": RandomForestRegressor(
                n_estimators=_tree.CONFIG["random_forest_n_estimators"],
                random_state=_tree.CONFIG["random_state"], n_jobs=1),
            "xgb": xgb.XGBRegressor(
                n_estimators=_tree.CONFIG["xgboost_n_estimators"],
                max_depth=_tree.CONFIG["xgboost_max_depth"],
                learning_rate=_tree.CONFIG["xgboost_learning_rate"],
                tree_method=_tree.CONFIG["xgboost_tree_method"],
                random_state=_tree.CONFIG["random_state"], n_jobs=1),
        }
        for name, model in models.items():
            model.fit(X[tr_mask], y_log[tr_mask])   # train rows only - the test rows below are never seen
            mae = float(np.abs(model.predict(X[te_mask]) - y_log[te_mask]).mean())
            test_mae[(target, name)] = mae
        fitted[target] = models
        print(f"  {target:<6} held-out test MAE(log10):  rf {test_mae[(target, 'rf')]:.4f}"
              f"   xgb {test_mae[(target, 'xgb')]:.4f}")
    return fitted, test_mae


def featurise_formulas(formulas, ari):
    """Formula strings -> (184-dim feature matrix, boolean keep-mask).

    A formula containing an element outside atom_init.json's 92-element
    coverage returns None from composition_features() and is masked out
    rather than given a fabricated feature row - the same rule 43 applies to
    its own training data.
    """
    rows, keep = [], np.zeros(len(formulas), dtype=bool)
    for i, formula in enumerate(formulas):
        try:
            fea = _tree.composition_features(formula, ari)   # the exact featuriser the models were fitted with
        except Exception:
            fea = None   # an unparseable formula string - treated the same as an uncovered element
        if fea is not None:
            rows.append(fea)
            keep[i] = True
    return np.array(rows), keep


def score_gnome(fitted, ari, gnome):
    """Predict K, G, gamma for every GNoME row and turn them into Slack kappa.

    Returns a DataFrame indexed by material_id carrying the columns listed in
    BASELINE_COLUMNS (minus the ratio column, which needs round-9's kappa and
    is added by the caller).
    """
    print(f"\nFeaturising {len(gnome)} GNoME formulas (composition only - no CIFs are read)...")
    X, keep = featurise_formulas(gnome["formula"].tolist(), ari)
    if (~keep).any():
        print(f"  {int((~keep).sum())} rows dropped (element outside atom_init.json's 92-element coverage)")
    scored = gnome.loc[keep].copy()   # every downstream array below is in THIS row order

    # Structural constants the full Slack formula needs, read straight off 39's
    # own table - identical values the round-9 kappa was computed from, so the
    # two kappas differ only through K, G and gamma.
    vol_m3 = scored["Volume (A3)"].values * 1e-30        # cubic angstrom -> cubic metre
    n_sites = scored["Number of Atoms"].values            # atom count in the primitive cell
    dens = scored["Density (g cm-3)"].values              # g/cm^3, kappa_full() converts internally

    out = pd.DataFrame({"material_id": scored["material_id"].values})
    for name in ["rf", "xgb"]:
        log_K = fitted["K_VRH"][name].predict(X)      # log10(K/GPa)
        log_G = fitted["G_VRH"][name].predict(X)      # log10(G/GPa)
        gamma = 10.0 ** fitted["gamma"][name].predict(X)   # undo log10 - kappa_full() wants gamma in real units
        with np.errstate(all="ignore"):   # non-physical rows can produce NaN/inf; keep the NaN, drop the warning
            kappa = _gamma.kappa_full(log_K, log_G, gamma, vol_m3, n_sites, dens)
        out[f"K_baseline_{name}"] = 10.0 ** log_K       # back to GPa for a human-readable column
        out[f"G_baseline_{name}"] = 10.0 ** log_G       # same
        out[f"gamma_baseline_{name}"] = gamma            # dimensionless
        out[f"Kappa_baseline_{name}"] = kappa            # W/m/K

    out["Kappa_baseline"] = out["Kappa_baseline_rf"]   # headline family - see module docstring for why RF
    return out.set_index("material_id")


def annotate_csv(path, preds, id_col="material_id"):
    """Left-join the baseline columns into one existing CSV, in place.

    Idempotent by construction: any column in BASELINE_COLUMNS already present
    is dropped before the join, so a re-run replaces rather than duplicates.
    The row count is asserted unchanged - a left join cannot add rows, but the
    assert catches a duplicated material_id in `preds` turning one row into
    several, which would silently corrupt a ranked shortlist.
    """
    if not os.path.exists(path):
        print(f"  SKIP (not found)  {path}")
        return
    df = pd.read_csv(path, dtype={id_col: str})
    n_before = len(df)
    df = df.drop(columns=[c for c in BASELINE_COLUMNS if c in df.columns])   # idempotency: clear this script's previous output
    merged = df.merge(preds.reset_index(), on=id_col, how="left")             # left join keeps every original row, in order
    assert len(merged) == n_before, f"{path}: row count changed {n_before} -> {len(merged)}"
    n_scored = int(merged["Kappa_baseline"].notna().sum())   # how many rows the baseline could actually score
    merged.to_csv(path, index=False)
    print(f"  {n_scored:>6}/{n_before:<6} rows scored   {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)   # --help shows the docstring verbatim
    for key, default in CONFIG.items():   # one CLI flag per CONFIG entry - CONFIG stays the single source of truth
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default) if default is not None else str,
                            default=None, help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            CONFIG[key] = value
    if CONFIG["limit"] is not None:
        CONFIG["limit"] = int(CONFIG["limit"])   # argparse can't infer int when the CONFIG default is None
    CONFIG["annotate"] = int(CONFIG["annotate"])

    print("=" * 78)
    print("  STEP 45 - GNoME screen with the composition-only TREE baseline")
    print("=" * 78)

    ari = AtomFeaturiser(os.path.join(PROJECT_ROOT, CONFIG["atom_init_json"]))
    fitted, test_mae = fit_baseline_models(ari)

    gnome_path = os.path.join(PROJECT_ROOT, CONFIG["gnome_scored_csv"])
    if not os.path.exists(gnome_path):
        sys.exit(f"No GNoME scored table at {gnome_path} - run 39_screen_gnome_gamma.py first")
    gnome = pd.read_csv(gnome_path, dtype={"material_id": str})
    if CONFIG["limit"]:
        gnome = gnome.head(CONFIG["limit"])
        print(f"\n  --limit {CONFIG['limit']}: smoke-testing on the first {len(gnome)} GNoME rows only")

    preds = score_gnome(fitted, ari, gnome)

    # ---- per-crystal disagreement with the round-9 CGCNN screen -------------
    r9 = gnome.set_index("material_id")["Kappa_r9_gamma"]   # the existing pipeline's kappa, same rows, same structural constants
    preds["Kappa_baseline_vs_r9_ratio"] = preds["Kappa_baseline"] / r9.reindex(preds.index)
    preds = preds[BASELINE_COLUMNS]   # fix the column order, and drop anything not declared above

    # ---- write this script's own three screen tables -------------------------
    results_dir = os.path.join(PROJECT_ROOT, CONFIG["results_dir"])
    prefix = CONFIG["out_prefix"]
    full = gnome.merge(preds.reset_index(), on="material_id", how="left")   # 39's columns + this script's, side by side
    full = full.sort_values("Kappa_baseline").reset_index(drop=True)         # ranked by the baseline's own point estimate

    all_path = os.path.join(results_dir, f"{prefix}gnome_screen_all_baseline.csv")
    full.to_csv(all_path, index=False)
    print(f"\nWrote {all_path} ({len(full)} scored candidates, pre-threshold)")

    candidates = full[full["Kappa_baseline"] <= CONFIG["kappa_threshold"]].copy()
    cand_path = os.path.join(results_dir, f"{prefix}gnome_screen_candidates_baseline.csv")
    candidates.to_csv(cand_path, index=False)
    print(f"kappa_L <= {CONFIG['kappa_threshold']} W/m/K (tree baseline): {len(candidates)} candidates")
    print(f"Wrote {cand_path}")

    oxides = candidates[candidates["has_oxygen"] == True].copy()   # noqa: E712 - the column is a real bool/NaN mix, `is True` would drop NaN rows differently
    oxide_path = os.path.join(results_dir, f"{prefix}gnome_oxide_low_kappa_candidates_baseline.csv")
    oxides.to_csv(oxide_path, index=False)
    print(f"  of which {len(oxides)} are oxides")
    print(f"Wrote {oxide_path}")

    # ---- how much does the baseline actually disagree with round 9? ---------
    print("\n" + "=" * 78)
    print("  COMPARISON: round-9 CGCNN screen vs composition-only tree baseline")
    print("=" * 78)
    valid = full.dropna(subset=["Kappa_r9_gamma", "Kappa_baseline"])
    valid = valid[(valid["Kappa_r9_gamma"] > 0) & (valid["Kappa_baseline"] > 0)]   # guard the log10 below
    a, b = np.log10(valid["Kappa_r9_gamma"].values), np.log10(valid["Kappa_baseline"].values)
    print(f"  rows scored by both pipelines : {len(valid)}")
    print(f"  log10(kappa) correlation      : r={np.corrcoef(a, b)[0, 1]:.3f}")
    print(f"  median |log10 disagreement|   : {np.median(np.abs(a - b)):.3f}"
          f"   (= a factor of {10 ** np.median(np.abs(a - b)):.2f}x on a typical crystal)")
    print(f"  median kappa, round-9         : {valid['Kappa_r9_gamma'].median():.3f} W/m/K")
    print(f"  median kappa, tree baseline   : {valid['Kappa_baseline'].median():.3f} W/m/K")

    r9_ids = set(full.loc[full["Kappa_r9_gamma"] <= CONFIG["kappa_threshold"], "material_id"])
    tree_ids = set(candidates["material_id"])
    both = r9_ids & tree_ids
    print(f"\n  round-9 candidates : {len(r9_ids)}")
    print(f"  tree candidates    : {len(tree_ids)}")
    print(f"  in both            : {len(both)}   "
          f"({100 * len(both) / max(1, len(r9_ids | tree_ids)):.1f}% jaccard overlap)")

    # Agreement on the part of the screen that is actually acted on: the 218
    # crystals already shortlisted for DFT. A screen is only as useful as its
    # top of the list, so this is the number that matters more than the
    # 33k-row correlation above.
    dft_path = os.path.join(CONFIG["desktop_dir"], "dft_candidates.csv")
    if os.path.exists(dft_path):
        dft_ids = set(pd.read_csv(dft_path, dtype={"material_id": str})["material_id"])
        sub = full[full["material_id"].isin(dft_ids)]
        n_agree = int((sub["Kappa_baseline"] <= CONFIG["kappa_threshold"]).sum())
        print(f"\n  of the {len(sub)} existing DFT shortlist crystals, the tree baseline also "
              f"calls {n_agree} low-kappa ({100 * n_agree / max(1, len(sub)):.0f}%)")

    # ---- annotate every downstream candidate table in place -----------------
    if CONFIG["annotate"]:
        print("\n" + "=" * 78)
        print("  ANNOTATING DOWNSTREAM CSVs (in place, idempotent)")
        print("=" * 78)
        targets = [
            os.path.join(results_dir, "39_gnome_screen_all_gamma.csv"),
            os.path.join(results_dir, "39_gnome_screen_candidates_gamma.csv"),
            os.path.join(results_dir, "39_gnome_oxide_low_kappa_candidates_gamma.csv"),
            os.path.join(CONFIG["desktop_dir"], "gnome_screen_candidates_gamma.csv"),
            os.path.join(CONFIG["desktop_dir"], "gnome_oxide_low_kappa_candidates_gamma.csv"),
            os.path.join(CONFIG["desktop_dir"], "dft_candidates.csv"),
            os.path.join(CONFIG["desktop_dir"], "dft_candidates_oxides.csv"),
        ]
        for path in targets:
            annotate_csv(path, preds)
    else:
        print("\n(--annotate 0: downstream CSVs left untouched)")


if __name__ == "__main__":
    main()
