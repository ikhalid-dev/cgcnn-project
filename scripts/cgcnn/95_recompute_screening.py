"""
STEP 95, PART 2b - recompute every SCREENING metric (confusion matrices,
precision, recall, F1, recall@10%, Spearman, bootstrap intervals) from the
inputs, and check each one against the number stored when it was first made.

Same rules as part 2a (95_recompute_accuracy.py):
  * every formula is written again from its definition - nothing is imported
    from the script that made the stored number, so a bug there would show here;
  * a CONVENTION that defines a number is copied on purpose and named where
    used: random seeds and the order random draws are taken in, np.argsort's
    default tie order, "linear" percentiles, Wilson z = 1.96, and each step's
    own rule for how many crystals make "the bottom 10%".

WHAT IS COVERED
---------------
    step 82   matbench test (1,648): Slack kappa from 3 models' moduli vs from DFT moduli
              24 confusion matrices, 28 "in range" rows, 13 trust-table rows, per-crystal kappas
    step 87   PhoNIX DFT kappa_L, 2,520 unseen crystals: 16 confusion rows, 100-point precision curve
    step 89   MLIP gamma vs Poisson gamma on the 617-crystal sample: 16 weighted precision/recall
              rows with bootstrap CIs, 4 median/Spearman rows, the per-crystal chain
    step 02   direct-kappa CGCNN vs the tree (AFLOW test, 835): F1 table + paired bootstrap
              (direct_kappa_no_slack/; its control and round-9 rows need torch -> part 3, "stored only")
    step 44   tree baseline's low-kappa confusion matrix on AFLOW test
    step 47   recall of the true bottom decile, 26 distinct runs + per-crystal misses + overlaps
    step 48   the same runs scored against AFLOW AGL kappa (441 matched test crystals)
    step 50   30 runs re-scored on AGL kappa + 29 paired-bootstrap CIs
    step 63   PINK Table 1 (45 crystals with experimental kappa): per-crystal kappas, MAEs
    step 65   leakage audit: test MAE split by "formula also in train", with bootstrap CIs
    step 25   calibration of the ensemble's uncertainty interval (coverage at 9 levels)

OUTPUT
------
    results/cgcnn/95_recomputed_screening.csv   same columns and status rules as part 2a
    results/cgcnn/95_recomputed_screening.png

A "per-crystal" row checks a whole column at once: its value is the LARGEST relative
difference between the recomputed and the stored column over every crystal (stored = 0,
so "match" means every crystal agrees to better than 5e-7). For a True/False or text
column the value is the number of rows that differ.

RUN (ml_env: it has pyarrow for the geometry parquet; no torch is imported)
---
    OMP_NUM_THREADS=1 ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/95_recompute_screening.py
"""

import ast
import glob
import importlib
import os
import sys

# TIE ORDER (a CONVENTION that defines numbers). Matbench moduli are whole numbers of GPa,
# so many soft crystals share an IDENTICAL kappa proxy, and "the lowest 10%" then depends on
# which of two equal values np.argsort happens to put first. numpy 2.x on an AVX2 processor
# sorts with a new fast routine that breaks ties in a different order from numpy 1.26, the
# version steps 47/48/50 ran under. Switching AVX2 off for this one process brings back
# numpy 1.26's order (tested 2026-10-06: same order on tied arrays of 20-1,648 values, and
# this file run under numpy 1.26.4 itself reproduces all 1,395 step 47/48/50 numbers).
# It changes speed only, and it must be set BEFORE numpy is imported - hence up here.
# os.environ is the table of settings a program inherits; setdefault keeps any value already set.
os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX2")

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from pymatgen.core import Composition, Element

# Part 2a's file name starts with a digit, which a normal "import" line cannot take,
# so importlib loads it by name. We reuse ONLY its bookkeeping (record, finish) and its
# rho-free Slack proxy (slack_parts), which 2a already checked against stored numbers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
m = importlib.import_module("95_recompute_accuracy")

ROOT, RES = m.ROOT, m.RES
m.OUT_CSV = os.path.join(RES, "95_recomputed_screening.csv")     # point 2a's finish() at our files
m.OUT_PNG = os.path.join(RES, "95_recomputed_screening.png")
m.TITLE = "STEP 95 PART 2b - screening metrics recomputed from the inputs"
m.FIG_TITLE = "Step 95 part 2b - screening metrics (confusion, precision, recall, bootstraps), recomputed"

PINK = os.path.expanduser("~/Desktop/pink_reproduction")
PHONIX = os.path.join(m.COH, "data", "phonix", "data_all.csv")
GAMMA_DIR = os.path.expanduser("~/Desktop/data_generation/results")
AGL = os.path.join(RES, "agl_target", "csv")
DEC = os.path.join(RES, "decile_stability", "csv")
REB = os.path.join(RES, "rebaseline", "csv")
DKC = os.path.join(m.DK, "results", "csv")

T = 300.0                    # K, the temperature every kappa in this project is quoted at
KB = 1.380649e-23            # Boltzmann constant, J/K
N_A = 6.02214076e23          # Avogadro's number, 1/mol
AMU_G = 1.66053906660e-24    # one atomic mass unit in grams

MB = m.MB
AF = m.AF
PX = "PhoNIX DFT kappa_L, unseen by every model, <= 20 atoms (2,520)"
PX617 = "PhoNIX MLIP sample (617 drawn, 451 usable; weighted to the 2,520)"
AGL441 = "matbench test crystals with an AFLOW AGL kappa (441)"
TAB1 = "PINK Table 1, experimental kappa_L (45)"
MODELS3 = ["ALIGNN", "CGCNN-ens", "newbase"]


# =============================================================================
#  1. Bookkeeping helpers
# =============================================================================
def lab(family, model="", run="", trained_on="", test_set="", target="", split="test", n=None):
    """The labels that say WHAT a number is, as a dict (passed on with **).
    n is only put in when given, so a caller may still pass n=... next to **lab(...)
    (Python refuses the same keyword twice)."""
    out = dict(family=family, model=model, run=run, trained_on=trained_on,
               test_set=test_set, split=split, target=target)
    if n is not None:
        out["n"] = n
    return out


def rel(path):
    """A path relative to the project folder, for the stored_in column."""
    return os.path.relpath(path, ROOT) if path.startswith(ROOT) else path.replace(
        os.path.expanduser("~"), "~")


def compare(new, old, keys, cols, where, stored_in, decimals=None, note=""):
    """Record every cell of `cols` in table `new` against the same cell of `old`.

    The two tables must list the same things in the same row order. The `keys`
    columns (e.g. model, threshold) are checked first and the number of rows
    whose labels differ is itself recorded (stored 0), so a row can never be
    compared silently against the wrong stored row.
    where(row) gives the labels for one stored row; decimals is None, one
    number, or a dict {column: decimals} for tables saved rounded.
    """
    new, old = new.reset_index(drop=True), old.reset_index(drop=True)
    if len(new) != len(old):
        raise SystemExit(f"{stored_in}: {len(new)} recomputed rows vs {len(old)} stored")
    differ = np.zeros(len(old), dtype=bool)
    for k in keys:
        a, b = new[k], old[k]
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            differ |= ~np.isclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12)
        else:                                       # text: compare as text, empty == empty
            differ |= a.fillna("").astype(str).values != b.fillna("").astype(str).values
    head = {**where(old.iloc[0]), "model": "", "run": "(whole table)", "n": len(old)}
    m.record(metric=f"rows whose labels ({', '.join(keys)}) differ from the stored table",
             value=differ.sum(), stored=0.0, stored_in=stored_in, decimals=0, **head)
    for i in range(len(old)):
        w = where(old.iloc[i])
        for c in cols:
            dec = decimals.get(c) if isinstance(decimals, dict) else decimals
            m.record(metric=c, value=float(new[c].iloc[i]), stored=float(old[c].iloc[i]),
                     stored_in=stored_in, decimals=dec, note=note, **w)


def check_column(new, old, what, stored_in, decimals=6, note="", **where):
    """One row for a whole per-crystal column: its WORST disagreement.

    Numbers: largest |new - old| / |old| over all crystals (|new - old| where old
    is 0; NaN on both sides counts as agreement, NaN on one side as infinite).
    True/False or text: how many rows differ. Stored as 0, so 0 = every row agrees.
    """
    a, b = np.asarray(new), np.asarray(old)
    if len(a) != len(b):
        raise SystemExit(f"{stored_in} '{what}': {len(a)} recomputed rows vs {len(b)} stored")
    if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":       # f = float, i/u = integers
        a, b = a.astype(float), b.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = np.abs(a - b) / np.where(b != 0, np.abs(b), 1.0)
        diff[np.isnan(a) & np.isnan(b)] = 0.0
        diff[np.isnan(a) ^ np.isnan(b)] = np.inf             # ^ = exactly one of the two
        value, metric = diff.max(), f"per-crystal '{what}': largest relative difference"
    else:
        a = pd.Series(a).fillna("").astype(str).values
        b = pd.Series(b).fillna("").astype(str).values
        value, metric, decimals = (a != b).sum(), f"per-crystal '{what}': rows that differ", 0
    where.setdefault("n", len(b))
    m.record(metric=metric, value=value, stored=0.0, stored_in=stored_in, decimals=decimals,
             note=note, **where)


# =============================================================================
#  2. Physics, written again from the textbook
# =============================================================================
def slack(K, G, rho, V_A3, n):
    """Slack kappa at 300 K in two halves, (prefactor, Poisson gamma).

    K, G in GPa; rho in g/cm^3; V in A^3; n atoms in that cell.  kappa = prefactor * exp(-gamma)
        sound speeds  v_l = sqrt((K + 4G/3) / rho),  v_t = sqrt(G / rho)   (x1000: km/s -> m/s)
        Debye average v_s = [ (1/v_l^3 + 2/v_t^3) / 3 ] ^ (-1/3)
        Poisson ratio nu  = (3K - 2G) / (2 (3K + G))      <- straight from K and G; the
        Grueneisen    gamma = 3 (1 + nu) / (2 (2 - 3 nu))     original reaches nu via v_l/v_t
        prefactor     G[Pa] * v_s * V[m^3]^(1/3) / (n T)
    """
    K, G, rho, V, n = (np.asarray(x, dtype=float) for x in (K, G, rho, V_A3, n))
    with np.errstate(invalid="ignore", divide="ignore"):    # a negative modulus -> NaN, kept
        v_l = np.sqrt((K + 4 * G / 3) / rho) * 1000
        v_t = np.sqrt(G / rho) * 1000
        v_s = ((v_l ** -3.0 + 2 * v_t ** -3.0) / 3) ** (-1 / 3)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        gamma = 3 * (1 + nu) / (2 * (2 - 3 * nu))
        prefactor = G * 1e9 * v_s * (V * 1e-30) ** (1 / 3) / (n * T)
    return prefactor, gamma


def gamma_from_log(log_K, log_G):
    """Poisson gamma from log10 moduli (the "derived gamma" of steps 37/48/50)."""
    r = 10.0 ** (np.asarray(log_K, float) - np.asarray(log_G, float))    # K/G
    nu = (3 * r - 2) / (2 * (3 * r + 1))           # = (3K - 2G) / (2 (3K + G)), divided by G
    return 3 * (1 + nu) / (2 * (2 - 3 * nu))


def kappa_si(log_K, log_G, gamma, volume_m3, n, density_g_cm3):
    """The same Slack kappa in SI units, from log10 moduli and a given gamma."""
    K = 10.0 ** np.asarray(log_K, float) * 1e9                 # Pa
    G = 10.0 ** np.asarray(log_G, float) * 1e9
    rho = np.asarray(density_g_cm3, float) * 1e3               # kg/m^3
    with np.errstate(invalid="ignore", divide="ignore"):
        v_l, v_t = np.sqrt((K + 4 * G / 3) / rho), np.sqrt(G / rho)
        v_s = ((v_l ** -3.0 + 2 * v_t ** -3.0) / 3) ** (-1 / 3)
        return (G * v_s * np.asarray(volume_m3, float) ** (1 / 3) / (np.asarray(n, float) * T)
                * np.exp(-np.asarray(gamma, float)))


def proxy(log_K, log_G):
    """rho- and V-free kappa (G v_s exp(-gamma)) - ranks crystals like the full formula would
    on one structure set. 2a's slack_parts, already matched to the stored joint numbers."""
    pre, anh, _ = m.slack_parts(np.asarray(log_K, float), np.asarray(log_G, float))
    return pre * anh


def cahill(K, G, rho, V_A3, n):
    """Cahill-Watson-Pohl minimum kappa, high-T form:
    0.5 (pi/6)^(1/3) k_B (n/V)^(2/3) (v_l + 2 v_t)."""
    v_l = np.sqrt((K + 4 * G / 3) / rho) * 1000
    v_t = np.sqrt(G / rho) * 1000
    return 0.5 * (np.pi / 6) ** (1 / 3) * KB * (n / (V_A3 * 1e-30)) ** (2 / 3) * (v_l + 2 * v_t)


def cell_volume_m3(compound, density_g_cm3):
    """Cell volume from its composition and density: mass / density (step 48's rule)."""
    return Composition(compound).weight / N_A / density_g_cm3 * 1e-6


# =============================================================================
#  3. Scores, written again from their definitions
# =============================================================================
def wilson(k, n, z=1.96):
    """Wilson 95% interval for a success rate k/n (NaN when n = 0)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


def safe_div(a, b, empty=np.nan):
    return a / b if b else empty


def confusion(pred, true, thr):
    """'low' = kappa <= thr. The four boxes and every rate built from them."""
    p, t = np.asarray(pred) <= thr, np.asarray(true) <= thr
    tp, fp = int(np.sum(p & t)), int(np.sum(p & ~t))
    fn, tn = int(np.sum(~p & t)), int(np.sum(~p & ~t))
    n = tp + fp + fn + tn
    lo, hi = wilson(tp, tp + fp)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / n,
            "always_no_accuracy": (fp + tn) / n,             # score of never calling anything low
            "precision": safe_div(tp, tp + fp), "precision_lo95": lo, "precision_hi95": hi,
            "recall": safe_div(tp, tp + fn),
            "balanced_accuracy": 0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp))}


def f1_call(pred, true, thr=1.0):
    """Step 51's screening call: same boxes, but 0.0 (not NaN) for an empty ratio."""
    p, t = np.asarray(pred) <= thr, np.asarray(true) <= thr
    tp, fp = int(np.sum(p & t)), int(np.sum(p & ~t))
    fn, tn = int(np.sum(~p & t)), int(np.sum(~p & ~t))
    prec, rec = safe_div(tp, tp + fp, 0.0), safe_div(tp, tp + fn, 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_called": tp + fp, "n_true_low": tp + fn,
            "precision": prec, "recall": rec, "f1": safe_div(2 * prec * rec, prec + rec, 0.0)}


def spearman(a, b):
    """Spearman rho = the ordinary correlation of the two RANK lists (ties share a rank)."""
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def row_corr(x, y):
    """Correlation of x[i] with y[i] for every row i at once (2-D arrays)."""
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))


def recall_at(true, pred, frac, rounding):
    """Share of the true lowest-`frac` crystals that the predicted lowest-`frac` also holds.
    rounding = how the step turned frac * n into a count (CONVENTION, differs by step):
    "int" (47, 48, 50) truncates, "round" (51 / direct 02) rounds."""
    n = len(true)
    k = max(1, int(frac * n)) if rounding == "int" else max(1, int(round(frac * n)))
    lowest_true = set(np.argsort(true)[:k].tolist())    # np.argsort default order (CONVENTION)
    lowest_pred = set(np.argsort(pred)[:k].tolist())
    return len(lowest_true & lowest_pred) / k, k


def bottom_mask(X, k):
    """Per row of X: True at the k smallest entries (np.argsort's default tie order)."""
    mask = np.zeros(X.shape, dtype=bool)
    np.put_along_axis(mask, np.argsort(X, axis=1)[:, :k], True, axis=1)
    return mask


def weighted_median_rows(X, W):
    """Per row: the value with half the total weight below it (first one reaching half)."""
    order = np.argsort(X, axis=1)
    xs = np.take_along_axis(X, order, axis=1)
    cw = np.cumsum(np.take_along_axis(W, order, axis=1), axis=1)
    k = (cw < 0.5 * cw[:, -1:]).sum(axis=1)               # how many sit below half the weight
    return xs[np.arange(len(X)), k]


# =============================================================================
#  4. Step 82 - confusion matrices and the trust table (matbench test)
# =============================================================================
def step82():
    print("step 82 ...", flush=True)
    F = "82 matbench screen"
    labels = pd.read_csv(os.path.join(ROOT, "data_full", "labels.csv"))
    geo = pd.read_parquet(os.path.join(PINK, "pink_geometry.parquet"))
    geo["mb_id"] = [f"mb-{i:05d}" for i in geo.mb_index]
    d = labels.merge(geo, on="mb_id", suffixes=("", "_geo"))
    assert (d.formula == d.formula_geo).all(), "geometry rows do not line up with labels.csv"
    for t in "KG":
        p = pd.read_csv(os.path.join(RES, f"predictions_{t}_VRH_ens.csv"))
        d[f"{t}_CGCNN-ens"] = d.mb_id.map(p[p.split == "test"].set_index("material_id").pred_GPa)
    a = pd.read_csv(os.path.join(m.ALI, "18_alignn_ensemble_predictions.csv"))
    for t in "KG":
        d[f"{t}_ALIGNN"] = d.mb_id.map(a[a.target == t].set_index("material_id").pred_GPa)
    for t, task in [("K", "kvrh"), ("G", "gvrh")]:
        p = pd.read_csv(os.path.join(PINK, "xgboost_oof",
                                     f"matbench_log_{task}_composition+structure+angular_oof.csv"))
        p.index = [f"mb-{i:05d}" for i in p.row_index]
        d[f"{t}_newbase"] = 10 ** d.mb_id.map(p.y_pred_oof)     # stored as log10(GPa)
    d = d.dropna(subset=["K_CGCNN-ens", "K_ALIGNN", "G_CGCNN-ens", "G_ALIGNN"]).reset_index(drop=True)

    pre_t, gam_t = slack(d.K_VRH, d.G_VRH, d.rho_g_cm3, d.V_A3, d.n_atoms)
    d["kappa_DFT"] = pre_t * np.exp(-gam_t)
    for mo in MODELS3:
        pre, gam = slack(d[f"K_{mo}"], d[f"G_{mo}"], d.rho_g_cm3, d.V_A3, d.n_atoms)
        d[f"{mo}_moduli_only"] = pre * np.exp(-gam_t)     # DFT's gamma: only the moduli differ
        d[f"{mo}_full_chain"] = pre * np.exp(-gam)        # the model's own Poisson gamma
    for chain in ["moduli_only", "full_chain"]:
        k3 = d[[f"{mo}_{chain}" for mo in MODELS3]].to_numpy()
        d[f"max3_{chain}"] = k3.max(axis=1)
        d[f"spread3_{chain}"] = k3.max(axis=1) / k3.min(axis=1)

    # ---- per-crystal table -------------------------------------------------------
    path = os.path.join(RES, "82_test_predictions.csv")
    old = pd.read_csv(path)
    w = lab(F, "all three", "per-crystal table", "matbench", MB, "kappa_L (Slack)")
    check_column(d.mb_id, old.mb_id, "mb_id (crystals and order)", rel(path), **w)
    for c in old.columns[2:]:
        check_column(d[c], old[c], c, rel(path), **w)

    # ---- confusion matrices ---------------------------------------------------------
    rows = []
    for chain in ["moduli_only", "full_chain"]:
        for mo in MODELS3 + ["max3"]:
            for thr in [1.0, 0.5, 0.3]:
                rows.append({"chain": chain, "model": mo, "threshold": thr,
                             **confusion(d[f"{mo}_{chain}"], d.kappa_DFT, thr)})
    path = os.path.join(RES, "82_confusion_matrices.csv")
    compare(pd.DataFrame(rows), pd.read_csv(path), ["chain", "model", "threshold"],
            ["TP", "FP", "FN", "TN", "accuracy", "always_no_accuracy", "precision",
             "precision_lo95", "precision_hi95", "recall", "balanced_accuracy"],
            lambda r: lab(F, r.model, f"{r.chain}, low = kappa <= {r.threshold}", "matbench", MB,
                          "kappa_L <= threshold", n=len(d)), rel(path))

    # ---- "in range": how close the prediction is to DFT ----------------------------------
    rows = []
    for mo in MODELS3 + ["max3"]:
        called = (d[f"{mo}_moduli_only"] <= 1.0).to_numpy()
        quantities = [] if mo == "max3" else [("K", d[f"K_{mo}"] / d.K_VRH),
                                              ("G", d[f"G_{mo}"] / d.G_VRH)]
        quantities += [("kappa moduli_only", d[f"{mo}_moduli_only"] / d.kappa_DFT),
                       ("kappa full_chain", d[f"{mo}_full_chain"] / d.kappa_DFT)]
        for what, ratio in quantities:
            ratio = ratio.to_numpy()
            for subset, keep in [("all test crystals", np.ones(len(d), bool)),
                                 ("crystals the model calls low", called)]:
                r = ratio[keep]
                rows.append({"model": mo, "quantity": what, "subset": subset, "n": len(r),
                             "within_25pct": np.mean(np.abs(r - 1) <= 0.25),
                             "within_2x": np.mean((r >= 0.5) & (r <= 2)),
                             "median_abs_pct_error": np.median(np.abs(r - 1)) * 100})
    path = os.path.join(RES, "82_accuracy_in_range.csv")
    compare(pd.DataFrame(rows), pd.read_csv(path), ["model", "quantity", "subset"],
            ["n", "within_25pct", "within_2x", "median_abs_pct_error"],
            lambda r: lab(F, r.model, f"in range: {r.quantity}, {r.subset}", "matbench", MB,
                          r.quantity, n=int(r.n)), rel(path))

    # ---- trust table: of the crystals the screen sends (max3 <= 1), how many DFT confirms --
    s = d[d.max3_moduli_only <= 1.0].reset_index(drop=True)
    x = s.max3_moduli_only
    band = np.select([x <= 0.3, x <= 0.5], ["<=0.3", "0.3-0.5"], "0.5-1.0")   # (0,.3] (.3,.5] (.5,1]
    agree = np.where(s.spread3_moduli_only <= 1.5, "agree (spread <= 1.5)", "disagree (> 1.5)")
    # "contains Cs or Rb", read from the parsed composition (the original used a text pattern)
    cs_rb = np.array([bool({"Cs", "Rb"} & {e.symbol for e in Composition(f).elements})
                      for f in s.formula])
    groups = {"all sent": np.ones(len(s), bool)}
    for b in ["<=0.3", "0.3-0.5", "0.5-1.0"]:
        groups[f"band {b}"] = band == b
    for a_ in ["agree (spread <= 1.5)", "disagree (> 1.5)"]:
        groups[a_] = agree == a_
    for b in ["<=0.3", "0.3-0.5", "0.5-1.0"]:
        for a_ in ["agree (spread <= 1.5)", "disagree (> 1.5)"]:
            groups[f"band {b}, {a_}"] = (band == b) & (agree == a_)
    groups["contains Cs or Rb"] = cs_rb
    path = os.path.join(RES, "82_trust_table.csv")
    old = pd.read_csv(path)
    rows = []
    for name in old.group:                           # same groups, in the stored order
        g = s[groups[name]]
        k = int((g.kappa_DFT <= 1.0).sum())
        lo, hi = wilson(k, len(g))
        ratio = (g.kappa_DFT / g.max3_moduli_only).to_numpy()
        rows.append({"group": name, "n": len(g), "confirmed_low": k,
                     "confirmed_pct": 100 * k / len(g), "ci95_lo": 100 * lo, "ci95_hi": 100 * hi,
                     "DFT_within_2x_pct": 100 * np.mean((ratio >= 0.5) & (ratio <= 2)),
                     "median_DFT_over_pred": np.median(ratio),
                     "p10_DFT_over_pred": np.quantile(ratio, 0.10),     # linear interpolation
                     "p90_DFT_over_pred": np.quantile(ratio, 0.90)})
    nonempty = sorted(k for k, v in groups.items() if v.any())
    m.record(metric="non-empty trust groups missing from the stored table",
             value=len(set(nonempty) - set(old.group)), stored=0.0, stored_in=rel(path), decimals=0,
             **{**lab(F, "max3", "(whole table)", "matbench", MB, "kappa_L <= 1"), "n": len(nonempty)})
    compare(pd.DataFrame(rows), old, ["group"],
            ["n", "confirmed_low", "confirmed_pct", "ci95_lo", "ci95_hi", "DFT_within_2x_pct",
             "median_DFT_over_pred", "p10_DFT_over_pred", "p90_DFT_over_pred"],
            lambda r: lab(F, "max3", f"trust: {r.group}", "matbench", MB,
                          "DFT kappa_L <= 1 among those sent", n=int(r.n)), rel(path))


# =============================================================================
#  5. Step 87 - the Poisson chain against PhoNIX DFT kappa_L
# =============================================================================
MASS = {}


def phonix_geometry(ids):
    """Volume (A^3), density (g/cm^3), atom count from PhoNIX's own structures,
    first copy per material (as step 87 did). Volume = |det(cell)|; mass = sum of
    the atomic masses - not pymatgen's Structure, so the geometry is checked too."""
    raw = pd.read_csv(PHONIX, usecols=["mp_id", "structure"])
    text = raw.drop_duplicates("mp_id").set_index("mp_id").structure
    rows = []
    for mp in ids:
        s = ast.literal_eval(text[mp])               # the text is a Python dict literal
        V = abs(np.linalg.det(np.array(s["cell"], float)))
        for z in s["numbers"]:
            if z not in MASS:
                MASS[z] = float(Element.from_Z(z).atomic_mass)
        mass = sum(MASS[z] for z in s["numbers"])
        rows.append({"mp_id": mp, "V_A3": V, "rho": mass * AMU_G / (V * 1e-24),
                     "n_struct": len(s["numbers"])})
    return pd.DataFrame(rows)


def step87():
    print("step 87 ...", flush=True)
    F = "87 PhoNIX Poisson chain"
    pool = pd.read_csv(os.path.join(RES, "85_phonix_matbench_match.csv"))
    pool = pool[(pool.split == "not in mb") & (pool.n_atoms <= 20)].reset_index(drop=True)
    pool = pool.merge(phonix_geometry(pool.mp_id), on="mp_id")
    assert (pool.n_struct == pool.n_atoms).all()
    tm = pd.read_csv(os.path.join(RES, "87_torch_moduli.csv"))
    nb = pd.read_csv(os.path.join(RES, "87_newbase_moduli.csv"))
    d = pool.merge(tm, on="mp_id", how="left").merge(nb, on="mp_id", how="left")
    cols = {"ALIGNN": ("K_alignn", "G_alignn"), "CGCNN-ens": ("K_cgcnn", "G_cgcnn"),
            "newbase": ("K_newbase", "G_newbase")}
    for name, (kc, gc) in cols.items():
        pre, gam = slack(d[kc], d[gc], d.rho, d.V_A3, d.n_atoms)
        d[name], d[f"gamma_{name}"] = pre * np.exp(-gam), gam
    k3 = d[MODELS3].to_numpy()
    usable = (d.K_alignn > 0.01) & (d.G_alignn > 0.01) & np.isfinite(k3).all(axis=1)  # NaN fails isfinite
    s = d[usable].reset_index(drop=True)
    k3 = s[MODELS3].to_numpy()
    s["kappa_max3"], s["kappa_min3"] = k3.max(axis=1), k3.min(axis=1)
    s["spread3"] = s.kappa_max3 / s.kappa_min3
    s["ratio_dft_over_max3"] = s.klat / s.kappa_max3

    path = os.path.join(RES, "87_phonix_poisson_predictions.csv")
    old = pd.read_csv(path)
    w = lab(F, "all three", "per-crystal table", "matbench", PX, "kappa_L (Slack, Poisson gamma)")
    check_column(s.mp_id, old.mp_id, "mp_id (crystals and order)", rel(path), **w)
    for c in old.columns[3:]:
        check_column(s[c], old[c], c, rel(path), **w)

    rows = []
    for truth in ["klat", "kp"]:
        for thr in [1.0, 0.5]:
            for name in MODELS3 + ["kappa_max3"]:
                rows.append({"truth": truth, "cutoff": thr, "model": name, "n": len(s),
                             "base_rate": np.mean(s[truth] <= thr),
                             **confusion(s[name], s[truth], thr),
                             "spearman_rho": spearman(s[name], s[truth]),
                             "median_dft_over_pred": np.median(s[truth] / s[name])})
    path = os.path.join(RES, "87_phonix_poisson_confusion.csv")
    compare(pd.DataFrame(rows), pd.read_csv(path), ["truth", "cutoff", "model"],
            ["base_rate", "TP", "FP", "FN", "TN", "accuracy", "always_no_accuracy", "precision",
             "precision_lo95", "precision_hi95", "recall", "balanced_accuracy", "spearman_rho",
             "median_dft_over_pred"],
            lambda r: lab(F, r.model, f"truth {r.truth}, low = kappa <= {r.cutoff}", "matbench", PX,
                          f"DFT {r.truth} <= {r.cutoff}", n=int(r.n)), rel(path))

    rows = []
    for name in MODELS3 + ["kappa_max3"]:
        for c in np.round(np.geomspace(0.1, 3.0, 25), 4):    # the stored grid (CONVENTION)
            sel = s[s[name] <= c]
            k = int((sel.klat <= 1.0).sum())
            lo, hi = wilson(k, len(sel))
            rows.append({"model": name, "pred_cutoff": c, "n_selected": len(sel), "n_dft_low": k,
                         "frac_dft_low": safe_div(k, len(sel)), "lo95": lo, "hi95": hi,
                         "frac_dft_below_2": np.mean(sel.klat <= 2.0) if len(sel) else np.nan})
    path = os.path.join(RES, "87_precision_vs_cutoff.csv")
    compare(pd.DataFrame(rows), pd.read_csv(path), ["model", "pred_cutoff"],
            ["n_selected", "n_dft_low", "frac_dft_low", "lo95", "hi95", "frac_dft_below_2"],
            lambda r: lab(F, r.model, f"precision curve, predicted <= {r.pred_cutoff}", "matbench",
                          PX, "DFT klat <= 1 among selected", n=int(r.n_selected)), rel(path))
    return s, pool


# =============================================================================
#  6. Step 89 - MLIP gamma vs Poisson gamma (weighted sample + stratified bootstrap)
# =============================================================================
def step89(s87, pool):
    print("step 89 ...", flush=True)
    F = "89 MLIP vs Poisson"
    plan = pd.read_csv(os.path.join(RES, "88_phonix_mlip_plan.csv"))[["mp_id", "band", "weight"]]
    files = sorted(glob.glob(os.path.join(GAMMA_DIR, "01_gamma_phonon_phonix_cal_s0?.csv")))
    mlip = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)[
        ["mp_id", "status", "dynamically_stable", "min_freq_THz", "gamma_c300", "gamma_mlip_1000K"]]
    d = plan.merge(s87, on="mp_id", how="left").merge(mlip, on="mp_id", how="left")
    # (V_A3 and rho came along with the step-87 table: PhoNIX's own structures)
    assert len(d) == 617 == d.mp_id.nunique() and d.klat.notna().all()
    stable = d.dynamically_stable.astype(bool)
    d["usable"] = (d.status == "ok") & stable & (d.gamma_c300 > 0) & (d.gamma_c300 < 10)
    # labelling rule copied (CONVENTION): any other unusable crystal is labelled "gamma <= 0"
    d["drop_reason"] = np.where(d.usable, "", np.where(~stable, "unstable (imaginary modes)", "gamma <= 0"))
    d["poisson_max3"] = d.kappa_max3
    # swap each model's Poisson gamma for the MLIP gamma: kappa * exp(gamma_poisson - gamma_mlip)
    for g_col, out in [("gamma_c300", "mlip_max3"), ("gamma_mlip_1000K", "mlip1000_max3")]:
        d[out] = np.max([d[mo] * np.exp(d[f"gamma_{mo}"] - d[g_col]) for mo in MODELS3], axis=0)
    b = pd.read_csv(os.path.join(GAMMA_DIR, "02_gamma_comparison.csv"))
    b = b[b.stable & b.clean].dropna(subset=["gamma_paper", "gamma_mlip"])
    ratio = (b.gamma_paper / b.gamma_mlip).to_numpy()
    typical, stress = np.median(ratio), np.quantile(ratio, 0.10)      # linear (CONVENTION)
    d["kappa_cahill"] = cahill(d.K_cgcnn, d.G_cgcnn, d.rho, d.V_A3, d.n_atoms)
    g = d.gamma_c300
    d["mlip_typical"] = np.maximum(d.mlip_max3 * np.exp(g * (1 - typical)), d.kappa_cahill)
    d["mlip_stress"] = np.maximum(d.mlip_max3 * np.exp(g * (1 - stress)), d.kappa_cahill)
    s15 = pd.read_csv(os.path.join(ROOT, "dft", "final_shortlist_15", "index.csv")).kappa_stress.max()

    path = os.path.join(RES, "89_mlip_vs_poisson_crystals.csv")
    old = pd.read_csv(path)
    w = lab(F, "max of three", "per-crystal table", "matbench", PX617, "kappa_L (Slack)")
    for c in ["mp_id", "band", "weight", "usable", "drop_reason", "poisson_max3", "mlip_max3",
              "mlip1000_max3"]:
        check_column(d[c], old[c], c, rel(path), **w)
    note = ("geometry from PhoNIX's structures; step 89 read the CIF files written from them, "
            "whose rounding moves the floor in the 7th digit")
    for c in ["kappa_cahill", "mlip_typical", "mlip_stress"]:
        check_column(d[c], old[c], c, rel(path), decimals=5, note=note, **w)
    m.record(metric="benchmark gamma ratio (reference / MLIP), median = 'typical'", value=typical,
             n=len(ratio), note="printed by step 86/89, never saved",
             **lab(F, "MACE (MLIP)", "gamma correction", "", "data_generation 02 gamma benchmark", "gamma"))
    m.record(metric="benchmark gamma ratio (reference / MLIP), 10th percentile = 'stress'", value=stress,
             n=len(ratio), note="printed by step 86/89, never saved",
             **lab(F, "MACE (MLIP)", "gamma correction", "", "data_generation 02 gamma benchmark", "gamma"))

    u = d[d.usable].reset_index(drop=True)
    wt, band = u.weight.to_numpy(), u.band.to_numpy()
    rng = np.random.default_rng(89)                 # seed + draw order = step 89's (CONVENTION)
    idx = np.concatenate([pos[rng.integers(0, len(pos), (10_000, len(pos)))]
                          for pos in (np.flatnonzero(band == b_) for b_ in sorted(set(band)))], axis=1)

    def rates(call, truth, rows=None):
        """Weighted precision and recall (weights = how many of the 2,520 a crystal stands for)."""
        W, C, Tr = (wt, call, truth) if rows is None else (wt[rows], call[rows], truth[rows])
        ax = None if rows is None else 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((W * C * Tr).sum(axis=ax) / (W * C).sum(axis=ax),
                    (W * C * Tr).sum(axis=ax) / (W * Tr).sum(axis=ax))

    T1 = "step-86 tier 1, stress <= 1 [added]"
    S15 = f"as safe as the final 15, stress <= {s15:.2f} [added]"
    rules = {"Poisson gamma (step 87 chain)": u.poisson_max3 <= 1,
             "MLIP gamma 300 K (production)": u.mlip_max3 <= 1,
             "MLIP gamma 1000 K [added]": u.mlip1000_max3 <= 1,
             "step-86 tiers 1+2, typical <= 1 [added]": u.mlip_typical <= 1,
             T1: u.mlip_stress <= 1, S15: u.mlip_stress <= s15}
    rules = {k: v.to_numpy() for k, v in rules.items()}
    # a Poisson cutoff that calls the SAME weighted number of crystals as each stress rule
    px = u.poisson_max3.to_numpy()
    order = np.argsort(px)
    cum = np.cumsum(wt[order])
    for name in [T1, S15]:
        c = px[order][np.searchsorted(cum, (wt * rules[name]).sum() - 1e-9)]
        rules[f"Poisson, cutoff {c:.2f} = as strict as '{name.split(',')[0]}' [added]"] = px <= c

    rows, boots = [], {}
    for truth_col in ["klat", "kp"]:
        truth = (u[truth_col] <= 1).to_numpy()
        for name, call in rules.items():
            p, r = rates(call, truth)
            bp, br = rates(call, truth, idx)
            if truth_col == "klat":
                boots[name] = bp
            rows.append({"truth": truth_col, "rule": name, "n_called_sample": call.sum(),
                         "n_confirmed_sample": (call & truth).sum(),
                         "est_called_of_usable_2520": (wt * call).sum(),
                         "precision": p, "precision_lo95": np.nanpercentile(bp, 2.5),
                         "precision_hi95": np.nanpercentile(bp, 97.5),
                         "recall": r, "recall_lo95": np.nanpercentile(br, 2.5),
                         "recall_hi95": np.nanpercentile(br, 97.5),
                         "base_rate": (wt * truth).sum() / wt.sum()})
    path = os.path.join(RES, "89_mlip_vs_poisson_summary.csv")
    compare(pd.DataFrame(rows), pd.read_csv(path), ["truth", "rule"],
            ["n_called_sample", "n_confirmed_sample", "est_called_of_usable_2520", "precision",
             "precision_lo95", "precision_hi95", "recall", "recall_lo95", "recall_hi95", "base_rate"],
            lambda r: lab(F, "max of three", r.rule, "matbench", PX617, f"DFT {r.truth} <= 1",
                          n=len(u)), rel(path))
    P, M = "Poisson gamma (step 87 chain)", "MLIP gamma 300 K (production)"
    dp = boots[M] - boots[P]
    k_ = pd.DataFrame(rows).query("truth == 'klat'").set_index("rule").precision
    for what, v in [("", k_[M] - k_[P]), (" lo95", np.nanpercentile(dp, 2.5)),
                    (" hi95", np.nanpercentile(dp, 97.5))]:
        m.record(metric=f"precision gain, MLIP minus Poisson{what}", value=v, n=len(u),
                 note="paired stratified bootstrap; printed by step 89, never saved",
                 **lab(F, "max of three", "MLIP 300 K vs Poisson", "matbench", PX617, "DFT klat <= 1"))

    sec, rho_boot = [], {}
    sub = idx[:2_000]                                # step 89 used the first 2,000 redraws here
    for name, col in [(P, "poisson_max3"), (M, "mlip_max3"), ("MLIP gamma 1000 K [added]", "mlip1000_max3"),
                      ("step-86 typical correction [added]", "mlip_typical")]:
        pred, klat = u[col].to_numpy(), u.klat.to_numpy()
        ratio = klat / pred
        med_boot = weighted_median_rows(ratio[sub], wt[sub])
        rb = row_corr(rankdata(pred[sub], axis=1), rankdata(klat[sub], axis=1))
        rho_boot[name] = rb
        sec.append({"chain": name, "median_dft_over_pred": weighted_median_rows(ratio[None], wt[None])[0],
                    "median_lo95": np.percentile(med_boot, 2.5), "median_hi95": np.percentile(med_boot, 97.5),
                    "spearman_rho_sample": spearman(pred, klat),
                    "rho_lo95": np.percentile(rb, 2.5), "rho_hi95": np.percentile(rb, 97.5)})
    path = os.path.join(RES, "89_mlip_vs_poisson_secondary.csv")
    compare(pd.DataFrame(sec), pd.read_csv(path), ["chain"],
            ["median_dft_over_pred", "median_lo95", "median_hi95", "spearman_rho_sample",
             "rho_lo95", "rho_hi95"],
            lambda r: lab(F, "max of three", r.chain, "matbench", PX617, "DFT klat / predicted",
                          n=len(u)), rel(path))
    dr = rho_boot[M] - rho_boot[P]
    for what, v in [("", sec[1]["spearman_rho_sample"] - sec[0]["spearman_rho_sample"]),
                    (" lo95", np.percentile(dr, 2.5)), (" hi95", np.percentile(dr, 97.5))]:
        m.record(metric=f"Spearman gain, MLIP minus Poisson{what}", value=v, n=len(u),
                 note="paired stratified bootstrap (first 2,000 redraws); printed by step 89, never saved",
                 **lab(F, "max of three", "MLIP 300 K vs Poisson", "matbench", PX617, "DFT klat"))


# =============================================================================
#  7. Direct-kappa step 02 - direct CGCNN vs the tree, on the AFLOW test split
# =============================================================================
def direct02():
    print("direct-kappa 02 ...", flush=True)
    F = "direct-kappa 02 screen"
    meta = pd.read_csv(os.path.join(ROOT, "data_full", "gamma_labels.csv"), dtype={"gid": str})
    meta = meta.set_index("gid")
    seeds = []
    for tag in ["s1", "s2", "s42"]:                 # sorted checkpoint names, as 02 ran them
        path = os.path.join(DKC, f"01_direct_kappa_test_{tag}.csv")
        f = pd.read_csv(path, dtype={"gid": str})
        seeds.append((tag, f, path))
    ids = seeds[0][1].gid.tolist()
    for tag, f, path in seeds[1:]:
        assert f.gid.tolist() == ids, f"{tag}: different crystal order"
    true = meta.loc[ids, "kappa_agl"].to_numpy(float)
    w = lab(F, "direct CGCNN (structure -> kappa)", "per-seed test file", "AFLOW", AF, "kappa_AGL")
    for tag, f, path in seeds:
        check_column(f.true_log10_kappa.to_numpy(float), np.log10(true),
                     f"{tag} true_log10_kappa vs log10(kappa_agl) in gamma_labels.csv", rel(path),
                     note="the per-seed file stores float32", **w)
    # 02's split: np.random.RandomState(42).permutation over the label rows, last 15% = test
    n_all = len(meta)
    perm = np.random.RandomState(42).permutation(n_all)
    test_ids = meta.index[perm[int(round(0.70 * n_all)) + int(round(0.15 * n_all)):]].tolist()
    m.record(metric="test crystals out of 02's split order (RandomState(42), 70/15/15)",
             value=sum(a != b for a, b in zip(ids, test_ids)) + abs(len(ids) - len(test_ids)),
             stored=0.0, stored_in=rel(seeds[0][2]), decimals=0, n=len(ids),
             note="the paired bootstrap resamples POSITIONS, so the order must be 02's", **w)

    def score(kappa):
        out = f1_call(kappa, true)
        out["spearman"] = spearman(true, kappa)
        for frac in [0.10, 0.20]:
            r, k = recall_at(true, kappa, frac, "round")          # step 51's rounding (CONVENTION)
            out[f"recall{int(frac * 100)}"], out[f"n_bin{int(frac * 100)}"] = 100 * r, k
        e = np.abs(np.log10(kappa) - np.log10(true))
        out["mae_log10"], out["median_ae_log10"] = e.mean(), np.median(e)
        return out

    rows = [{"arm": "direct", "tag": tag, "member": "seed",
             **score(10.0 ** f.pred_log10_kappa.to_numpy(float))} for tag, f, _ in seeds]
    ens_log = np.mean([f.pred_log10_kappa.to_numpy(float) for _, f, _ in seeds], axis=0)
    direct = 10.0 ** ens_log                          # ensemble = mean of the log10 predictions
    rows.append({"arm": "direct", "tag": "direct ensemble", "member": "ensemble", **score(direct)})
    tree = pd.read_csv(os.path.join(RES, "baseline_tree", "csv", "43_baseline_tree_predictions.csv"))
    tt = tree[(tree.source == "aflow") & (tree.split == "test")]
    tt = tt.set_index(tt.material_id.astype(str)).loc[ids]
    mt = meta.loc[ids]
    tree_kappa = kappa_si(tt.K_VRH_pred_log10_random_forest, tt.G_VRH_pred_log10_random_forest,
                          10.0 ** tt.gamma_pred_log10_random_forest.to_numpy(float),
                          mt.volume_m3, mt.n_sites, mt.density_g_cm3)
    rows.append({"arm": "tree", "tag": "tree (random_forest)", "member": "ensemble", **score(tree_kappa)})
    path = os.path.join(DKC, "02_direct_vs_slack_scores.csv")
    old = pd.read_csv(path)
    old = old[old.arm.isin(["direct", "tree"])]
    # Checked 2026-10-06 (infer_env, 02's own predict_direct, nothing written): 02's re-run of
    # the checkpoints differs from 01's saved predictions on 7 of 835 crystals - the SAME 7 in
    # all three seeds (high-symmetry 5-15 atom cells, up to 0.09 in log10), so their input
    # graphs differ, not the networks - and scored with THIS file's formulas, 02's own
    # predictions reproduce every stored value to 6 digits.
    note = ("02 re-ran the checkpoints locally; 01's saved predictions differ on the same 7 of 835 "
            "crystals in every seed (up to 0.09 log10). Scored on 02's own re-run, this file's "
            "formulas reproduce every stored value exactly (checked 2026-10-06)")
    compare(pd.DataFrame(rows), old, ["arm", "tag"],
            ["tp", "fp", "fn", "tn", "n_called", "n_true_low", "precision", "recall", "f1", "spearman",
             "recall10", "n_bin10", "recall20", "n_bin20", "mae_log10", "median_ae_log10"],
            lambda r: lab(F, "direct CGCNN" if r.arm == "direct" else "random forest (K, G, gamma)",
                          r.tag, "AFLOW", AF, "kappa_AGL <= 1", n=len(true)), rel(path), note=note)

    # paired bootstrap of F1(direct ensemble) - F1(tree): default_rng(0), 10,000 x 835 (CONVENTION)
    idx = np.random.default_rng(0).integers(0, len(true), (10_000, len(true)))

    def f1_rows(pred):
        P, Tr = pred[idx] <= 1.0, true[idx] <= 1.0
        tp, fp, fn = (P & Tr).sum(1), (P & ~Tr).sum(1), (~P & Tr).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            return np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    dd = f1_rows(direct) - f1_rows(tree_kappa)
    lo, hi = np.percentile(dd, [2.5, 97.5])
    path = os.path.join(DKC, "02_direct_vs_slack_bootstrap.csv")
    new = pd.DataFrame([{"vs": "tree", "d_f1": dd.mean(), "ci_lo": lo, "ci_hi": hi,
                         "decisive": (lo > 0) or (hi < 0)}])
    compare(new, pd.read_csv(path).query("vs == 'tree'"), ["vs"], ["d_f1", "ci_lo", "ci_hi", "decisive"],
            lambda r: lab(F, "direct CGCNN vs random forest", "F1 gap, paired bootstrap", "AFLOW", AF,
                          "kappa_AGL <= 1", n=len(true)), rel(path), note=note)


# =============================================================================
#  8. Step 44 - the tree baseline's low-kappa call on AFLOW test
# =============================================================================
def step44():
    print("step 44 ...", flush=True)
    F = "44 tree screen"
    tree = pd.read_csv(os.path.join(RES, "baseline_tree", "csv", "43_baseline_tree_predictions.csv"))
    sub = tree[(tree.source == "aflow") & (tree.split == "test")].copy()
    meta = pd.read_csv(os.path.join(ROOT, "data_full", "gamma_labels.csv"), dtype={"gid": str}).set_index("gid")
    mt = meta.loc[sub.material_id.astype(str)]
    pre, gam = slack(10 ** sub.K_VRH_pred_log10_xgboost.to_numpy(), 10 ** sub.G_VRH_pred_log10_xgboost.to_numpy(),
                     mt.density_g_cm3, mt.volume_m3.to_numpy() * 1e30, mt.n_sites)
    pred, true = pre * np.exp(-gam), mt.kappa_agl.to_numpy(float)
    ok = np.isfinite(pred) & np.isfinite(true) & (pred > 0)
    c = f1_call(pred[ok], true[ok])
    w = lab(F, "xgboost (K, G) + Poisson gamma", "low-kappa call, kappa <= 1", "AFLOW", AF, "kappa_AGL <= 1",
            n=int(ok.sum()))
    path = os.path.join(RES, "baseline_tree", "csv", "44_confusion_matrix_aflow_low_kappa.csv")
    old = pd.read_csv(path)
    for name, v, (i, j) in [("TP", c["tp"], (0, 1)), ("FN", c["fn"], (0, 2)),
                            ("FP", c["fp"], (1, 1)), ("TN", c["tn"], (1, 2))]:
        m.record(metric=name, value=v, stored=old.iat[i, j], stored_in=rel(path), **w)
    path = path.replace(".csv", "_metrics.csv")
    old = pd.read_csv(path).iloc[0]
    for name, v in [("n_test_crystals", ok.sum()), ("n_dropped_non_finite", (~ok).sum()),
                    ("n_true_low_kappa", c["n_true_low"]), ("n_pred_low_kappa", c["n_called"]),
                    ("precision", c["precision"]), ("recall", c["recall"]), ("f1", c["f1"])]:
        m.record(metric=name, value=v, stored=old[name], stored_in=rel(path), **w)


# =============================================================================
#  9. Steps 47, 48, 50 - recall@10% and Spearman of every saved moduli run
# =============================================================================
def roster47():
    """Step 47's list of runs, read out of its source file (it is data, not a formula)."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "47_missed_decile_stability.py")).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ROSTER":
            return ast.literal_eval(node.value)


def load_run(files):
    """One run's TEST predictions as log10 K and G (true and predicted), sorted by crystal id."""
    paths = [os.path.join(RES, f) for f in files]
    if not all(os.path.exists(p) for p in paths):
        return None
    if len(paths) == 1:                                       # joint model: one file, both moduli
        f = pd.read_csv(paths[0])
        out = f[f.split == "test"].set_index("material_id")[
            ["true_log10_K", "true_log10_G", "pred_log10_K", "pred_log10_G"]]
    else:                                                     # separate K and G models
        k, g = (pd.read_csv(p) for p in paths)
        k, g = (x[x.split == "test"].set_index("material_id") for x in (k, g))
        out = pd.DataFrame({"true_log10_K": k.true_log10, "true_log10_G": g.true_log10,
                            "pred_log10_K": k.pred_log10, "pred_log10_G": g.pred_log10}).dropna()
    return out.sort_index()


def distinct_runs():
    """The roster with byte-identical runs collapsed (first tag kept), as step 47 did:
    two runs are the same if their predictions agree to 9 decimals."""
    runs, seen = [], set()
    for label, tier, files in roster47():
        f = load_run(files)
        if f is None:
            continue
        key = np.round(np.concatenate([f.pred_log10_K.values, f.pred_log10_G.values]), 9).tobytes()
        if key not in seen:
            seen.add(key)
            runs.append((label, tier, f))
    return runs


def step47(runs):
    print("step 47 ...", flush=True)
    F = "47 decile recall"
    ref = runs[0][2]
    kt_all = proxy(ref.true_log10_K, ref.true_log10_G)
    fin = np.isfinite(kt_all) & (kt_all > 0)
    kt, ids = kt_all[fin], ref.index[fin]
    n10 = max(1, int(0.10 * len(kt)))                         # step 47's count rule (CONVENTION)
    truth_rows = np.argsort(kt)[:n10]
    truth_set = set(truth_rows.tolist())
    rows, missed, rank = [], {}, {}
    for label, tier, f in runs:
        assert f.index.equals(ref.index)
        kp = proxy(f.pred_log10_K, f.pred_log10_G)[fin]
        order = np.argsort(kp)
        miss = truth_set - set(order[:n10].tolist())
        r = np.empty(len(kp), int)
        r[order] = np.arange(1, len(kp) + 1)                  # 1 = the model's lowest kappa
        missed[label], rank[label] = miss, r
        rows.append({"run": label, "tier": tier, "n_decile": n10, "n_hit": n10 - len(miss),
                     "n_missed": len(miss), "recall_pct": 100 * (n10 - len(miss)) / n10,
                     "median_pred_rank_of_missed": int(np.median([r[i] for i in miss])) if miss else 0})
    path = os.path.join(DEC, "47_missed_decile_run_summary.csv")
    compare(pd.DataFrame(rows), pd.read_csv(path), ["run", "tier"],
            ["n_decile", "n_hit", "n_missed", "recall_pct", "median_pred_rank_of_missed"],
            lambda r: lab(F, f"CGCNN {r.tier}", r.run, "matbench", MB, "bottom 10% of kappa proxy",
                          n=len(kt)), rel(path))

    # per crystal of the true bottom decile: which runs missed it, and at what rank they put it
    ens = [lb for lb, t, _ in runs if t == "ensemble"]
    sing = [lb for lb, t, _ in runs if t == "single"]
    pc = pd.DataFrame({"material_id": ids[truth_rows], "rank_in_true_decile": np.arange(1, n10 + 1),
                       "true_kappa_proxy": kt[truth_rows]})
    pc["n_ensembles_missing"] = [sum(i in missed[lb] for lb in ens) for i in truth_rows]
    pc["n_singles_missing"] = [sum(i in missed[lb] for lb in sing) for i in truth_rows]
    pc["missed_by_every_run"] = (pc.n_ensembles_missing + pc.n_singles_missing) == len(runs)
    pc["missed_by_every_ensemble"] = pc.n_ensembles_missing == len(ens)
    pc["missed_by_every_single"] = pc.n_singles_missing == len(sing)
    for lb, _, _ in runs:
        pc[f"rank__{lb}"] = rank[lb][truth_rows]
    rk = pc[[f"rank__{lb}" for lb, _, _ in runs]].to_numpy()
    pc["worst_pred_rank"], pc["best_pred_rank"] = rk.max(axis=1), rk.min(axis=1)
    path = os.path.join(DEC, "47_missed_decile_per_crystal.csv")
    old = pd.read_csv(path)
    w = lab(F, "all 26 runs", "per-crystal table (true bottom decile)", "matbench", MB, "kappa proxy")
    for c in ["material_id", "rank_in_true_decile", "true_kappa_proxy", "n_ensembles_missing",
              "n_singles_missing", "missed_by_every_run", "missed_by_every_ensemble",
              "missed_by_every_single", "worst_pred_rank", "best_pred_rank"] + \
             [c for c in old.columns if c.startswith("rank__")]:
        check_column(pc[c], old[c], c, rel(path), **w)
    for name, v, col in [("crystals missed by every run", pc.missed_by_every_run.sum(), "missed_by_every_run"),
                         ("crystals missed by every ensemble", pc.missed_by_every_ensemble.sum(),
                          "missed_by_every_ensemble"),
                         ("crystals no run missed", ((pc.n_ensembles_missing + pc.n_singles_missing) == 0).sum(),
                          None)]:
        stored = old[col].sum() if col else ((old.n_ensembles_missing + old.n_singles_missing) == 0).sum()
        m.record(metric=name, value=v, stored=stored, stored_in=rel(path) + " (column total)",
                 **{**w, "run": "summary of the 164", "n": n10})

    labels = [lb for lb, _, _ in runs]
    both = np.array([[len(missed[a] & missed[b]) for b in labels] for a in labels], float)
    jac = np.array([[len(missed[a] & missed[b]) / len(missed[a] | missed[b]) for b in labels]
                    for a in labels])
    for name, mat in [("counts", both), ("jaccard", jac)]:
        path = os.path.join(DEC, f"47_missed_decile_pairwise_{name}.csv")
        old = pd.read_csv(path, index_col=0).loc[labels, labels].to_numpy(float)
        check_column(mat.ravel(), old.ravel(), f"pairwise {name} matrix, every cell", rel(path),
                     **{**w, "run": f"{len(labels)} x {len(labels)} matrix", "n": mat.size})


def step48(runs):
    print("step 48 ...", flush=True)
    F = "48 AGL target"
    path = os.path.join(AGL, "48_agl_matched_test_crystals.csv")
    mt = pd.read_csv(path)
    vol = np.array([cell_volume_m3(c, rho) for c, rho in zip(mt.compound, mt.density)])
    w = lab(F, "", "matched test crystals", "matbench", AGL441, "cell volume")
    check_column(vol, mt.cell_volume_m3, "cell_volume_m3", rel(path), **w)
    order = mt.mb_id
    agl, agl_gamma = mt.agl_kappa.to_numpy(float), mt.agl_gamma.to_numpy(float)
    ref = runs[0][2].loc[order]
    inhouse = proxy(ref.true_log10_K, ref.true_log10_G)
    rows = []
    for label, tier, f in runs:
        f = f.loc[order]
        pK, pG = f.pred_log10_K.to_numpy(), f.pred_log10_G.to_numpy()
        preds = {"proxy": proxy(pK, pG),
                 "full": kappa_si(pK, pG, gamma_from_log(pK, pG), vol, mt.natoms, mt.density),
                 "oracleGamma": kappa_si(pK, pG, agl_gamma, vol, mt.natoms, mt.density)}
        ok = np.isfinite(agl) & (agl > 0)
        for v in preds.values():
            ok &= np.isfinite(v)
        row = {"run": label, "tier": tier, "n_scored": ok.sum()}
        for tname, tgt in [("agl", agl), ("inhouse", inhouse)]:
            for pname, pv in preds.items():
                if tname == "inhouse" and pname != "proxy":
                    continue
                for frac in [0.10, 0.20]:
                    r, k = recall_at(tgt[ok], pv[ok], frac, "int")
                    row[f"recall{int(frac * 100)}_{tname}_{pname}"] = 100 * r
                    row[f"n_in_bin_{int(frac * 100)}"] = k
                row[f"spearman_{tname}_{pname}"] = spearman(tgt[ok], pv[ok])
        rows.append(row)
    path = os.path.join(AGL, "48_agl_vs_inhouse_scores.csv")
    old = pd.read_csv(path)
    cols = [c for c in old.columns if c not in ("run", "tier")]
    dec = {c: (1 if c.startswith("recall") else 3 if c.startswith("spearman") else None) for c in cols}
    compare(pd.DataFrame(rows), old, ["run", "tier"], cols,
            lambda r: lab(F, f"CGCNN {r.tier}", r.run, "matbench", AGL441, "AGL kappa (agl) / own proxy (inhouse)",
                          n=int(r.n_scored)), rel(path), decimals=dec)


def load_alignn_pair():
    frames = {}
    for t, name in [("K", "alignn_bulk_modulus_kv"), ("G", "alignn_shear_modulus_gv")]:
        f = pd.read_csv(os.path.join(m.ALI, name, "prediction_results_test_set.csv"))
        f.columns = [c.strip() for c in f.columns]            # ALIGNN pads its header with spaces
        f["id"] = f["id"].astype(str).str.strip()
        frames[t] = f.set_index("id")
    k, g = frames["K"], frames["G"]
    ids = k.index.intersection(g.index)
    pk, pg = k.loc[ids, "prediction"].to_numpy(float), g.loc[ids, "prediction"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):      # a modulus <= 0 has no log -> dropped
        out = pd.DataFrame({"true_log10_K": np.log10(k.loc[ids, "target"].to_numpy(float)),
                            "true_log10_G": np.log10(g.loc[ids, "target"].to_numpy(float)),
                            "pred_log10_K": np.log10(np.where(pk > 0, pk, np.nan)),
                            "pred_log10_G": np.log10(np.where(pg > 0, pg, np.nan))}, index=ids)
    return out.dropna().sort_index()


def step50(runs):
    print("step 50 (10,000 bootstrap redraws x 29 runs) ...", flush=True)
    F = "50 AGL re-baseline"
    mt = pd.read_csv(os.path.join(AGL, "48_agl_matched_all_matbench.csv")).set_index("mb_id")
    mt["volume_m3"] = [cell_volume_m3(c, rho) for c, rho in zip(mt.compound, mt.density)]
    tree = pd.read_csv(os.path.join(RES, "baseline_tree", "csv", "43_baseline_tree_predictions.csv"))
    tree = tree[(tree.source == "matbench") & (tree.split == "test")]
    all_runs = [(lb, "cgcnn", t, f) for lb, t, f in runs] + [("ALIGNN (K+G)", "alignn", "single", load_alignn_pair())]
    for mdl in ["decision_tree", "random_forest", "xgboost"]:
        f = pd.DataFrame({"true_log10_K": tree.K_VRH_true_log10.values, "true_log10_G": tree.G_VRH_true_log10.values,
                          "pred_log10_K": tree[f"K_VRH_pred_log10_{mdl}"].values,
                          "pred_log10_G": tree[f"G_VRH_pred_log10_{mdl}"].values},
                         index=tree.material_id.astype(str)).dropna().sort_index()
        all_runs.append((f"tree: {mdl}", "tree", "single", f))
    common = mt.index
    for _, _, _, f in all_runs:
        common = common.intersection(f.index)
    common = common.sort_values()
    meta = mt.loc[common]
    agl = meta.agl_kappa.to_numpy(float)
    rows, full = [], {}
    for label, fam, tier, f in all_runs:
        f = f.loc[common]
        pK, pG = f.pred_log10_K.to_numpy(), f.pred_log10_G.to_numpy()
        kinds = {"full": kappa_si(pK, pG, gamma_from_log(pK, pG), meta.volume_m3, meta.natoms, meta.density),
                 "proxy": proxy(pK, pG)}
        full[label] = kinds["full"]
        row = {"run": label, "family": fam, "tier": tier, "n_scored": len(common)}
        for kind, pv in kinds.items():
            for frac in [0.10, 0.20]:
                r, k = recall_at(agl, pv, frac, "int")
                row[f"recall{int(frac * 100)}_{kind}"], row[f"n_bin{int(frac * 100)}_{kind}"] = 100 * r, k
            row[f"spearman_{kind}"] = spearman(agl, pv)
            e = np.abs(np.log10(pv) - np.log10(agl))
            row[f"mae_log10_{kind}"], row[f"median_ae_log10_{kind}"] = e.mean(), np.median(e)
            row[f"p95_ae_log10_{kind}"] = np.percentile(e, 95)
        rows.append(row)
    path = os.path.join(REB, "50_rebaseline_scores.csv")
    old = pd.read_csv(path)
    compare(pd.DataFrame(rows), old, ["run", "family", "tier"],
            [c for c in old.columns if c not in ("run", "family", "tier")],
            lambda r: lab(F, f"{r.family} {r.tier}", r.run, "matbench", AGL441, "AGL kappa",
                          n=int(r.n_scored)), rel(path))

    # paired bootstrap vs the reference run: default_rng(0) per comparison -> the SAME redraws
    idx = np.random.default_rng(0).integers(0, len(agl), (10_000, len(agl)))
    k10 = max(1, int(0.10 * len(agl)))
    Tt = agl[idx]
    rank_t, low_t, logT = rankdata(Tt, axis=1), bottom_mask(Tt, k10), np.log10(Tt)

    def stats(pred):
        P = pred[idx]
        return (row_corr(rank_t, rankdata(P, axis=1)), (low_t & bottom_mask(P, k10)).sum(axis=1) / k10,
                np.abs(np.log10(P) - logT).mean(axis=1))

    ref = "separate 3-ens (baseline)"
    r_s, r_r, r_m = stats(full[ref])
    rows = []
    for label, _, _, _ in all_runs:
        if label == ref:
            continue
        s_, r_, m_ = stats(full[label])
        row = {"run": label}
        for name, v in [("d_spearman", s_ - r_s), ("d_recall10", 100 * (r_ - r_r)), ("d_mae_log10", m_ - r_m)]:
            lo, hi = np.percentile(v, [2.5, 97.5])
            row[name + ("_pts" if name == "d_recall10" else "")], row[name + "_lo"], row[name + "_hi"] = v.mean(), lo, hi
        row["spearman_significant"] = (row["d_spearman_lo"] > 0) or (row["d_spearman_hi"] < 0)
        row["recall10_significant"] = (row["d_recall10_lo"] > 0) or (row["d_recall10_hi"] < 0)
        rows.append(row)
    path = os.path.join(REB, "50_rebaseline_bootstrap_ci.csv")
    old = pd.read_csv(path)
    compare(pd.DataFrame(rows), old, ["run"], [c for c in old.columns if c != "run"],
            lambda r: lab(F, "", f"{r.run} minus baseline", "matbench", AGL441, "AGL kappa, paired bootstrap",
                          n=len(agl)), rel(path),
            note="recall10 CIs resample duplicates, whose ties np.argsort breaks by its own (CONVENTION) order")


# =============================================================================
#  10. Step 63 (Table 1), step 65 (leakage), step 25 (calibration)
# =============================================================================
def step63():
    print("step 63 ...", flush=True)
    F = "63 Table 1 experiment"
    path = os.path.join(RES, "63_experimental_vs_predicted.csv")
    d = pd.read_csv(path)
    new = {}
    for src in ["cgcnn", "alignn"]:
        pre, gam = slack(d[f"K_{src}"], d[f"G_{src}"], d.density_g_cm3, d.volume_A3, d.n_atoms)
        new[f"kappa_{src}_derived_gamma"] = pre * np.exp(-gam)
        new[f"gamma_derived_{src}"] = gam
        new[f"kappa_{src}_equal_footing"] = pre * np.exp(-d.gamma_paper.to_numpy())   # Table 1's gamma

    def mae(pred):
        p, e = np.asarray(pred, float), d.kappa_exp.to_numpy(float)
        ok = np.isfinite(p) & np.isfinite(e) & (p > 0) & (e > 0)
        return np.abs(np.log10(p[ok]) - np.log10(e[ok])).mean(), int(ok.sum())

    best = min(["cgcnn", "alignn"], key=lambda s_: mae(new[f"kappa_{s_}_derived_gamma"])[0])
    new["kappa_pred_derived_gamma"] = new[f"kappa_{best}_derived_gamma"]
    new["kappa_pred_equal_footing"] = new[f"kappa_{best}_equal_footing"]
    new["best_model"] = np.full(len(d), "ALIGNN" if best == "alignn" else "CGCNN ensemble")
    for c, src in [("derived_gamma", "kappa_pred_derived_gamma"), ("equal_footing", "kappa_pred_equal_footing"),
                   ("pink", "kappa_pink")]:
        p = new[src] if src in new else d[src].to_numpy()
        new[f"abs_log10_err_{c}"] = np.abs(np.log10(p) - np.log10(d.kappa_exp.to_numpy()))
    w = lab(F, "CGCNN ens / ALIGNN ens", "per-crystal table", "matbench", TAB1, "kappa_L (experiment)")
    for c, v in new.items():
        check_column(v, d[c], c, rel(path), **w)
    for name, col in [("CGCNN ens, own Poisson gamma", "kappa_cgcnn_derived_gamma"),
                      ("ALIGNN ens, own Poisson gamma", "kappa_alignn_derived_gamma"),
                      ("CGCNN ens, Table 1's gamma", "kappa_cgcnn_equal_footing"),
                      ("ALIGNN ens, Table 1's gamma", "kappa_alignn_equal_footing"),
                      ("PINK paper's own kappa", "kappa_pink")]:
        v, n = mae(new[col] if col in new else d[col])
        m.record(metric="MAE_log10 vs experiment", value=v, n=n,
                 note="printed by step 63, never saved",
                 **lab(F, name.split(",")[0], name, "matbench" if "PINK" not in name else "PINK (paper)",
                       TAB1, "kappa_L (experiment)"))


def step65():
    print("step 65 ...", flush=True)
    F = "65 leakage audit"
    labels = pd.read_csv(os.path.join(ROOT, "data_full", "labels.csv"))
    k_all = pd.read_csv(os.path.join(RES, "predictions_K_VRH_ens.csv"))
    labels["split"] = labels.mb_id.map(k_all.set_index("material_id").split)
    assert labels.split.notna().all()
    labels["reduced"] = [Composition(f).reduced_formula for f in labels.formula]
    reduced = labels.set_index("mb_id").reduced
    train_formulas = set(labels.reduced[labels.split == "train"])
    frames = []
    for t in "KG":
        p = pd.read_csv(os.path.join(RES, f"predictions_{t}_VRH_ens.csv"))
        p = p[p.split == "test"]
        frames.append(pd.DataFrame({"model": "CGCNN ens", "target": t, "material_id": p.material_id,
                                    "err": (p.pred_log10 - p.true_log10).abs()}))
    a = pd.read_csv(os.path.join(m.ALI, "18_alignn_ensemble_predictions.csv"))
    frames.append(pd.DataFrame({"model": "ALIGNN ens", "target": a.target, "material_id": a.material_id,
                                "err": (a.pred_log10 - a.true_log10).abs()}))
    allp = pd.concat(frames, ignore_index=True)
    allp["seen"] = allp.material_id.map(reduced).isin(train_formulas)
    rng = np.random.default_rng(0)                   # ONE generator shared across groups (CONVENTION)
    rows = []
    for (model, target), g in allp.groupby(["model", "target"]):     # sorted: ALIGNN G, ALIGNN K, CGCNN G, CGCNN K
        seen, unseen = g.err[g.seen].to_numpy(), g.err[~g.seen].to_numpy()
        for err, name in [(unseen, "formula NOT seen in train"), (seen, "formula ALSO in train")]:
            rows.append({"model": model, "target": target, "group": name, "n": len(err),
                         "mae_log10": err.mean(), "relative_error_pct": 100 * (10 ** err.mean() - 1)})
        diffs = np.empty(2000)
        for i in range(2000):                        # draw order: seen, then unseen, each round
            diffs[i] = (rng.choice(seen, len(seen), replace=True).mean()
                        - rng.choice(unseen, len(unseen), replace=True).mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"model": model, "target": target, "group": "ALL", "n": len(g),
                     "mae_log10": g.err.mean(), "relative_error_pct": 100 * (10 ** g.err.mean() - 1),
                     "optimism_vs_unseen": g.err.mean() - unseen.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi})
    new = pd.DataFrame(rows)
    path = os.path.join(RES, "65_leakage_audit.csv")
    old = pd.read_csv(path)
    where = lambda r: lab(F, r.model, r.group, "matbench", MB, f"{r.target}_VRH (log10)", n=int(r.n))  # noqa: E731
    is_all = old.group == "ALL"
    compare(new[~is_all.values], old[~is_all], ["model", "target", "group"],
            ["n", "mae_log10", "relative_error_pct"], where, rel(path))
    compare(new[is_all.values], old[is_all], ["model", "target", "group"],
            ["n", "mae_log10", "relative_error_pct", "optimism_vs_unseen", "diff_ci_lo", "diff_ci_hi"],
            where, rel(path))


def step25():
    print("step 25 ...", flush=True)
    F = "25 interval calibration"
    rows = {L: {"nominal": L} for L in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    for t in ["K_VRH", "G_VRH"]:
        f = pd.read_csv(os.path.join(RES, f"predictions_{t}_ens.csv"))
        val, test = f[f.split == "val"], f[f.split == "test"]
        z_val = (np.abs(val.true_log10 - val.pred_log10) / val.member_std_log10).to_numpy()
        miss = np.abs(test.true_log10 - test.pred_log10).to_numpy()   # how far off each test crystal is
        sd = test.member_std_log10.to_numpy()
        for L, row in rows.items():
            z = norm.ppf(0.5 + L / 2)                     # normal: +-z sd holds a fraction L
            c = np.nanquantile(z_val, L)                  # val's own L-quantile (linear, NaN skipped)
            row[f"{t}_raw"] = np.mean(miss <= z * sd)
            row[f"{t}_recalibrated"] = np.mean(miss <= c * sd)
            row[f"{t}_scale_factor"] = c
    path = os.path.join(RES, "calibration_check.csv")
    old = pd.read_csv(path)
    compare(pd.DataFrame(list(rows.values())), old, ["nominal"], [c for c in old.columns if c != "nominal"],
            lambda r: lab(F, "CGCNN 3-ensemble", f"coverage at nominal {r.nominal:.1f}", "matbench", MB,
                          "K_VRH and G_VRH (log10)", n=1648), rel(path))


def main():
    step82()
    s87, pool = step87()
    step89(s87, pool)
    direct02()
    step44()
    runs = distinct_runs()
    step47(runs)
    step48(runs)
    step50(runs)
    step63()
    step65()
    step25()
    m.finish()


if __name__ == "__main__":
    main()
