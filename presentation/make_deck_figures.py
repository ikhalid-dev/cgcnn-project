#!/usr/bin/env python3
"""
make_deck_figures.py - draws the charts of the ALIGNN/PINK deck that were
redrawn in the October 2026 clean-up, and writes every number those slides
quote into one file, so no slide number is typed in by hand.

How to run (from the project folder, in the ml_env Python):

    ~/miniconda3/envs/ml_env/bin/python presentation/make_deck_figures.py

What it writes, all into presentation/figures/:

    deck_training_data.png  + .csv   slide 11  what the models were trained on
    deck_parity.png         + .csv   slide 12  predicted vs DFT, both ensembles
    deck_residuals.png      + .csv   slide 13  error by true-kappa fifth
    deck_ensembling.png     + .csv   slide 14  one seed vs three seeds averaged
    deck_calibration.png    + .csv   slide 15  the uncertainty interval
    deck_measured.png       + .csv   slide 18  45 measured kappa_L
    deck_pink_agreement.png + .csv   slide 19  ALIGNN vs CGCNN, 1,213 crystals
    deck_candidates.png     + .csv   slide 22  the 33,053 screened crystals
    deck_screen_agreement.png + .csv slide 23  ALIGNN vs CGCNN over the screen
    deck_elements_rows.tex  + .csv   slide 24  element table rows
    deck_numbers.tex        + .csv   every number above, as \\def macros

Every chart uses deck_style.py, so each model has one fixed colour everywhere.

Python notes for readers new to the language
--------------------------------------------
* A line starting with # is a comment; Python ignores it.
* `import pandas as pd` loads the pandas table library under the short name pd.
* pd.read_csv(path) reads a CSV file into a "DataFrame" (a table).
  table["col"] picks one column; table[table["col"] > 1] keeps matching rows.
* def name(args): starts a function; its indented lines are its body.
* f"text {x:.3f}" is an "f-string": {x:.3f} is replaced by x with 3 decimals.
"""

# =============================================================================
#  CONFIG - every path, size, threshold and choice of model lives here
# =============================================================================
import os

HERE = os.path.dirname(os.path.abspath(__file__))   # the presentation/ folder
ROOT = os.path.dirname(HERE)                         # the project folder
FIGS = os.path.join(HERE, "figures")                 # where outputs go


def p(*parts):
    """Join path pieces onto the project folder: p("results", "x.csv")."""
    return os.path.join(ROOT, *parts)


# --- input files (read only; this script never changes them) ---------------
FINAL_METRICS   = p("results/cgcnn/95_final_all_metrics.csv")   # step 95: every metric
MATBENCH_LABELS = p("data_full/labels.csv")                     # 10,987 training crystals
PINK_CIF_DIR    = p("complete-data")                            # PINK's 1,213 prediction CIFs
CGCNN_PRED      = p("results/cgcnn/predictions_{t}_VRH_{run}.csv")   # {t}=K/G, {run}=ens/full
ALIGNN_ENS_PRED = p("results/alignn/18_alignn_ensemble_predictions.csv")
ALIGNN_S42_PRED = {"K": p("results/alignn/alignn_bulk_modulus_kv_s42/prediction_results_test_set.csv"),
                   "G": p("results/alignn/alignn_shear_modulus_gv_s42/prediction_results_test_set.csv")}
CALIBRATION     = p("results/cgcnn/calibration_check.csv")      # step 25
MEASURED        = p("results/cgcnn/63_experimental_vs_predicted.csv")   # step 63, PINK Table 1
CGCNN_PINK      = p("results/cgcnn/pink_kappa_predictions.csv")
ALIGNN_PINK     = p("results/alignn/alignn_kappa_predictions.csv")
CGCNN_PINK_MOD  = p("results/cgcnn/pink_moduli_predictions.csv")
ALIGNN_PINK_MOD = p("results/alignn/alignn_moduli_predictions.csv")
SCREEN_ALIGNN   = p("results/alignn/57_gnome_screen_alignn.csv")    # ALIGNN on the screen
SCREEN_CGCNN    = p("results/cgcnn/39_gnome_screen_all_gamma.csv")  # CGCNN ens on the screen
SCREEN_FILTERED = p("results/cgcnn/gnome_screen_all.csv")       # every crystal passing the filters
GNOME_SUMMARY   = p("gnome_data/stable_materials_summary.csv")  # GNoME's own table
PINK_LIST       = p("external/AI4Kappa/JMI_Supporting_Information/Nature-filtered-low-kappa.csv")

# --- which column holds each model's kappa_L on the screen -----------------
SCREEN_MODELS = {"CGCNN ensemble": "Kappa_cal_derived_matbench",
                 "ALIGNN":         "Kappa_alignn"}

# --- thresholds and choices ------------------------------------------------
LOW_KAPPA = 1.0              # W/m/K: the screen calls kappa_L <= 1 "low"
MIN_ELEMENT_COUNT = 400      # elements in fewer candidates are left out of slide 24
# slide 18's worked examples: the softest, a middle one and the hardest measured crystal
MEASURED_EXAMPLES = {"AgCl": "AgCl", "PbTe": "PbTe", "C": "C (diamond)"}  # formula: label
N_ELEMENTS_TABLE = 10        # 5 most enriched + 5 most depleted
N_TOP_ELEMENTS_PLOT = 20     # bars in the training-data element panel
MAX_ATOMS_SHOWN = 40         # cell-size panel: bigger cells are pooled into one last bar
SYMPREC_RETRY = 0.1          # looser symmetry tolerance for CIFs that fail at the default
REUSE_SYMMETRY_CACHE = True  # True = do not re-parse 1,213 CIFs if the cache CSV exists
N_BOOTSTRAP = 10000          # resamples for the ensembling confidence interval
RANDOM_SEED = 0

# --- figure sizes in inches (width, height); slides are 16:9 ---------------
SIZE_TRAINING  = (11.0, 6.2)
SIZE_PARITY    = (8.4, 7.6)
SIZE_RESIDUALS = (10.0, 4.0)
SIZE_ENSEMBLE  = (10.0, 4.0)
SIZE_CALIB     = (5.6, 5.0)
SIZE_MEASURED  = (6.0, 5.4)
SIZE_PINK      = (5.6, 5.2)
SIZE_SCREEN    = (5.8, 5.2)
SIZE_CANDID    = (11.0, 4.2)

# --- axis limits (GPa) for the parity plot; None = work them out from data -
PARITY_LIMITS = {"K": None, "G": None}
# =============================================================================


import re
import collections
import warnings
import numpy as np                    # numbers and arrays
import pandas as pd                   # tables
from scipy import stats               # Pearson r, Spearman rho
import matplotlib.pyplot as plt
import deck_style as ds               # the shared look (presentation/deck_style.py)

warnings.filterwarnings("ignore")     # pymatgen prints many harmless CIF warnings
ds.apply()
os.makedirs(FIGS, exist_ok=True)      # make the folder if it is missing

# Every number a slide quotes is collected here as (name, value, source, how).
# At the end it is written to deck_numbers.csv and deck_numbers.tex.
NUMBERS = []


def record(name, value, source, how):
    """Remember one number for the slides. name becomes the LaTeX macro \\D<name>."""
    NUMBERS.append({"name": name, "value": value, "source": source, "how": how})


def out(name):
    """Full path of an output file in presentation/figures/."""
    return os.path.join(FIGS, name)


def mae_log(true, pred):
    """Mean absolute error in log10 space: the average of |log10 pred - log10 true|."""
    return float(np.mean(np.abs(np.log10(pred) - np.log10(true))))


def r2_log(true, pred):
    """R^2 of log10 values: 1 - (squared error) / (spread of the truth)."""
    t, q = np.log10(true), np.log10(pred)
    return float(1 - np.sum((q - t) ** 2) / np.sum((t - t.mean()) ** 2))


def fmt_int(n):
    """12345 -> '12{,}345' (LaTeX thousands separator)."""
    return f"{int(n):,}".replace(",", "{,}")


# =============================================================================
#  The step 95 table: the single source of truth for accuracy numbers
# =============================================================================
final = pd.read_csv(FINAL_METRICS)
final = final[final["primary"] == True]          # one row per (model, metric)


def final_value(run, target, metric, split="test"):
    """Look up one number in the step 95 table. Stops loudly if it is not
    there exactly once, so a slide can never quote a missing number."""
    rows = final[(final["run"] == run) & (final["target"] == target)
                 & (final["metric"] == metric) & (final["split"] == split)]
    if len(rows) != 1:
        raise SystemExit(f"95 table: {len(rows)} rows for {run}/{target}/{metric}/{split}")
    return float(rows["value"].iloc[0])


# =============================================================================
#  Slide 11 - the training data
# =============================================================================
def crystal_system(number):
    """International space-group number (1-230) -> crystal system name."""
    for upper, name in [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
                        (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
                        (230, "cubic")]:
        if number <= upper:
            return name


SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal",
           "trigonal", "hexagonal", "cubic"]


def pink_symmetry():
    """Space group of every PINK CIF. Cached, because parsing takes minutes."""
    cache = out("deck_training_data_pink_symmetry.csv")
    if REUSE_SYMMETRY_CACHE and os.path.exists(cache):
        return pd.read_csv(cache)
    from pymatgen.core import Structure          # imported here: only needed once
    rows = []
    for fn in sorted(os.listdir(PINK_CIF_DIR)):  # every file in the folder, sorted
        if not fn.endswith(".cif"):
            continue
        path = os.path.join(PINK_CIF_DIR, fn)
        st = Structure.from_file(path)
        try:                                     # try ... except: run, and catch a failure
            num, how = st.get_space_group_info()[1], "symprec 0.01"   # default tolerance
        except Exception:
            try:
                num = st.get_space_group_info(symprec=SYMPREC_RETRY)[1]
                how = f"symprec {SYMPREC_RETRY}"
            except Exception:
                # Last resort: the space group the CIF file declares in its own
                # header. Needed for mp-770620, whose CIF has two atoms 0.0009 A
                # apart, so no symmetry finder can work on it.
                text = open(path).read()
                num = int(re.search(r"_symmetry_Int_Tables_number\s+(\d+)", text).group(1))
                how = "CIF header"
        rows.append({"cif": fn, "space_group_number": num, "found_by": how,
                     "crystal_system": crystal_system(num)})
    table = pd.DataFrame(rows)
    table.to_csv(cache, index=False)
    return table


def fig_training_data():
    lab = pd.read_csv(MATBENCH_LABELS)
    sym = pink_symmetry()
    n_mb, n_pink = len(lab), len(sym)
    n_retry = int((sym["found_by"] != "symprec 0.01").sum())   # != means "is not"

    fig, axes = plt.subplots(2, 2, figsize=SIZE_TRAINING)
    # axes is a 2x2 grid; axes[0, 1] = top row, right column

    # A - the two targets, in log10 (why the models train in log10)
    ax = axes[0, 0]
    bins = np.linspace(-0.5, 2.8, 50)
    ax.hist(np.log10(lab["K_VRH"]), bins=bins, histtype="step", lw=1.8,
            color=ds.INK, label="bulk $K$")
    ax.hist(np.log10(lab["G_VRH"]), bins=bins, histtype="stepfilled", alpha=0.45,
            color=ds.MUTED, label="shear $G$")
    ax.set_xlabel("$\\log_{10}$ modulus (GPa)")
    ax.set_ylabel("crystals")
    ax.legend(frameon=False, loc="upper left")
    ds.panel_title(ax, "A", f"the two targets   (matbench, n = {n_mb:,})")
    ds.tidy(ax)

    # B - atoms per cell
    ax = axes[0, 1]
    # np.clip(..., upper=M + 1) puts every cell bigger than M into ONE last bar,
    # so no crystal silently drops off the right edge of the plot.
    n_big = int((lab["n_sites"] > MAX_ATOMS_SHOWN).sum())
    sizes = np.clip(lab["n_sites"], None, MAX_ATOMS_SHOWN + 1)
    ax.hist(sizes, bins=np.arange(0.5, MAX_ATOMS_SHOWN + 2.5), color=ds.MUTED,
            edgecolor="white", lw=0.4)
    ticks = list(range(0, MAX_ATOMS_SHOWN, 10)) + [MAX_ATOMS_SHOWN + 1]   # 0,10,20,30 then 'M+'
    ax.set_xticks(ticks)
    # the last bar holds cells with MORE than M atoms, so it is labelled ">M"
    ax.set_xticklabels([str(t) for t in ticks[:-1]] + [f">{MAX_ATOMS_SHOWN}"])
    ax.text(MAX_ATOMS_SHOWN + 1, ax.get_ylim()[1] * 0.12, f"{n_big}", ha="center",
            fontsize=9, color=ds.INK)
    ax.set_xlabel("atoms per unit cell")
    ax.set_ylabel("crystals")
    ds.panel_title(ax, "B", f"cell size   (matbench, n = {n_mb:,})")
    ds.tidy(ax)

    # C - crystal system of the PINK prediction set (matbench structures are
    #     not stored offline, only their formulas and atom counts)
    ax = axes[1, 0]
    counts = sym["crystal_system"].value_counts()          # how many of each
    vals = [int(counts.get(s, 0)) for s in SYSTEMS]
    ax.bar(range(len(SYSTEMS)), vals, color=ds.MUTED, edgecolor="white")
    for i, v in enumerate(vals):                           # enumerate gives (index, value)
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=9)
    ax.set_xticks(range(len(SYSTEMS)))
    ax.set_xticklabels(SYSTEMS, rotation=25, ha="right")
    ax.set_ylabel("crystals")
    ds.panel_title(ax, "C", f"crystal system   (PINK prediction set, n = {n_pink:,})")
    ds.tidy(ax)

    # D - most common elements in matbench
    from pymatgen.core import Composition
    freq = collections.Counter()                           # a dict that counts
    for f in lab["formula"]:
        for el in Composition(str(f)).elements:
            freq[el.symbol] += 1
    top = freq.most_common(N_TOP_ELEMENTS_PLOT)            # [(symbol, count), ...]
    ax = axes[1, 1]
    ax.bar(range(len(top)), [c for _, c in top], color=ds.MUTED, edgecolor="white")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([s for s, _ in top], fontsize=9)
    ax.set_ylabel("crystals containing it")
    ds.panel_title(ax, "D", f"{N_TOP_ELEMENTS_PLOT} most common elements   (matbench)")
    ds.tidy(ax)

    fig.tight_layout()
    ds.save(fig, out("deck_training_data.png"))
    pd.DataFrame({"crystal_system": SYSTEMS, "pink_set_count": vals}).to_csv(
        out("deck_training_data.csv"), index=False)

    record("MatbenchN", fmt_int(n_mb), MATBENCH_LABELS, "rows")
    record("PinkN", fmt_int(n_pink), PINK_CIF_DIR, "CIF files")
    record("PinkSymRetry", n_retry, "deck_training_data_pink_symmetry.csv",
           "CIFs whose symmetry was not found at the default tolerance")
    record("MatbenchBigCells", n_big, MATBENCH_LABELS, f"crystals with > {MAX_ATOMS_SHOWN} atoms")
    # .median() = the middle value once sorted: half the cells are smaller, half bigger
    record("MatbenchMedianSites", f"{lab['n_sites'].median():.0f}", MATBENCH_LABELS,
           "median atoms per cell")
    # freq["O"] = how many matbench crystals contain oxygen (counted in panel D)
    record("MatbenchOxygenPct", f"{100 * freq['O'] / n_mb:.0f}", MATBENCH_LABELS,
           "percent of crystals containing O")
    # Slide 11's takeaway says "oxygen is the most common element". top[0] is the
    # (symbol, count) pair with the highest count, so top[0][0] is its symbol.
    # Stop the build if that ever stops being true, instead of printing a false line.
    if top[0][0] != "O":
        raise SystemExit(f"slide 11 says O is the most common element, but it is {top[0][0]}")
    record("MatbenchNoOxygenPct", f"{100 * (n_mb - freq['O']) / n_mb:.0f}", MATBENCH_LABELS,
           "percent of crystals with NO O; O is the most common element (checked)")
    # .value_counts() = how many rows have each value of the "found_by" column
    print(f"slide 11: matbench {n_mb}, PINK CIFs {n_pink}; how each space group was found:",
          sym["found_by"].value_counts().to_dict())


# =============================================================================
#  Slides 12-14 - accuracy on the 1,648 held-out test crystals
# =============================================================================
def test_predictions():
    """Both ensembles' test-set predictions in one table, one row per crystal
    and modulus. Columns: model, target, material_id, true_GPa, pred_GPa."""
    parts = []
    for t in ("K", "G"):
        c = pd.read_csv(CGCNN_PRED.format(t=t, run="ens"))
        c = c[c["split"] == "test"]
        parts.append(pd.DataFrame({"model": "CGCNN ensemble", "target": t,
                                   "material_id": c["material_id"],
                                   "true_GPa": c["true_GPa"], "pred_GPa": c["pred_GPa"]}))
    a = pd.read_csv(ALIGNN_ENS_PRED)
    parts.append(pd.DataFrame({"model": "ALIGNN", "target": a["target"],
                               "material_id": a["material_id"],
                               "true_GPa": a["true_GPa"], "pred_GPa": a["pred_GPa"]}))
    return pd.concat(parts, ignore_index=True)       # stack the tables vertically


def fig_parity(pred):
    fig, axes = plt.subplots(2, 2, figsize=SIZE_PARITY)
    rows = []
    for i, t in enumerate(("K", "G")):
        sub = pred[pred["target"] == t]
        lim = PARITY_LIMITS[t] or (0.8 * min(sub["true_GPa"].min(), sub["pred_GPa"].min()),
                                   1.25 * max(sub["true_GPa"].max(), sub["pred_GPa"].max()))
        for j, model in enumerate(("CGCNN ensemble", "ALIGNN")):
            d = sub[sub["model"] == model]
            m, r2 = mae_log(d["true_GPa"], d["pred_GPa"]), r2_log(d["true_GPa"], d["pred_GPa"])
            rows.append({"model": model, "target": t, "n": len(d),
                         "mae_log10": m, "r2_log10": r2,
                         "min_pred_GPa": d["pred_GPa"].min()})
            ax = axes[i, j]
            ax.scatter(d["true_GPa"], d["pred_GPa"], s=7, alpha=0.35, lw=0,
                       color=ds.MODEL_COLOURS[model])
            ax.plot(lim, lim, color=ds.INK, lw=0.9, ls="--")        # the y = x line
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.text(0.04, 0.96, f"MAE log$_{{10}}$ = {m:.4f}\n$R^2$ = {r2:.3f}\nn = {len(d):,}",
                    transform=ax.transAxes, va="top", fontsize=10)
            name = "bulk $K$" if t == "K" else "shear $G$"
            ax.set_title(f"{model}, {name}", loc="left", color=ds.MODEL_COLOURS[model])
            ax.set_xlabel("DFT (GPa)")
            ax.set_ylabel("predicted (GPa)")
            ds.tidy(ax, grid="both")
    fig.tight_layout()
    ds.save(fig, out("deck_parity.png"))
    table = pd.DataFrame(rows)
    table.to_csv(out("deck_parity.csv"), index=False)

    # cross-check against the step 95 table (it must agree to 4 decimals)
    runs = {("CGCNN ensemble", "K"): "K_VRH_ens", ("CGCNN ensemble", "G"): "G_VRH_ens",
            ("ALIGNN", "K"): "18_alignn_ensemble (K)", ("ALIGNN", "G"): "18_alignn_ensemble (G)"}
    for r in rows:
        want = final_value(runs[(r["model"], r["target"])], f"{r['target']}_VRH", "MAE_log10")
        if abs(want - r["mae_log10"]) > 5e-5:
            raise SystemExit(f"parity MAE {r} disagrees with 95 table {want}")
    print("slide 12: parity MAEs match the 95 table;",
          f"lowest ALIGNN test prediction {table[table.model == 'ALIGNN'].min_pred_GPa.min():.3f} GPa")
    for r in rows:
        key = ("Cg" if r["model"].startswith("CGCNN") else "Al") + r["target"]
        record(f"Parity{key}Rtwo", f"{r['r2_log10']:.3f}", "deck_parity.csv", "R^2 of log10, test set")
        # the smallest prediction made for any test crystal (shows none hit the 1e-3 floor)
        record(f"Parity{key}Min", f"{r['min_pred_GPa']:.3f}", "deck_parity.csv",
               "lowest test prediction, GPa")


def fig_residuals():
    """MAE by fifth of the test set, ranked by TRUE kappa_L (Q1 = lowest kappa,
    the fifth a screen reads). Numbers come straight from the step 95 table."""
    runs = {"CGCNN ensemble": "{K,G}_VRH_ens", "ALIGNN": "18_alignn_ensemble"}
    # BOTH runs above are 3-model ensembles (like with like). The dict keys are
    # also colour names in deck_style, so the legend text is set separately here.
    LEGEND = {"CGCNN ensemble": "CGCNN ensemble (3 models)",
              "ALIGNN": "ALIGNN ensemble (3 models)"}
    rows = []
    for model, run in runs.items():                # .items() gives (key, value) pairs
        for q in range(1, 6):
            row = {"model": model, "quintile": f"Q{q}"}
            for t in ("K", "G"):
                row[f"mae_log_{t}"] = final_value(run, "kappa quintiles", f"Q{q}_mae_log_{t}")
            row["n"] = final_value(run, "kappa quintiles", f"Q{q}_n")
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out("deck_residuals.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=SIZE_RESIDUALS, sharey=True)
    x = np.arange(5)
    BAR_W = 0.42                       # width of one bar; two bars per quintile
    for ax, t, name in zip(axes, ("K", "G"), ("bulk $K$", "shear $G$")):
        for k, model in enumerate(runs):
            v = table[table["model"] == model][f"mae_log_{t}"].values
            ax.bar(x + (k - 0.5) * BAR_W, v, width=BAR_W, color=ds.MODEL_COLOURS[model],
                   label=LEGEND[model])
            for xi, vi in zip(x, v):
                ax.text(xi + (k - 0.5) * BAR_W, vi + 0.002, f"{vi:.3f}",
                        ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1\nlowest $\\kappa_L$", "Q2", "Q3", "Q4", "Q5\nhighest"])
        ax.set_ylim(0, 1.12 * table[["mae_log_K", "mae_log_G"]].values.max())
        ax.set_title(name, loc="left")
        ds.tidy(ax)
    axes[0].set_ylabel("MAE $\\log_{10}$ (test set)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    ds.save(fig, out("deck_residuals.png"))

    for model, key in (("CGCNN ensemble", "Cg"), ("ALIGNN", "Al")):
        v = table[table["model"] == model].set_index("quintile")["mae_log_G"]
        record(f"QOneG{key}", f"{v['Q1']:.3f}", FINAL_METRICS, f"{model} Q1_mae_log_G")
        record(f"QFiveG{key}", f"{v['Q5']:.3f}", FINAL_METRICS, f"{model} Q5_mae_log_G")
        record(f"QRatioG{key}", f"{v['Q1'] / v['Q5']:.1f}", FINAL_METRICS, "Q1 / Q5")
    record("QuintN", int(table["n"].iloc[0]), FINAL_METRICS, "crystals in Q1")
    print("slide 13:", table.round(3).to_string(index=False))


def bootstrap_gain(true, single, ens, rng):
    """Paired bootstrap of (ensemble MAE - single MAE) in log10.
    Resample the SAME crystals for both models each time; return 95% interval."""
    e1 = np.abs(np.log10(single) - np.log10(true))
    e3 = np.abs(np.log10(ens) - np.log10(true))
    idx = rng.integers(0, len(true), (N_BOOTSTRAP, len(true)))   # random rows
    diff = e3[idx].mean(axis=1) - e1[idx].mean(axis=1)
    return np.percentile(diff, [2.5, 97.5])


def fig_ensembling(pred):
    runs = {"CGCNN": {"K": ["K_VRH_full", "K_VRH_s1", "K_VRH_s2", "K_VRH_ens"],
                      "G": ["G_VRH_full", "G_VRH_s1", "G_VRH_s2", "G_VRH_ens"]},
            "ALIGNN": {"K": ["alignn_bulk_modulus_kv_s42", "alignn_bulk_modulus_kv_s1",
                             "alignn_bulk_modulus_kv_s2", "18_alignn_ensemble (K)"],
                       "G": ["alignn_shear_modulus_gv_s42", "alignn_shear_modulus_gv_s1",
                             "alignn_shear_modulus_gv_s2", "18_alignn_ensemble (G)"]}}
    labels = ["seed 42", "seed 1", "seed 2", "3 averaged"]
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for fam, by_t in runs.items():
        for t, rl in by_t.items():
            v = [final_value(r, f"{t}_VRH", "MAE_log10") for r in rl]
            mean_single = np.mean(v[:3])
            row = {"model": fam, "target": t, **dict(zip(labels, v)),
                   "gain_vs_seed42_pct": 100 * (v[0] - v[3]) / v[0],
                   "gain_vs_mean_single_pct": 100 * (mean_single - v[3]) / mean_single}
            # paired bootstrap: ensemble vs the seed-42 single model
            if fam == "CGCNN":
                s = pd.read_csv(CGCNN_PRED.format(t=t, run="full"))
                s = s[s["split"] == "test"][["material_id", "pred_GPa"]]
            else:
                s = pd.read_csv(ALIGNN_S42_PRED[t], skipinitialspace=True)
                s.columns = [c.strip() for c in s.columns]       # remove stray spaces
                s = s.rename(columns={"id": "material_id", "prediction": "pred_GPa"})
                s["pred_GPa"] = s["pred_GPa"].clip(lower=1e-3)   # same floor as step 17
            model = "CGCNN ensemble" if fam == "CGCNN" else "ALIGNN"
            e = pred[(pred["model"] == model) & (pred["target"] == t)]
            m = e.merge(s, on="material_id", suffixes=("_ens", "_single"))
            lo, hi = bootstrap_gain(m["true_GPa"].values, m["pred_GPa_single"].values,
                                    m["pred_GPa_ens"].values, rng)
            row.update({"n_paired": len(m), "ci95_lo": lo, "ci95_hi": hi})
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out("deck_ensembling.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=SIZE_ENSEMBLE, sharey=True)
    for ax, t, name in zip(axes, ("K", "G"), ("bulk $K$", "shear $G$")):
        for k, fam in enumerate(("CGCNN", "ALIGNN")):
            r = table[(table["model"] == fam) & (table["target"] == t)].iloc[0]
            light = ds.MODEL_COLOURS[f"{fam} single"]
            dark = ds.MODEL_COLOURS["CGCNN ensemble" if fam == "CGCNN" else "ALIGNN"]
            xs = np.arange(4) + 5 * k
            ax.bar(xs, [r[l] for l in labels], color=[light] * 3 + [dark], width=0.8)
            for x_, l in zip(xs, labels):
                ax.text(x_, r[l] + 0.001, f"{r[l]:.4f}", ha="center", fontsize=7.5, rotation=90,
                        va="bottom")
            # get_xaxis_transform(): x in data units, y as a fraction of the panel
            # height, so y = -0.24 sits just below the tick labels
            ax.text(xs[1] + 0.5, -0.24, fam, ha="center", va="top", color=dark,
                    fontsize=11, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.set_xticks(list(range(4)) + list(range(5, 9)))
        tick_text = ["seed\n42", "seed\n1", "seed\n2", "ensemble"]
        ax.set_xticklabels(tick_text * 2, fontsize=8.5)
        ax.set_ylim(0, 0.105)
        ax.set_title(name, loc="left")
        ds.tidy(ax)
    axes[0].set_ylabel("MAE $\\log_{10}$ (test set, 1,648)")
    fig.tight_layout()
    ds.save(fig, out("deck_ensembling.png"))

    for _, r in table.iterrows():                        # loop over table rows
        key = ("Cg" if r["model"] == "CGCNN" else "Al") + r["target"]
        record(f"EnsGainSeed{key}", f"{r['gain_vs_seed42_pct']:.1f}", "deck_ensembling.csv",
               "% lower MAE log10, ensemble vs the seed-42 single model")
        record(f"EnsGainMean{key}", f"{r['gain_vs_mean_single_pct']:.1f}", "deck_ensembling.csv",
               "% lower MAE log10, ensemble vs the mean of its 3 members")
        record(f"EnsCi{key}", f"[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}]".replace("+", "{+}"),
               "deck_ensembling.csv", "95% paired-bootstrap CI of (ens - seed 42) MAE log10")
    for fam, key in (("CGCNN", "Cg"), ("ALIGNN", "Al")):
        for t in ("K", "G"):
            r = table[(table["model"] == fam) & (table["target"] == t)].iloc[0]
            record(f"OneMae{key}{t}", f"{r['seed 42']:.4f}", FINAL_METRICS, "seed 42 MAE log10")
            record(f"EnsMae{key}{t}", f"{r['3 averaged']:.4f}", FINAL_METRICS, "3-ensemble MAE log10")
    print("slide 14:\n", table.round(4).to_string(index=False))


# =============================================================================
#  Slide 15 - is the p05-p95 interval honest?
# =============================================================================
def fig_calibration():
    c = pd.read_csv(CALIBRATION)
    fig, ax = plt.subplots(figsize=SIZE_CALIB)
    blue = ds.MODEL_COLOURS["CGCNN ensemble"]
    for t, mark in (("K", "o"), ("G", "s")):
        ax.plot(c["nominal"], c[f"{t}_VRH_raw"], marker=mark, ls="--", color=ds.MUTED,
                label=f"{t}, as produced")
        ax.plot(c["nominal"], c[f"{t}_VRH_recalibrated"], marker=mark, color=blue,
                label=f"{t}, widened (factor fit on validation)")
    ax.plot([0, 1], [0, 1], color=ds.INK, lw=0.9, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("interval claims to cover")
    ax.set_ylabel("actually covers (test set)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ds.tidy(ax, grid="both")
    fig.tight_layout()
    ds.save(fig, out("deck_calibration.png"))
    c.to_csv(out("deck_calibration.csv"), index=False)

    r90 = c[np.isclose(c["nominal"], 0.9)].iloc[0]
    record("CalRawK", f"{100 * r90['K_VRH_raw']:.0f}", CALIBRATION, "K coverage at nominal 90%")
    record("CalRawG", f"{100 * r90['G_VRH_raw']:.0f}", CALIBRATION, "G coverage at nominal 90%")
    record("CalFixK", f"{100 * r90['K_VRH_recalibrated']:.0f}", CALIBRATION, "K after widening")
    record("CalFixG", f"{100 * r90['G_VRH_recalibrated']:.0f}", CALIBRATION, "G after widening")
    # step 25 reports the widening as scale_factor / 1.645 (1.645 = the z of a 90% interval)
    record("CalWidenK", f"{r90['K_VRH_scale_factor'] / 1.645:.2f}", CALIBRATION, "scale/1.645")
    record("CalWidenG", f"{r90['G_VRH_scale_factor'] / 1.645:.2f}", CALIBRATION, "scale/1.645")


# =============================================================================
#  Slide 18 - 45 materials with measured kappa_L (PINK's Table 1)
# =============================================================================
def fig_measured():
    d = pd.read_csv(MEASURED)
    err = lambda col: float(np.mean(np.abs(np.log10(d[col]) - np.log10(d["kappa_exp"]))))
    # (lambda = a one-line function: err("x") = mean |log10 x - log10 measured|)
    series = [("CGCNN ens., $\\gamma$ from its own $K,G$", "kappa_cgcnn_derived_gamma",
               "o", "none", ds.MODEL_COLOURS["CGCNN ensemble"]),
              ("CGCNN ens., paper's tabulated $\\gamma$", "kappa_cgcnn_equal_footing",
               "o", ds.MODEL_COLOURS["CGCNN ensemble"], ds.MODEL_COLOURS["CGCNN ensemble"]),
              ("PINK paper", "kappa_pink", "x", ds.MODEL_COLOURS["PINK paper"],
               ds.MODEL_COLOURS["PINK paper"])]
    fig, ax = plt.subplots(figsize=SIZE_MEASURED)
    for lbl, col, mk, face, edge in series:
        ax.scatter(d["kappa_exp"], d[col], marker=mk, s=34, facecolors=face,
                   edgecolors=edge, color=edge if mk == "x" else None, lw=1.2,
                   label=f"{lbl}   ({err(col):.3f})")
    lim = (0.1, 6000)
    ax.plot(lim, lim, color=ds.INK, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured $\\kappa_L$ (W m$^{-1}$K$^{-1}$)")
    ax.set_ylabel("predicted $\\kappa_L$")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left",
              title=f"n = {len(d)}   (mean |$\\Delta\\log_{{10}}$|)", title_fontsize=8.5)
    ds.tidy(ax, grid="both")
    fig.tight_layout()
    ds.save(fig, out("deck_measured.png"))

    rows = {k: err(c) for k, c in [("cgcnn_derived", "kappa_cgcnn_derived_gamma"),
                                   ("cgcnn_paper_gamma", "kappa_cgcnn_equal_footing"),
                                   ("alignn_derived", "kappa_alignn_derived_gamma"),
                                   ("alignn_paper_gamma", "kappa_alignn_equal_footing"),
                                   ("pink_paper", "kappa_pink")]}
    pd.DataFrame([rows]).to_csv(out("deck_measured.csv"), index=False)
    record("MeasN", len(d), MEASURED, "rows")
    # "provenance" says whether the crystal was among the moduli training crystals
    record("MeasInTrain", int((d["provenance"] == "in_matbench_training").sum()), MEASURED,
           "provenance == in_matbench_training")
    for k, name in [("cgcnn_derived", "MeasCgDerived"), ("cgcnn_paper_gamma", "MeasCgPaper"),
                    ("alignn_derived", "MeasAlDerived"), ("alignn_paper_gamma", "MeasAlPaper"),
                    ("pink_paper", "MeasPink")]:
        record(name, f"{rows[k]:.3f}", MEASURED, f"mean |dlog10| {k}")
    print("slide 18:", {k: round(v, 3) for k, v in rows.items()})

    # The example rows for slide 18's table, written as LaTeX table rows.
    # "&" separates the cells and two backslashes end the row, as in any tabular.
    # In Python, "\\" inside a string stands for ONE backslash, so " \\\\\n" below
    # writes a space, two backslashes and a newline.
    with open(out("deck_measured_rows.tex"), "w") as fh:
        for formula, label in MEASURED_EXAMPLES.items():
            r = d[d["formula"] == formula].iloc[0]        # .iloc[0] = the first match
            cells = [label, f"{r['kappa_exp']:.1f}", f"{r['kappa_cgcnn_derived_gamma']:.2f}",
                     f"{r['kappa_cgcnn_equal_footing']:.2f}", f"{r['kappa_pink']:.2f}"]
            fh.write(" & ".join(cells) + " \\\\\n")


# =============================================================================
#  Slide 19 - ALIGNN vs CGCNN kappa_L on PINK's 1,213 crystals
# =============================================================================
def fig_pink_agreement():
    c = pd.read_csv(CGCNN_PINK)[["material_id", "Kappa_cal (W m-1 K-1)"]]
    a = pd.read_csv(ALIGNN_PINK)[["material_id", "Kappa_cal (W m-1 K-1)"]]
    m = c.merge(a, on="material_id", suffixes=("_cgcnn", "_alignn"))
    x = np.log10(m["Kappa_cal (W m-1 K-1)_cgcnn"])
    y = np.log10(m["Kappa_cal (W m-1 K-1)_alignn"])
    r = stats.pearsonr(x, y)[0]          # [0] = the r value ([1] is the p-value)
    rho = stats.spearmanr(x, y)[0]
    mae = float(np.mean(np.abs(x - y)))

    # moduli only: the same comparison before the physics chain
    cm = pd.read_csv(CGCNN_PINK_MOD).merge(pd.read_csv(ALIGNN_PINK_MOD), on="material_id",
                                           suffixes=("_cgcnn", "_alignn"))
    mod = {}
    for t in ("K", "G"):
        u = np.log10(cm[f"{t}_VRH_pred_cgcnn"]); v = np.log10(cm[f"{t}_VRH_pred_alignn"])
        mod[t] = (stats.pearsonr(u, v)[0], stats.spearmanr(u, v)[0], float(np.mean(np.abs(u - v))))

    fig, ax = plt.subplots(figsize=SIZE_PINK)
    ax.scatter(10 ** x, 10 ** y, s=9, alpha=0.45, lw=0, color=ds.MODEL_COLOURS["ALIGNN"])
    lim = (10 ** min(x.min(), y.min()) / 1.5, 10 ** max(x.max(), y.max()) * 1.5)
    ax.plot(lim, lim, color=ds.INK, lw=0.9, ls="--")
    ax.axvline(LOW_KAPPA, color=ds.THRESHOLD, lw=0.8, ls=":")
    ax.axhline(LOW_KAPPA, color=ds.THRESHOLD, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("CGCNN ensemble $\\kappa_L$ (W m$^{-1}$K$^{-1}$)")
    ax.set_ylabel("ALIGNN ensemble $\\kappa_L$")
    ax.text(0.04, 0.96, f"n = {len(m):,}\nPearson r = {r:.3f}\nSpearman $\\rho$ = {rho:.3f}",
            transform=ax.transAxes, va="top", fontsize=10)
    ds.tidy(ax, grid="both")
    fig.tight_layout()
    ds.save(fig, out("deck_pink_agreement.png"))
    pd.DataFrame([{"n": len(m), "kappa_r_log": r, "kappa_rho": rho, "kappa_mae_log": mae,
                   **{f"{t}_{k}": mod[t][i] for t in mod for i, k in
                      enumerate(("r_log", "rho", "mae_log"))}}]).to_csv(
        out("deck_pink_agreement.csv"), index=False)
    record("PinkKappaR", f"{r:.3f}", "deck_pink_agreement.csv", "Pearson r of log10 kappa")
    record("PinkKappaRho", f"{rho:.3f}", "deck_pink_agreement.csv", "Spearman rho")
    record("PinkKappaMae", f"{mae:.3f}", "deck_pink_agreement.csv", "mean |dlog10 kappa|")
    for t in ("K", "G"):
        record(f"PinkMod{t}R", f"{mod[t][0]:.3f}", "deck_pink_agreement.csv",
               f"Pearson r of log10 {t}, ALIGNN vs CGCNN")
        record(f"PinkMod{t}Mae", f"{mod[t][2]:.3f}", "deck_pink_agreement.csv",
               f"mean |dlog10 {t}|")
    print(f"slide 19: n {len(m)} r {r:.3f} rho {rho:.3f} mae {mae:.3f};",
          {t: tuple(round(z, 3) for z in mod[t]) for t in mod})


# =============================================================================
#  Slides 21-24 - the GNoME screen
# =============================================================================
def load_screen():
    """Every screened crystal with both models' kappa_L on one row.
    Keeps only crystals where ALIGNN's prediction is reliable (not at its
    ~1e-3 GPa floor)."""
    a = pd.read_csv(SCREEN_ALIGNN, dtype={"material_id": str})
    c = pd.read_csv(SCREEN_CGCNN, dtype={"material_id": str})
    d = a.merge(c, on="material_id", suffixes=("", "_c"))
    n_all = len(d)
    d = d[d["alignn_prediction_reliable"] == True].copy()
    for col in SCREEN_MODELS.values():
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d, n_all


def count_chain(d, n_scored):
    """The one count chain the funnel slide quotes."""
    n_gnome = sum(1 for _ in open(GNOME_SUMMARY)) - 1          # data rows (minus header)
    n_filtered = sum(1 for _ in open(SCREEN_FILTERED)) - 1
    record("GnomeN", fmt_int(n_gnome), GNOME_SUMMARY, "rows")
    record("FilterN", fmt_int(n_filtered), SCREEN_FILTERED, "rows passing the paper's filters")
    record("IdDropN", fmt_int(n_filtered - n_scored), SCREEN_FILTERED,
           "rows whose GNoME ID could not be verified")
    record("ScoredN", fmt_int(n_scored), SCREEN_ALIGNN, "rows scored by both models")
    record("FloorN", fmt_int(n_scored - len(d)), SCREEN_ALIGNN, "ALIGNN at its regression floor")
    record("ScreenN", fmt_int(len(d)), SCREEN_ALIGNN, "reliable rows")
    for model, col in SCREEN_MODELS.items():
        low = int((d[col] <= LOW_KAPPA).sum())
        key = "Cg" if model.startswith("CGCNN") else "Al"
        record(f"Low{key}", fmt_int(low), "deck_screen_agreement.csv", f"{model} kappa <= 1")
        record(f"Low{key}Pct", f"{100 * low / len(d):.1f}", "deck_screen_agreement.csv", "%")
    print(f"slide 21: {n_gnome} -> {n_filtered} -> {n_scored} -> {len(d)}")


def overlap_with_pink(d):
    """How many of PINK's own 11,869 published candidates have a formula that is
    also on our kappa_L <= 1 list (formulas compared as pymatgen reduced formulas)."""
    from pymatgen.core import Composition
    pink = pd.read_csv(PINK_LIST)
    rows = []
    for model, col in SCREEN_MODELS.items():
        ours = {Composition(f).reduced_formula for f in d.loc[d[col] <= LOW_KAPPA, "formula"]}
        hit = int(pink["Reduced Formula"].isin(ours).sum())
        rows.append({"model": model, "our_low_kappa": int((d[col] <= LOW_KAPPA).sum()),
                     "pink_list": len(pink), "pink_rows_also_ours": hit,
                     "pct_of_pink": 100 * hit / len(pink)})
        key = "Cg" if model.startswith("CGCNN") else "Al"
        record(f"Overlap{key}", fmt_int(hit), "deck_overlap.csv", "PINK rows also on our list")
        record(f"Overlap{key}Pct", f"{100 * hit / len(pink):.1f}", "deck_overlap.csv", "% of 11,869")
    record("PinkListN", fmt_int(len(pink)), PINK_LIST, "rows")
    pd.DataFrame(rows).to_csv(out("deck_overlap.csv"), index=False)
    print("slide 21 overlap:", pd.DataFrame(rows).round(1).to_string(index=False))


def fig_screen_agreement(d):
    x = np.log10(d[SCREEN_MODELS["CGCNN ensemble"]])
    y = np.log10(d[SCREEN_MODELS["ALIGNN"]])
    rho = stats.spearmanr(x, y)[0]
    lo_c, lo_a = x <= 0, y <= 0                     # log10(1) = 0
    quad = {"both low": int((lo_c & lo_a).sum()), "CGCNN only": int((lo_c & ~lo_a).sum()),
            "ALIGNN only": int((~lo_c & lo_a).sum()), "neither": int((~lo_c & ~lo_a).sum())}

    fig, ax = plt.subplots(figsize=SIZE_SCREEN)
    hb = ax.hexbin(x, y, gridsize=70, bins="log", cmap="Greens", mincnt=1)
    lim = (min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1)
    ax.plot(lim, lim, color=ds.INK, lw=0.9, ls="--")
    ax.axvline(0, color=ds.THRESHOLD, lw=0.9, ls=":")
    ax.axhline(0, color=ds.THRESHOLD, lw=0.9, ls=":")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("CGCNN ensemble  $\\log_{10}\\kappa_L$")
    ax.set_ylabel("ALIGNN ensemble  $\\log_{10}\\kappa_L$")
    # One count in each quadrant, written in DATA coordinates so each label sits
    # in the quadrant it describes (x <= 0 means CGCNN calls it low, y <= 0 ALIGNN).
    # Each entry: (x, y, horizontal alignment, vertical alignment, text)
    box = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    labels = [
        (lim[0] + 0.15, -0.15, "left", "top", f"both $\\leq$1\n{quad['both low']:,}"),
        (lim[0] + 0.15, lim[1] - 0.15, "left", "top", f"CGCNN only $\\leq$1\n{quad['CGCNN only']:,}"),
        (0.15, lim[0] + 0.15, "left", "bottom", f"ALIGNN only\n$\\leq$1\n{quad['ALIGNN only']:,}"),
        (0.15, lim[1] - 0.15, "left", "top", f"neither\n{quad['neither']:,}"),
    ]
    for tx, ty, ha, va, text in labels:
        ax.text(tx, ty, text, ha=ha, va=va, fontsize=9, color=ds.INK, bbox=box)
    ax.set_title(f"n = {len(d):,}    Spearman $\\rho$ = {rho:.3f}", loc="left")
    fig.colorbar(hb, ax=ax, shrink=0.8, label="crystals per hexagon")
    ds.tidy(ax, grid=None)
    fig.tight_layout()
    ds.save(fig, out("deck_screen_agreement.png"))
    pd.DataFrame([{"n": len(d), "spearman_rho": rho, **quad}]).to_csv(
        out("deck_screen_agreement.csv"), index=False)
    record("ScreenRho", f"{rho:.3f}", "deck_screen_agreement.csv", "Spearman rho, log10 kappa")
    for k, v in quad.items():
        record("Quad" + re.sub(r"[^A-Za-z]", "", k.title()), fmt_int(v),
               "deck_screen_agreement.csv", k)
    print("slide 23:", round(rho, 3), quad)


def fig_candidates(d):
    sg = pd.read_csv(GNOME_SUMMARY, usecols=["MaterialId", "Crystal System"])
    sg = sg.dropna(subset=["MaterialId"]).drop_duplicates("MaterialId")
    m = d.merge(sg, left_on="material_id", right_on="MaterialId", how="left")
    m["system"] = m["Crystal System"].str.lower()

    fig, axes = plt.subplots(1, 2, figsize=SIZE_CANDID, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    bins = np.linspace(-3.5, 2.5, 80)
    for model, col in SCREEN_MODELS.items():
        v = np.log10(m[col])
        ax.hist(v, bins=bins, histtype="step", lw=1.8, color=ds.MODEL_COLOURS[model],
                label=f"{model}: {int((m[col] <= LOW_KAPPA).sum()):,} at $\\leq$1")
    ax.axvline(0, color=ds.THRESHOLD, lw=1.0, ls=":")
    ax.set_xlabel("$\\log_{10}\\kappa_L$ (W m$^{-1}$K$^{-1}$)")
    ax.set_ylabel("crystals")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)      # headroom so the legend clears the bars
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ds.panel_title(ax, "A", f"$\\kappa_L$ of all {len(m):,}")
    ds.tidy(ax)

    ax = axes[1]
    rows = []
    for s in SYSTEMS:
        sel = m["system"] == s
        row = {"crystal_system": s, "n": int(sel.sum())}
        for model, col in SCREEN_MODELS.items():
            row[f"pct_low_{model}"] = 100 * float((m.loc[sel, col] <= LOW_KAPPA).mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    xs = np.arange(len(SYSTEMS))
    for k, model in enumerate(SCREEN_MODELS):
        ax.bar(xs + (k - 0.5) * 0.38, t[f"pct_low_{model}"], width=0.38,
               color=ds.MODEL_COLOURS[model], label=model)
    # crystal count written above each pair of bars, so the tick labels stay short
    for xi, (top, n) in enumerate(zip(t[[f"pct_low_{mdl}" for mdl in SCREEN_MODELS]].max(axis=1),
                                      t["n"])):
        ax.text(xi, top + 1.5, f"n = {n:,}", ha="center", fontsize=8, color=ds.MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(t["crystal_system"], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, t[[f"pct_low_{mdl}" for mdl in SCREEN_MODELS]].values.max() * 1.18)
    ax.set_ylabel("% called $\\kappa_L \\leq 1$")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ds.panel_title(ax, "B", "share called low, by crystal system")
    ds.tidy(ax)
    fig.tight_layout()
    ds.save(fig, out("deck_candidates.png"))
    t.to_csv(out("deck_candidates.csv"), index=False)
    record("SysUnmatched", int(m["system"].isna().sum()), GNOME_SUMMARY,
           "screened crystals with no crystal system in GNoME's table")
    print("slide 22:\n", t.round(1).to_string(index=False),
          "\n unmatched:", int(m["system"].isna().sum()))


def elements_table(d):
    """Low-kappa enrichment of each element: (share of crystals containing the
    element that are called low) / (share of ALL crystals called low).
    1.0 = no preference, 2.0 = twice as likely, 0.5 = half as likely."""
    from pymatgen.core import Composition
    base = {mdl: float((d[col] <= LOW_KAPPA).mean()) for mdl, col in SCREEN_MODELS.items()}
    freq = collections.Counter()
    low = {mdl: collections.Counter() for mdl in SCREEN_MODELS}
    for _, row in d.iterrows():
        for el in {e.symbol for e in Composition(str(row["formula"])).elements}:
            freq[el] += 1
            for mdl, col in SCREEN_MODELS.items():
                if row[col] <= LOW_KAPPA:
                    low[mdl][el] += 1
    rows = []
    for el, n in freq.items():
        if n < MIN_ELEMENT_COUNT:
            continue
        r = {"element": el, "n_candidates": n}
        for mdl in SCREEN_MODELS:
            r[f"enrich_{mdl}"] = (low[mdl][el] / n) / base[mdl]
        r["enrich_mean"] = np.mean([r[f"enrich_{mdl}"] for mdl in SCREEN_MODELS])
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("enrich_mean", ascending=False)
    t.to_csv(out("deck_elements.csv"), index=False)

    half = N_ELEMENTS_TABLE // 2
    # WHICH elements are shown is decided by the mean of the two models (t is
    # sorted by it). Within each half the rows are then ordered by the CGCNN
    # column, the first number column on the slide, so the column reads in order.
    by_cgcnn = dict(by="enrich_CGCNN ensemble", ascending=False)   # ascending=False = largest first
    pick = pd.concat([t.head(half).sort_values(**by_cgcnn), t.tail(half).sort_values(**by_cgcnn)])
    lines = []
    for i, (_, r) in enumerate(pick.iterrows()):
        if i == half:
            lines.append("\\midrule")
        lines.append(f"{r['element']} & {fmt_int(r['n_candidates'])} & "
                     f"{r['enrich_CGCNN ensemble']:.2f} & {r['enrich_ALIGNN']:.2f} \\\\")
    with open(out("deck_elements_rows.tex"), "w") as fh:   # "w" = write (overwrite)
        fh.write("% Auto-generated by make_deck_figures.py - do not edit by hand.\n")
        fh.write("\n".join(lines) + "\n")
    record("ElemMinN", MIN_ELEMENT_COUNT, "make_deck_figures.py CONFIG", "min candidates")
    record("ElemCount", len(t), "deck_elements.csv", "elements meeting the minimum")
    for el in ("I", "Br", "Cl", "F", "O"):
        if el in set(t["element"]):
            r = t[t["element"] == el].iloc[0]
            record(f"Enr{el}Cg", f"{r['enrich_CGCNN ensemble']:.2f}", "deck_elements.csv", el)
            record(f"Enr{el}Al", f"{r['enrich_ALIGNN']:.2f}", "deck_elements.csv", el)
    print("slide 24:\n", pick.round(2).to_string(index=False))


# =============================================================================
#  Write every recorded number as a LaTeX macro and as a CSV
# =============================================================================
def write_numbers():
    table = pd.DataFrame(NUMBERS)
    table.to_csv(out("deck_numbers.csv"), index=False)
    with open(out("deck_numbers.tex"), "w") as fh:
        fh.write("% Auto-generated by make_deck_figures.py - do not edit by hand.\n")
        fh.write("% Every number here is traced in deck_numbers.csv (name, value, source, how).\n")
        for r in NUMBERS:
            fh.write(f"\\def\\D{r['name']}{{{r['value']}}}\n")
    print(f"wrote {len(NUMBERS)} numbers to figures/deck_numbers.tex + .csv")


if __name__ == "__main__":       # true only when this file is run, not imported
    fig_training_data()
    pred = test_predictions()
    fig_parity(pred)
    fig_residuals()
    fig_ensembling(pred)
    fig_calibration()
    fig_measured()
    fig_pink_agreement()
    screen, n_scored = load_screen()
    count_chain(screen, n_scored)
    overlap_with_pink(screen)
    fig_screen_agreement(screen)
    fig_candidates(screen)
    elements_table(screen)
    write_numbers()
