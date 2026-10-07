#!/usr/bin/env python3
"""
Figures and tables for the AUDIT and DFT-LIST slides of the ALIGNN/PINK deck.

  THE AUDIT (step 95: every metric of every model, re-checked)
    figures/audit_status.png          one bar per kind of row in the step 95
                                      table, split by what the re-check found
    figures/audit_moduli_table.tex    headline accuracy on the matbench test set
    figures/audit_phonix_table.tex    the scores against phonon-DFT kappa_L
                                      (PhoNIX). The scores against MEASURED
                                      kappa_L are on the slide as macros from
                                      make_deck_figures.py instead.

  THE DFT LIST (the 15 handed to the supervisor)
    figures/shortlist_kappa.png       every crystal's kappa_L from the three
                                      independent prediction chains, plus the
                                      4 high-kappa controls
    figures/shortlist_table.tex       the table of 15 for the slide

WHY the tables are written by a script instead of typed into the .tex:
the whole point of step 95 was that hand-typed numbers drift from their
source. Every number on these slides is read from
results/cgcnn/95_final_all_metrics.csv or dft/final_shortlist_15/ here,
and the .tex file only \\input{}s the result.

Run with:  OMP_NUM_THREADS=1 ~/miniconda3/envs/ml_env/bin/python make_audit_figures.py
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # "Agg" = draw to a file, never open a window
import matplotlib.pyplot as plt

# --- where things are -------------------------------------------------------
# __file__ is the path of THIS script; dirname() strips the last part, so
# HERE = presentation/ and ROOT = the project folder above it.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figures")
METRICS = os.path.join(ROOT, "results", "cgcnn", "95_final_all_metrics.csv")
SHORT = os.path.join(ROOT, "dft", "final_shortlist_15")
# Step 82's per-crystal file: every moduli model's K, G and kappa on the SAME
# 1,648 test crystals. The descriptor XGBoost's row on slide 26 comes from here.
PRED82 = os.path.join(ROOT, "results", "cgcnn", "82_test_predictions.csv")
# The sibling repo where the MLIP phonon runs live (slide 27's funnel counts).
# expanduser() turns "~" into the home folder, e.g. /Users/mac.
DATAGEN = os.path.expanduser("~/Desktop/data_generation/results")

# =============================================================================
#  CONFIG - colours. Model colours come from deck_style.py, so a model has the
#  same colour here as on every other slide. Only colours that are NOT models
#  (audit statuses, prediction chains) are set here.
# =============================================================================
import sys
sys.path.insert(0, HERE)           # let Python find deck_style.py next to this file
import deck_style as ds            # "import X as Y" = load file X, call it Y here

INK, MUTED, GRID, RED = ds.INK, ds.MUTED, ds.GRID, ds.THRESHOLD
SUBTLE = "#AAB6C8"                 # pale grey-blue, for "nothing to see" segments

# Figure sizes in inches, as (width, height). The slide scales each figure to
# a fixed width, so a SMALLER figure here prints its labels LARGER there.
SIZE_STATUS    = (10.0, 2.7)      # audit_status.png, slide 25
SIZE_SHORTLIST = (8.0, 5.6)       # shortlist_kappa.png, slide 29

# Slide 28's table of the 15. True = a "supercell" column in the table;
# False = no column, and one generated sentence (\AudSupercellNote) in the
# footnote instead, which frees width for a larger font.
SHORTLIST_SUPERCELL_COLUMN = False

# What the step 95 re-check found, one colour per outcome
STATUS_COLOURS = {
    "match":       "#1E8E5A",      # green  - recomputed value = stored value
    "close":       "#6FA8FF",      # light blue
    "new":         "#1450AA",      # blue   - recomputed, never stored before
    "stored only": SUBTLE,         # grey   - inputs gone, cannot recompute
    "MISMATCH":    RED,
}

# The DFT-list figure shows three prediction CHAINS, not single models.
# Each chain is "the physics chain + some gamma, highest of 3 models", except the
# direct one, which is a single model and so takes its deck_style colour.
CHAIN_COLOURS = {
    "MLIP gamma":    "#0F1B33",                          # navy (theme.tex Navy)
    "Poisson gamma": "#5B6B82",                          # slate - the PINK paper's own recipe
    "direct":        ds.MODEL_COLOURS["direct ALIGNN"],  # purple, as on every slide
}


def style_axes(ax):
    """Light, quiet axes: no top/right frame, pale grid behind the data."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# =============================================================================
#  Reading ONE number out of the step 95 table
# =============================================================================
# low_memory=False makes pandas read each column in one go, so a column that
# mixes text and numbers is not guessed differently in different chunks.
ALL = pd.read_csv(METRICS, low_memory=False)

# Only "primary" metric rows: when the same number was found in several
# files, primary=True marks the one that wins (recomputed beats stored).
# (ALL.primary == True) is a column of True/False, one per row; putting it
# inside ALL[...] keeps only the True rows. "&" means "and" between columns.
PRIMARY = ALL[(ALL.primary == True) & (ALL.row_type == "metric")]


def one(metric, split="test", **match):
    """Return the single value of `metric` in the row(s) matching `match`.

    `**match` collects keyword arguments into a dict, so
        one("R2_log10", family="CGCNN moduli", target="K_VRH")
    filters on family == "CGCNN moduli" AND target == "K_VRH".
    It STOPS the script if zero or several rows match - a slide must never
    silently show the wrong row's number.
    """
    rows = PRIMARY[(PRIMARY.metric == metric) & (PRIMARY.split == split)]
    for column, wanted in match.items():          # .items() = (key, value) pairs
        rows = rows[rows[column] == wanted]
    if len(rows) != 1:
        raise SystemExit(f"expected 1 row for {metric} {match}, found {len(rows)}")
    # .iloc[0] = "the first row by position"; float() turns it into a number
    return float(rows.value.iloc[0])


# =============================================================================
#  1. AUDIT STATUS - what the re-check found, as two stacked bars
# =============================================================================
def figure_audit_status(path):
    metric_rows = PRIMARY
    check_rows = ALL[ALL.row_type.isin(["check", "cross-file check"]) & (ALL.primary == True)]

    # value_counts() counts how many rows carry each status word.
    m = metric_rows.status.value_counts()
    c = check_rows.status.value_counts()

    # The order and colour of each status in the bar. A list of tuples:
    # (status word in the CSV, label on the slide, colour).
    order = [("match", "recomputed = stored", STATUS_COLOURS["match"]),
             ("close", "close (within 0.1%)", STATUS_COLOURS["close"]),
             ("new", "recomputed, never stored before", STATUS_COLOURS["new"]),
             ("stored only", "stored only (inputs gone)", STATUS_COLOURS["stored only"]),
             ("MISMATCH", "MISMATCH", STATUS_COLOURS["MISMATCH"])]

    fig, ax = plt.subplots(figsize=SIZE_STATUS)
    n_rebuild = int((check_rows.row_type == "check").sum())
    n_cross = int((check_rows.row_type == "cross-file check").sum())
    bars = [(f"metric values\n({len(metric_rows):,})", m, len(metric_rows)),
            (f"checks ({len(check_rows):,})\n{n_rebuild} rebuild + {n_cross} cross-file",
             c, len(check_rows))]
    for y, (name, counts, total) in enumerate(bars):
        left = 0.0                              # where the next segment starts
        for status, label, colour in order:
            # .get(key, 0) = the count, or 0 if that status never occurs
            n = int(counts.get(status, 0))
            if n == 0:
                continue
            share = 100.0 * n / total
            ax.barh(y, share, left=left, color=colour, edgecolor="white", height=0.62,
                    label=label if y == 0 else None)
            if share > 4:                       # only label segments wide enough
                text_colour = "white" if colour not in (SUBTLE, STATUS_COLOURS["close"]) else INK
                ax.text(left + share / 2, y, f"{n:,}", ha="center", va="center",
                        color=text_colour, fontsize=10, fontweight="bold")
            elif n > 0:
                # too thin to write inside: label it above the bar, with an arrow
                ax.annotate(f"{n} {status}", (left + share / 2, y - 0.31), xytext=(0, 12),
                            textcoords="offset points", ha="center", color=colour,
                            fontsize=9, arrowprops=dict(arrowstyle="-", color=colour, lw=0.8))
            left += share
    # The checks bar is all "match" - say so explicitly, and say what is NOT there.
    ax.text(101, 1, "all pass", va="center", color=STATUS_COLOURS["match"], fontsize=10, fontweight="bold")
    n_bad = int(m.get("MISMATCH", 0)) + int(c.get("MISMATCH", 0))
    ax.text(101, 0, f"{n_bad} MISMATCH", va="center", color=STATUS_COLOURS["match"] if n_bad == 0 else RED,
            fontsize=10, fontweight="bold")

    ax.set_yticks([0, 1])
    ax.set_yticklabels([b[0] for b in bars], fontsize=10, color=INK)
    ax.invert_yaxis()                           # first bar on top
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of rows (%)", color=MUTED, fontsize=9)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=4, frameon=False,
              fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return m, c


# =============================================================================
#  2. HEADLINE TABLES - typed by the script, \input by the slide
# =============================================================================
def f4(x):
    """Format with 4 decimals; '---' for a missing value (x is None)."""
    return "---" if x is None else f"{x:.4f}"


def f3(x):
    return "---" if x is None else f"{x:.3f}"


def scores_from_step82(tag):
    """log10 MAE and R^2 of K and G, and the kappa MAE, for one model of step 82.

    `tag` is the model's name inside 82_test_predictions.csv: its columns are
    K_<tag>, G_<tag> (predicted GPa) and <tag>_full_chain (predicted kappa, the
    model's own Poisson gamma). The truth is K_VRH, G_VRH and kappa_DFT (the same
    chain on the DFT moduli) - the same comparison step 95 makes for the networks.
    Returns a dict, e.g. {"kmae": 0.0591, "kr2": 0.918, ...}.
    """
    d = pd.read_csv(PRED82)

    def mae_r2(true, pred):
        t, p = np.log10(true), np.log10(pred)        # work in log10, like every table here
        mae = float(np.mean(np.abs(t - p)))
        # R^2 = 1 - (squared error left over) / (squared spread of the truth)
        r2 = float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
        return mae, r2

    kmae, kr2 = mae_r2(d.K_VRH, d[f"K_{tag}"])
    gmae, gr2 = mae_r2(d.G_VRH, d[f"G_{tag}"])
    kappa = float(np.mean(np.abs(np.log10(d[f"{tag}_full_chain"]) - np.log10(d.kappa_DFT))))
    return dict(n=len(d), kmae=kmae, kr2=kr2, gmae=gmae, gr2=gr2, kappa=kappa)


def descriptor_xgboost_scores():
    """The 3rd moduli model (XGBoost on 297 matminer descriptors) on slide 26.

    Step 95 never stored its log10 MAE / R^2, so they are computed here from
    step 82's file. SELF-CHECK first: the same function, run on ALIGNN and the
    CGCNN ensemble, must give back the numbers step 95 stores for them. If it
    does, the XGBoost numbers are computed exactly like the rest of the table.
    """
    checks = [("ALIGNN", dict(family="ALIGNN moduli", model="ALIGNN 3-ensemble"),
               dict(family="ALIGNN separate -> kappa", model="ALIGNN 3-ensemble, K and G")),
              ("CGCNN-ens", dict(family="CGCNN moduli", model="CGCNN 3-ensemble"),
               dict(family="CGCNN separate -> kappa", model="CGCNN separate, 3-ensemble each"))]
    for tag, mod, kap in checks:
        mine = scores_from_step82(tag)
        stored = dict(kmae=one("MAE_log10", target="K_VRH", **mod), kr2=one("R2_log10", target="K_VRH", **mod),
                      gmae=one("MAE_log10", target="G_VRH", **mod), gr2=one("R2_log10", target="G_VRH", **mod),
                      kappa=one("kappa_mae_total", target="K+G -> kappa", **kap))
        for key in stored:                       # looping over a dict gives its keys
            if abs(mine[key] - stored[key]) > 1e-4:
                raise SystemExit(f"step 82 vs step 95 disagree for {tag} {key}: "
                                 f"{mine[key]:.5f} vs {stored[key]:.5f}")
    print("  self-check: step 82's file reproduces step 95's ALIGNN and CGCNN rows (to 1e-4)")
    return scores_from_step82("newbase")


def table_moduli(path):
    """Every model family on the SAME 1,648 matbench test crystals."""
    # Each entry: (label on the slide, dict of filters for K, for G, for kappa).
    # A filter dict of None means "this model has no such number" -> '---'.
    rows = [
        ("Random forest (composition only)",
         dict(family="tree baseline", model="random forest", run="43 matbench"), None),
        ("XGBoost (composition only)",
         dict(family="tree baseline", model="xgboost", run="43 matbench"), None),
        ("CGCNN, 1 model (seed 42)",
         dict(family="CGCNN moduli", model="CGCNN single (seed 42)"),
         dict(family="CGCNN separate -> kappa", model="CGCNN separate, 1 model each")),
        ("CGCNN, 3-ensemble",
         dict(family="CGCNN moduli", model="CGCNN 3-ensemble"),
         dict(family="CGCNN separate -> kappa", model="CGCNN separate, 3-ensemble each")),
        ("ALIGNN, 1 model (seed 42)",
         dict(family="ALIGNN moduli", model="ALIGNN seed 42"), None),
        ("ALIGNN, 3-ensemble",
         dict(family="ALIGNN moduli", model="ALIGNN 3-ensemble"),
         dict(family="ALIGNN separate -> kappa", model="ALIGNN 3-ensemble, K and G")),
    ]
    desc = descriptor_xgboost_scores()
    lines = []
    for label, mod, kap in rows:             # each tuple is unpacked into 3 names
        kmae = one("MAE_log10", target="K_VRH", **mod)
        gmae = one("MAE_log10", target="G_VRH", **mod)
        kr2 = one("R2_log10", target="K_VRH", **mod)
        gr2 = one("R2_log10", target="G_VRH", **mod)
        kappa = None if kap is None else one("kappa_mae_total", target="K+G -> kappa", **kap)
        lines.append(f"{label} & {f4(kmae)} & {f3(kr2)} & {f4(gmae)} & {f3(gr2)} & {f4(kappa)} \\\\")
        print(f"  {label:34s} K {kmae:.4f} R2 {kr2:.3f}  G {gmae:.4f} R2 {gr2:.3f}  kappa "
              f"{'---' if kappa is None else f'{kappa:.4f}'}")
        if label == "XGBoost (composition only)":
            # the descriptor model goes right after the two composition-only trees
            lines.append(f"XGBoost (297 descriptors)$^{{*}}$ & {f4(desc['kmae'])} & {f3(desc['kr2'])} & "
                         f"{f4(desc['gmae'])} & {f3(desc['gr2'])} & {f4(desc['kappa'])} \\\\")
            print(f"  {'XGBoost (297 descriptors)':34s} K {desc['kmae']:.4f} R2 {desc['kr2']:.3f}  "
                  f"G {desc['gmae']:.4f} R2 {desc['gr2']:.3f}  kappa {desc['kappa']:.4f}  (n {desc['n']})")
    with open(path, "w") as fh:                 # "w" = write (replace the file)
        fh.write("% written by make_audit_figures.py from 95_final_all_metrics.csv - do not edit\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def table_phonix(path):
    """Scores against phonon-DFT kappa_L (PhoNIX) - crystals no model trained on.

    Two blocks:
      1. the screen's own chain (physics chain + Poisson gamma, highest of 3 models),
         scored on the 2,520 PhoNIX crystals that are NOT in matbench and have
         <= 20 atoms (step 87). Precision = of the crystals called low, the
         share DFT also finds low. Read it against the base rate.
      2. a direct structure -> kappa_L ALIGNN, 5-fold cross-validated on PhoNIX.
    """
    p87 = dict(family="87 PhoNIX Poisson chain", run="truth klat, low = kappa <= 1.0")
    # split="out-of-fold": every crystal predicted once, by the fold that did not train on it
    dk = dict(family="direct kappa", model="direct ALIGNN (PhoNIX, 5-fold OOF)", split="out-of-fold")

    prec = one("precision", model="kappa_max3", **p87)
    lo, hi = one("precision_lo95", model="kappa_max3", **p87), one("precision_hi95", model="kappa_max3", **p87)
    recall = one("recall", model="kappa_max3", **p87)
    base = one("base_rate", model="kappa_max3", **p87)
    # each model on its own, for comparison with the max of the three
    singles = [one("precision", model=m, **p87) for m in ("ALIGNN", "CGCNN-ens", "newbase")]
    # the test-set size is stored in the "n" column of the same row
    n87 = int(PRIMARY[(PRIMARY.metric == "precision") & (PRIMARY.model == "kappa_max3")
                      & (PRIMARY.family == p87["family"]) & (PRIMARY.run == p87["run"])].n.iloc[0])
    dk_mae, dk_r2, dk_rho = (one(x, **dk) for x in ("MAE_log10", "R2_log10", "spearman_rho"))
    # how many crystals the 5-fold run predicted = the n column of that same row
    n_dk = int(PRIMARY[(PRIMARY.metric == "MAE_log10") & (PRIMARY.family == dk["family"])
                       & (PRIMARY.model == dk["model"]) & (PRIMARY.split == dk["split"])].n.iloc[0])

    # Step 89: a sample of those crystals also got MLIP phonon gamma. Two rules
    # of EQUAL strictness - the MLIP rule the final 15 pass, and the Poisson cut
    # that keeps as many crystals - so neither row is flattered by a looser cut.
    g89 = dict(family="89 MLIP vs Poisson", model="max of three", target="DFT klat <= 1")
    mlip_run = "as safe as the final 15, stress <= 0.85 [added]"
    pois_run = "Poisson, cutoff 0.46 = as strict as 'as safe as the final 15' [added]"
    mlip = [one(x, run=mlip_run, **g89) for x in ("precision", "precision_lo95", "precision_hi95")]
    pois = [one(x, run=pois_run, **g89) for x in ("precision", "precision_lo95", "precision_hi95")]
    n89 = int(PRIMARY[(PRIMARY.metric == "precision") & (PRIMARY.family == g89["family"])
                      & (PRIMARY.run == mlip_run) & (PRIMARY.target == g89["target"])].n.iloc[0])

    # Each line: test set & model & what was measured & the number.
    # TinyTeX here has no multirow package, so a block's label is simply
    # split over its rows (line 1 of the label on row 1, line 2 on row 2).
    k = "$\\kappa_L$"
    lines = [
        f"PhoNIX, not in matbench & Poisson $\\gamma$, highest of 3 & precision at {k}$\\le1$ & "
        f"{f3(prec)} [{f3(lo)}, {f3(hi)}] \\\\",
        f"{n87:,} crystals, $\\le$20 atoms & each model alone & precision at {k}$\\le1$ & "
        f"{min(singles):.3f}--{max(singles):.3f} \\\\",
        f" & highest of 3 & recall (true lows found) & {f3(recall)} \\\\",
        f" & a blind pick & base rate (truly $\\le1$) & {f3(base)} \\\\",
        "\\midrule",
        f"{n89} of them, sampled, & MLIP $\\gamma$, as safe as the 15 & precision at {k}$\\le1$ & "
        f"{f3(mlip[0])} [{f3(mlip[1])}, {f3(mlip[2])}] \\\\",
        f"with MLIP $\\gamma$ (step 89) & Poisson $\\gamma$, equally strict & precision at {k}$\\le1$ & "
        f"{f3(pois[0])} [{f3(pois[1])}, {f3(pois[2])}] \\\\",
        "\\midrule",
        f"PhoNIX, 5-fold & direct ALIGNN, structure$\\to${k} & MAE log$_{{10}}$ & {f3(dk_mae)} \\\\",
        f"{n_dk:,} of the 6,641, held out once & & $R^2$ / Spearman $\\rho$ & {f3(dk_r2)} / {f3(dk_rho)} \\\\",
    ]
    with open(path, "w") as fh:
        fh.write("% written by make_audit_figures.py from 95_final_all_metrics.csv - do not edit\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(f"  PhoNIX n {n87}: precision {prec:.3f} [{lo:.3f}, {hi:.3f}], singles {singles}, "
          f"recall {recall:.3f}, base {base:.3f}; direct MAE {dk_mae:.3f} R2 {dk_r2:.3f} rho {dk_rho:.3f}")
    print(f"  step 89 (n {n89}): MLIP as safe as the 15 {mlip}, Poisson equally strict {pois}")


# =============================================================================
#  3. THE DFT LIST - read index.csv, plus the README's space-group table
# =============================================================================
def read_shortlist():
    d = pd.read_csv(os.path.join(SHORT, "index.csv"))
    ctrl = pd.read_csv(os.path.join(SHORT, "controls", "index.csv"))

    # The README's markdown table has space group / supercell / spin, which
    # index.csv does not. A regex picks out lines like
    #   | 1  | Cs4NbCrI12   | d7e7d897af | 18 | P4/mmm (123) | 2x2x1 (72) | yes, Cr(III) |
    # \s* = any spaces, (\d+) = a number we keep, ([^|]+) = text up to the next |
    readme = open(os.path.join(SHORT, "README.md")).read()
    pat = re.compile(r"^\|\s*(\d+)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|", re.M)
    extra = {}
    for g in pat.findall(readme):               # findall -> one tuple per line
        rank, formula, mid, atoms, sg, cell, spin = (x.strip() for x in g)
        extra[int(rank)] = dict(formula=formula, mid=mid, sg=sg, cell=cell, spin=spin)

    # Cross-check: README and index.csv must describe the SAME 15, in the same order.
    if sorted(extra) != list(range(1, 16)):
        raise SystemExit(f"README table has ranks {sorted(extra)}")
    for _, r in d.iterrows():                   # iterrows() = one row at a time
        e = extra[int(r["rank"])]
        if e["formula"] != r.formula or e["mid"] != r.material_id[:10]:
            raise SystemExit(f"README row {r['rank']} {e} != index.csv {r.formula} {r.material_id}")
    print(f"README and index.csv agree on all {len(d)} crystals")
    return d, ctrl, extra


def latex_formula(s):
    """Cs4NbCrI12 -> Cs$_{4}$NbCrI$_{12}$ (every run of digits becomes a subscript)."""
    return re.sub(r"(\d+)", r"$_{\1}$", s)


def latex_sg(s):
    """P-1 (2) -> P$\\bar{1}$ (2): crystallographers write -1 as 1 with a bar."""
    return re.sub(r"-(\d)", r"$\\bar{\1}$", s)


def table_shortlist(d, extra, path):
    lines = []
    for _, r in d.iterrows():
        e = extra[int(r["rank"])]
        # 2x2x1 -> 2$\times$2$\times$1. Done before the f-string because this
        # Python (3.11) forbids a backslash inside an f-string's { }.
        cell = e["cell"].replace("x", "$\\times$")
        spin = e["spin"].replace("I2+", "I$_2^+$")     # the [I2]+ cation
        # ":.2f" = 2 decimals, which is all a x1.6-error model deserves
        # the supercell cell (with its "&") only when the CONFIG switch asks for it
        cell_col = f"{cell} & " if SHORTLIST_SUPERCELL_COLUMN else ""
        lines.append(
            f"{int(r['rank'])} & {latex_formula(r.formula)} & {int(r.n_atoms)} & "
            f"{latex_sg(e['sg'])} & {cell_col}{spin} & "
            f"{r.kappa_max3:.2f} & {r.kappa_stress:.2f} & {r.kappa_poisson_max3:.2f} & "
            f"{r.kappa_direct:.2f} \\\\")
    with open(path, "w") as fh:
        fh.write("% written by make_audit_figures.py from dft/final_shortlist_15/ - do not edit\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def stress_test_factor():
    """The stress test's gamma factor, asked of step 86 rather than typed here.

    Step 86 compares MLIP gamma with reference gamma on a benchmark and keeps
    the 10th percentile of reference/MLIP (~0.68): "the MLIP overshooting as
    badly as 1 benchmark material in 10". import_module() loads a file whose
    name starts with a digit, which a plain "import" statement cannot do.
    """
    from importlib import import_module
    sys.path.insert(0, os.path.join(ROOT, "scripts", "cgcnn"))
    s86 = import_module("86_low_range_list")
    _typical, stress = s86.gamma_corrections()     # it returns two numbers; keep the 2nd
    return stress


def figure_shortlist(d, ctrl, path):
    """One row per crystal; three chains' kappa on a log axis; kappa = 1 line."""
    # The stress test shrinks every MLIP gamma by the same factor. Step 86
    # computes that factor from its gamma benchmark; ask it, rather than
    # typing the number here. import_module() loads a file whose name starts
    # with a digit, which a plain "import" statement cannot do.
    stress_factor = stress_test_factor()
    cut_pct = round(100 * (1 - stress_factor))     # e.g. factor 0.684 -> 32 (%)
    mlip, poisson, direct = (CHAIN_COLOURS[c] for c in ("MLIP gamma", "Poisson gamma", "direct"))

    # Build one list of rows: the 15, then the 4 controls. For the controls
    # the 'MLIP' value is kappa_pred_slack_300K (= their highest kappa_mlip_*)
    # and there is no stress-test value.
    rows = []
    for _, r in d.iterrows():
        rows.append(dict(name=f"{int(r['rank'])}. {latex_formula(r.formula)}", mlip=r.kappa_max3,
                         stress=r.kappa_stress, poisson=r.kappa_poisson_max3,
                         direct=r.kappa_direct, control=False))
    for _, r in ctrl.iterrows():
        rows.append(dict(name=f"control: {latex_formula(r.formula)}", mlip=r.kappa_pred_slack_300K,
                         stress=np.nan, poisson=r.kappa_poisson_max3,
                         direct=r.kappa_direct, control=True))

    # y positions: 0..14 for the 15, then skip one row, 16..19 for the controls
    y = [i if not row["control"] else i + 1 for i, row in enumerate(rows)]

    fig, ax = plt.subplots(figsize=SIZE_SHORTLIST)
    ax.axvspan(1e-2, 1.0, color=RED, alpha=0.04, lw=0)      # the target zone
    ax.axvline(1.0, color=RED, lw=1.4, ls="--")
    ax.text(1.06, -0.9, "target: $\\kappa_L \\leq 1$ W/m/K", color=RED, fontsize=9, va="center")

    for yi, row in zip(y, rows):                # zip pairs the two lists item by item
        # The MLIP chain's range: best guess -> worst plausible gamma error
        if not np.isnan(row["stress"]):
            ax.plot([row["mlip"], row["stress"]], [yi, yi], color=mlip, lw=2.2, alpha=0.30,
                    solid_capstyle="round")
            ax.plot(row["stress"], yi, "|", color=mlip, ms=11, mew=2)
        ax.plot(row["mlip"], yi, "o", color=mlip, ms=7)
        ax.plot(row["poisson"], yi, "D", color=poisson, ms=6)
        ax.plot(row["direct"], yi, "s", color=direct, ms=6.5)

    # Legend entries drawn once, off-screen, so each symbol appears exactly once
    ax.plot([], [], "o", color=mlip, ms=7, label="physics chain + MLIP phonon $\\gamma$ (max of 3 models)")
    ax.plot([], [], "|", color=mlip, ms=11, mew=2, label=f"... with $\\gamma$ cut by {cut_pct}% (stress test)")
    ax.plot([], [], "D", color=poisson, ms=6, label="physics chain + Poisson $\\gamma$ (max of 3 models)")
    ax.plot([], [], "s", color=direct, ms=6.5, label="direct ALIGNN, trained on 6,641 phonon-DFT $\\kappa_L$")

    ax.set_xscale("log")
    ax.set_xlim(1e-2, 300)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=9, color=INK)
    for lab, row in zip(ax.get_yticklabels(), rows):
        if row["control"]:
            lab.set_color(MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("predicted $\\kappa_L$ at 300 K (W/m/K, log scale)", color=MUTED, fontsize=10)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", edgecolor=GRID)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # Print the facts the slide text will quote, so they come from data too.
    worst = max(max(r["mlip"], r["stress"], r["poisson"], r["direct"]) for r in rows if not r["control"])
    lowest_ctrl = min(min(r["mlip"], r["poisson"], r["direct"]) for r in rows if r["control"])
    n_poisson = sum(r["poisson"] <= 0.54 for r in rows if not r["control"])
    print(f"  highest kappa of ANY chain on the 15: {worst:.3f}")
    print(f"  lowest kappa of ANY chain on the 4 controls: {lowest_ctrl:.1f}")
    print(f"  Poisson chain <= 0.54 cut: {n_poisson}/15")


def gamma_benchmark_size(stress):
    """How many benchmark crystals step 86's stress factor was measured on.

    Same file and same filter as 86_low_range_list.gamma_corrections(); the
    10th percentile is recomputed and must equal the factor it returned, so
    the count below is certainly the count behind that factor.
    """
    from importlib import import_module
    s86 = import_module("86_low_range_list")     # already loaded once; Python reuses it
    b = pd.read_csv(s86.BENCH)
    b = b[b.stable & b.clean].dropna(subset=["gamma_paper", "gamma_mlip"])
    if abs((b.gamma_paper / b.gamma_mlip).quantile(0.10) - stress) > 1e-12:
        raise SystemExit("benchmark filter here differs from step 86's")
    return len(b)


def mlip_stage_counts():
    """Crystals that got an MLIP phonon run, per route (slide 27's funnel).

    Halide route: 01_gamma_phonon_halide_s0*.csv + the pilot file (the recut
    files re-run a subset of these, so they add no new crystals).
    Oxide route: 01_gamma_phonon_gnome_*.csv, minus gnome_selected.csv, which
    is a list of inputs rather than results.
    Returns (halides, oxides, in both) as three counts of unique crystal ids.
    """
    import glob                               # glob = list the files matching a * pattern

    def ids(files):
        # pd.concat stacks several tables into one; set() keeps each id once
        return set(pd.concat([pd.read_csv(f, dtype={"mp_id": str}) for f in files]).mp_id)

    hal_files = (sorted(glob.glob(os.path.join(DATAGEN, "01_gamma_phonon_halide_s0*.csv")))
                 + [os.path.join(DATAGEN, "01_gamma_phonon_pilot_mb3.csv")])
    ox_files = sorted(f for f in glob.glob(os.path.join(DATAGEN, "01_gamma_phonon_gnome_*.csv"))
                      if not f.endswith("selected.csv"))
    hal, ox = ids(hal_files), ids(ox_files)
    # Guard: every halide run must come from step 71's top-300 list ("<=" = subset)
    top = pd.read_csv(os.path.join(ROOT, "results", "cgcnn", "71_top300_matbench3.csv"),
                      dtype={"material_id": str})
    if not hal <= set(top.material_id):
        raise SystemExit("a halide MLIP run is not in step 71's top 300")
    print(f"  MLIP stage: {len(hal)} halide-route + {len(ox)} oxide-route crystals "
          f"({len(hal & ox)} in both; {len(hal_files)} + {len(ox_files)} files)")
    return len(hal), len(ox), len(hal & ox)      # "&" between sets = the ids in both


def supercell_note(extra):
    """The supercells as one short phrase, for slide 28's footnote, e.g.
    '2x2x1; #6, 8, 13: 2x2x2; #14: 3x2x2.' (the slide types the rule
    'smallest, every vector >= 10 A' in front of it)"""
    by_cell = {}                                  # supercell -> list of ranks
    for rank in sorted(extra):
        cell = extra[rank]["cell"].split()[0]     # "2x2x1 (72)" -> "2x2x1"
        by_cell.setdefault(cell, []).append(rank) # setdefault = start an empty list once
    # "lambda c: ..." = a tiny one-line function of c; here 2x2x1 -> 2$\times$2$\times$1
    tex = lambda c: c.replace("x", "$\\times$")
    common = max(by_cell, key=lambda c: len(by_cell[c]))   # the most frequent supercell
    rest = "; ".join(f"\\#{', '.join(str(r) for r in ranks)}: {tex(c)}"
                     for c, ranks in by_cell.items() if c != common)
    return f"{tex(common)}; {rest}."


# =============================================================================
#  4. NUMBERS QUOTED IN THE SLIDE TEXT - written as LaTeX macros
# =============================================================================
# Same idea as figures/deck_numbers.tex (make_deck_figures.py): the slide
# writes \AudChecks instead of typing 473, and audit_numbers.csv says where
# each value came from. Macro names start with "Aud" so they cannot clash
# with theme.tex's \AccentCard or metrics.tex.
CROSSWORK_05 = os.path.join(os.path.dirname(ROOT), "transport_program", "crosswork", "data",
                            "05_dft_list_direct_klat.csv")


def write_audit_numbers(d, ctrl, extra, tex_path, csv_path):
    rows = []                                   # (name, text for the slide, source, how)

    def add(name, text, source, how):
        rows.append((name, text, source, how))

    # --- the audit itself ------------------------------------------------------
    checks = ALL[ALL.row_type.isin(["check", "cross-file check"]) & (ALL.primary == True)]
    status = PRIMARY.status.value_counts()
    src = "results/cgcnn/95_final_all_metrics.csv"
    add("AudRows", f"{len(ALL):,}", src, "all rows")
    add("AudMetricRows", f"{len(PRIMARY):,}", src, "primary == True and row_type == metric")
    for key, name in (("match", "AudMatch"), ("close", "AudClose"), ("new", "AudNew"),
                      ("stored only", "AudStoredOnly")):
        add(name, f"{int(status.get(key, 0)):,}", src, f"primary metric rows, status == {key}")
    add("AudChecks", f"{len(checks):,}", src, "primary check + cross-file check rows")
    add("AudRebuildChecks", f"{int((checks.row_type == 'check').sum()):,}", src, "row_type == check")
    add("AudCrossChecks", f"{int((checks.row_type == 'cross-file check').sum()):,}", src,
        "row_type == cross-file check")
    n_bad = int((ALL[ALL.primary == True].status == "MISMATCH").sum())
    add("AudMismatch", f"{n_bad}", src, "primary rows with status MISMATCH")
    # Rows with primary == False repeat a metric already counted once (the
    # same number stored in a second file). "~" flips True/False in pandas.
    add("AudRepeatRows", f"{int((~ALL.primary.astype(bool)).sum()):,}", src, "primary == False")
    # Step 95 part 1 listed every file it would read; "kind" says what each is.
    inv_path = os.path.join(ROOT, "results", "cgcnn", "95_metrics_inventory.csv")
    kinds = pd.read_csv(inv_path).kind.value_counts()
    add("AudPredFiles", f"{int(kinds['predictions'])}", "results/cgcnn/95_metrics_inventory.csv",
        "kind == predictions")
    add("AudStoredFiles", f"{int(kinds['stored'])}", "results/cgcnn/95_metrics_inventory.csv",
        "kind == stored")

    # --- phonon-DFT precision for picks as safe as the 15 (step 89) -------------
    g = dict(family="89 MLIP vs Poisson", model="max of three", target="DFT klat <= 1",
             run="as safe as the final 15, stress <= 0.85 [added]")
    prec = one("precision", **g)
    lo, hi = one("precision_lo95", **g), one("precision_hi95", **g)
    add("AudSafePrec", f"{prec:.2f}", src, "step 89, " + g["run"] + ", precision")
    add("AudSafePrecLo", f"{lo:.2f}", src, "same row family, precision_lo95")
    add("AudSafePrecHi", f"{hi:.2f}", src, "same row family, precision_hi95")
    n_sampled = int(PRIMARY[(PRIMARY.metric == "precision") & (PRIMARY.family == g["family"])
                            & (PRIMARY.run == g["run"]) & (PRIMARY.target == g["target"])].n.iloc[0])
    add("AudSafeN", f"{n_sampled:,}", src, "n column of the same row")
    # Expected number of the 15 that DFT confirms, if that precision carries over
    add("AudExpect", f"{15 * prec:.0f}", src, "15 x precision")
    add("AudExpectLo", f"{15 * lo:.0f}", src, "15 x precision_lo95")
    add("AudExpectHi", f"{15 * hi:.0f}", src, "15 x precision_hi95")

    # --- the 15, the 4 controls, the reserves ------------------------------------
    s15 = "dft/final_shortlist_15/index.csv"
    stress = stress_test_factor()
    add("AudStressFactor", f"{stress:.3f}", "scripts/cgcnn/86_low_range_list.py",
        "gamma_corrections(): 10th percentile of reference/MLIP gamma")
    add("AudStressPct", f"{round(100 * (1 - stress))}", "scripts/cgcnn/86_low_range_list.py",
        "100 x (1 - factor)")
    add("AudGammaBenchN", f"{gamma_benchmark_size(stress)}",
        "~/Desktop/data_generation/results/02_gamma_comparison.csv",
        "stable & clean rows with gamma_paper and gamma_mlip (step 86's filter)")

    # --- slide 27: how many crystals reached the MLIP phonon stage ---------------
    n_hal, n_ox, n_both = mlip_stage_counts()
    dg = "~/Desktop/data_generation/results/"
    add("AudMlipHalideN", f"{n_hal}", dg + "01_gamma_phonon_halide_s0*.csv + 01_gamma_phonon_pilot_mb3.csv",
        "unique mp_id (all in results/cgcnn/71_top300_matbench3.csv)")
    add("AudMlipOxideN", f"{n_ox}", dg + "01_gamma_phonon_gnome_{lowk_s0*,sel_0*}.csv",
        "unique mp_id")
    add("AudMlipBothN", f"{n_both}", dg + "both sets above", "ids in both")

    # --- slide 26: the descriptor XGBoost row (also in audit_moduli_table.tex) ---
    desc = scores_from_step82("newbase")
    for key, name in (("kmae", "AudDescKMae"), ("kr2", "AudDescKRtwo"), ("gmae", "AudDescGMae"),
                      ("gr2", "AudDescGRtwo"), ("kappa", "AudDescKappaMae")):
        # errors to 4 decimals and R^2 to 3, as in the table itself
        add(name, f"{desc[key]:.4f}" if key in ("kmae", "gmae", "kappa") else f"{desc[key]:.3f}",
            "results/cgcnn/82_test_predictions.csv",
            f"{key} of K_newbase/G_newbase/newbase_full_chain vs K_VRH/G_VRH/kappa_DFT, log10")

    # --- slide 28: the supercell sentence (used when the column is switched off) --
    add("AudSupercellNote", supercell_note(extra), "dft/final_shortlist_15/README.md",
        "supercell column of the README table, grouped")
    worst = d[["kappa_max3", "kappa_stress", "kappa_poisson_max3", "kappa_direct"]].max().max()
    add("AudWorstOfFifteen", f"{worst:.2f}", s15, "highest kappa of any route on the 15")
    ctrl_vals = ctrl[["kappa_pred_slack_300K", "kappa_poisson_max3", "kappa_direct"]]
    add("AudCtrlMin", f"{ctrl_vals.min().min():.1f}", "dft/final_shortlist_15/controls/index.csv",
        "lowest kappa of any route on the 4 controls")
    add("AudCtrlMax", f"{ctrl_vals.max().max():.0f}", "dft/final_shortlist_15/controls/index.csv",
        "highest kappa of any route on the 4 controls")
    add("AudPoissonCut", "0.54", "scripts/cgcnn/92_shortlist_poisson_check.py",
        "P_CUT: the Poisson cut as strict as tier 1 (step 89)")
    add("AudPoissonCutN", f"{int((d.kappa_poisson_max3 <= 0.54).sum())}", s15,
        "kappa_poisson_max3 <= 0.54 (step 92's cut)")
    miss = d[d.kappa_poisson_max3 > 0.54]
    if len(miss) == 1:                          # name the one that misses, if exactly one
        add("AudPoissonMissName", latex_formula(miss.formula.iloc[0]), s15, "the crystal above 0.54")
        add("AudPoissonMissValue", f"{miss.kappa_poisson_max3.iloc[0]:.2f}", s15, "its Poisson max3")
    add("AudDirectHalfN", f"{int((d.kappa_direct <= 0.50).sum())}", s15, "kappa_direct <= 0.50")
    add("AudSoftMode", f"{d.min_freq_THz.min():.3f}", s15, "most negative min_freq_THz")

    res = pd.read_csv(os.path.join(SHORT, "reserves", "index.csv"))
    r35 = res.kappa_direct.iloc[2:5]            # reserves 3, 4, 5 (rows 2-4 counting from 0)
    add("AudReserveDirectLo", f"{r35.min():.1f}", "dft/final_shortlist_15/reserves/index.csv",
        "min kappa_direct of reserves 3-5")
    add("AudReserveDirectHi", f"{r35.max():.1f}", "dft/final_shortlist_15/reserves/index.csv",
        "max kappa_direct of reserves 3-5")

    # the direct model on the 53 earlier candidates (crosswork step 05)
    c5 = pd.read_csv(CROSSWORK_05)
    # .str.startswith("control") is True on the 4 control rows; "~" flips it
    c53 = c5[~c5.decision.str.startswith("control")]
    if len(c53) != 53:
        raise SystemExit(f"crosswork 05: expected 53 non-control rows, found {len(c53)}")
    add("AudDirectLowOfFiftyThree", f"{int(c53.direct_calls_low.sum())}",
        "transport_program/crosswork/data/05_dft_list_direct_klat.csv", "direct_calls_low on the 53")

    with open(tex_path, "w") as fh:
        fh.write("% Auto-generated by make_audit_figures.py - do not edit by hand.\n")
        fh.write("% Every number here is traced in audit_numbers.csv (name, value, source, how).\n")
        for name, text, _src, _how in rows:
            # LaTeX wants 1{,}234: a bare comma in maths mode adds a space
            fh.write(f"\\def\\{name}{{{text.replace(',', '{,}')}}}\n")
    pd.DataFrame(rows, columns=["name", "value", "source", "how"]).to_csv(csv_path, index=False)
    print(f"wrote {len(rows)} numbers to {tex_path}")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    m, c = figure_audit_status(os.path.join(FIGS, "audit_status.png"))
    print("  metric rows by status:", dict(m), "| checks:", dict(c))
    table_moduli(os.path.join(FIGS, "audit_moduli_table.tex"))
    table_phonix(os.path.join(FIGS, "audit_phonix_table.tex"))
    d, ctrl, extra = read_shortlist()
    table_shortlist(d, extra, os.path.join(FIGS, "shortlist_table.tex"))
    figure_shortlist(d, ctrl, os.path.join(FIGS, "shortlist_kappa.png"))
    write_audit_numbers(d, ctrl, extra, os.path.join(FIGS, "audit_numbers.tex"),
                        os.path.join(FIGS, "audit_numbers.csv"))
