#!/usr/bin/env python3
"""
Figures and tables for the two NEW deck sections (October 2026 update).

  SECTION "THE AUDIT" (step 95: every metric of every model, re-checked)
    figures/audit_status.png          one bar per kind of row in the step 95
                                      table, split by what the re-check found
    figures/audit_moduli_table.tex    headline accuracy on matbench test
    figures/audit_truth_table.tex     the only numbers scored against REAL
                                      physics (experiment / phonon DFT)

  SECTION "THE DFT LIST" (the 15 handed to the supervisor)
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

# Same palette as make_figures.py / make_alignn_figures.py, so the new
# figures look like the old ones. Each name is a hex colour string.
NAVY, BLUE, LIGHT_BLUE = "#0F1B33", "#1450AA", "#6FA8FF"
GREEN, AMBER, RED, INK = "#1E8E5A", "#D97B12", "#B02A2A", "#1F2937"
MUTED, GRID, SUBTLE = "#5B6B82", "#E1E6EF", "#AAB6C8"


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
    order = [("match", "recomputed = stored", GREEN),
             ("close", "close (within 0.1%)", LIGHT_BLUE),
             ("new", "recomputed, never stored before", BLUE),
             ("stored only", "stored only (inputs gone)", SUBTLE),
             ("MISMATCH", "MISMATCH", RED)]

    fig, ax = plt.subplots(figsize=(12.5, 2.9))
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
                text_colour = "white" if colour not in (SUBTLE, LIGHT_BLUE) else INK
                ax.text(left + share / 2, y, f"{n:,}", ha="center", va="center",
                        color=text_colour, fontsize=10, fontweight="bold")
            elif n > 0:
                # too thin to write inside: label it above the bar, with an arrow
                ax.annotate(f"{n} {status}", (left + share / 2, y - 0.31), xytext=(0, 12),
                            textcoords="offset points", ha="center", color=BLUE,
                            fontsize=9, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
            left += share
    # The checks bar is all "match" - say so explicitly, and say what is NOT there.
    ax.text(101, 1, "all pass", va="center", color=GREEN, fontsize=10, fontweight="bold")
    n_bad = int(m.get("MISMATCH", 0)) + int(c.get("MISMATCH", 0))
    ax.text(101, 0, f"{n_bad} MISMATCH", va="center", color=GREEN if n_bad == 0 else RED,
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
        ("Joint CGCNN, 3-ensemble (r5 A)",
         dict(family="joint CGCNN", run="r5_ensA"),
         dict(family="joint CGCNN", run="r5_ensA")),
        ("ALIGNN, 1 model (seed 42)",
         dict(family="ALIGNN moduli", model="ALIGNN seed 42"), None),
        ("ALIGNN, 3-ensemble",
         dict(family="ALIGNN moduli", model="ALIGNN 3-ensemble"),
         dict(family="ALIGNN separate -> kappa", model="ALIGNN 3-ensemble, K and G")),
    ]
    lines = []
    for label, mod, kap in rows:
        # The joint model stores its per-modulus MAE under the kappa target,
        # the others under K_VRH / G_VRH - so ask the right place for each.
        if mod["family"] == "joint CGCNN":
            kmae = one("mae_log_K", target="K+G -> kappa", **mod)
            gmae = one("mae_log_G", target="K+G -> kappa", **mod)
        else:
            kmae = one("MAE_log10", target="K_VRH", **mod)
            gmae = one("MAE_log10", target="G_VRH", **mod)
        kr2 = one("R2_log10", target="K_VRH", **mod)
        gr2 = one("R2_log10", target="G_VRH", **mod)
        kappa = None if kap is None else one("kappa_mae_total", target="K+G -> kappa", **kap)
        lines.append(f"{label} & {f4(kmae)} & {f3(kr2)} & {f4(gmae)} & {f3(gr2)} & {f4(kappa)} \\\\")
        print(f"  {label:34s} K {kmae:.4f} R2 {kr2:.3f}  G {gmae:.4f} R2 {gr2:.3f}  kappa "
              f"{'---' if kappa is None else f'{kappa:.4f}'}")
    with open(path, "w") as fh:                 # "w" = write (replace the file)
        fh.write("% written by make_audit_figures.py from 95_final_all_metrics.csv - do not edit\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def table_truth(path):
    """The few scores against REAL physics: measured kappa, phonon-DFT kappa."""
    t1 = dict(family="63 Table 1 experiment")
    p87 = dict(family="87 PhoNIX Poisson chain", run="truth klat, low = kappa <= 1.0")
    # split="out-of-fold": every crystal predicted once, by the fold that did not train on it
    dk = dict(family="direct kappa", model="direct ALIGNN (PhoNIX, 5-fold OOF)", split="out-of-fold")
    g89 = dict(family="89 MLIP vs Poisson", model="max of three", run="MLIP 300 K vs Poisson",
               target="DFT klat <= 1")

    prec = one("precision", model="kappa_max3", **p87)
    lo, hi = one("precision_lo95", model="kappa_max3", **p87), one("precision_hi95", model="kappa_max3", **p87)
    gain = one("precision gain, MLIP minus Poisson", **g89)
    glo = one("precision gain, MLIP minus Poisson lo95", **g89)
    ghi = one("precision gain, MLIP minus Poisson hi95", **g89)

    # Table 1 numbers. The run names contain an apostrophe, so they are
    # written in double quotes: "Table 1's" is fine inside "...".
    t1_poisson = one("MAE_log10 vs experiment", run="CGCNN ens, own Poisson gamma", **t1)
    t1_real = one("MAE_log10 vs experiment", run="CGCNN ens, Table 1's gamma", **t1)
    t1_paper = one("MAE_log10 vs experiment", run="PINK paper's own kappa", **t1)
    dk_mae, dk_r2, dk_rho = (one(x, **dk) for x in ("MAE_log10", "R2_log10", "spearman_rho"))

    # Each line: test set & model & what was measured & the number.
    # TinyTeX here has no multirow package, so a block's label is simply
    # split over its rows (line 1 of the label on row 1, line 2 on row 2).
    k = "$\\kappa_L$"
    lines = [
        f"PINK Table 1 & CGCNN ens., $\\gamma$ from Poisson & MAE log$_{{10}}$ & {f3(t1_poisson)} \\\\",
        f"45 measured {k} & CGCNN ens., Table 1's $\\gamma$ & MAE log$_{{10}}$ & {f3(t1_real)} \\\\",
        f" & the PINK paper itself & MAE log$_{{10}}$ & {f3(t1_paper)} \\\\",
        "\\midrule",
        f"PhoNIX phonon DFT & Poisson chain, max of 3 & precision at $\\kappa\\le1$ & {f3(prec)} [{f3(lo)}, {f3(hi)}] \\\\",
        f"2,520 unseen & MLIP $\\gamma$ instead (451 sampled) & precision gain & {gain:+.3f} [{glo:+.3f}, {ghi:+.3f}] \\\\",
        "\\midrule",
        f"PhoNIX, 5-fold & direct ALIGNN, structure$\\to${k} & MAE log$_{{10}}$ & {f3(dk_mae)} \\\\",
        f"5,316 held out once & & $R^2$ / Spearman $\\rho$ & {f3(dk_r2)} / {f3(dk_rho)} \\\\",
    ]
    with open(path, "w") as fh:
        fh.write("% written by make_audit_figures.py from 95_final_all_metrics.csv - do not edit\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    print(f"  PhoNIX precision {prec:.3f} [{lo:.3f}, {hi:.3f}];  MLIP gain {gain:+.3f} [{glo:+.3f}, {ghi:+.3f}]")


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
        lines.append(
            f"{int(r['rank'])} & {latex_formula(r.formula)} & {int(r.n_atoms)} & "
            f"{latex_sg(e['sg'])} & {cell} & {spin} & "
            f"{r.kappa_max3:.2f} & {r.kappa_stress:.2f} & {r.kappa_poisson_max3:.2f} & "
            f"{r.kappa_direct:.2f} \\\\")
    with open(path, "w") as fh:
        fh.write("% written by make_audit_figures.py from dft/final_shortlist_15/ - do not edit\n")
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def figure_shortlist(d, ctrl, path):
    """One row per crystal; three chains' kappa on a log axis; kappa = 1 line."""
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

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.axvspan(1e-2, 1.0, color=GREEN, alpha=0.06, lw=0)      # the target zone
    ax.axvline(1.0, color=RED, lw=1.4, ls="--")
    ax.text(1.06, -0.9, "target: $\\kappa_L \\leq 1$ W/m/K", color=RED, fontsize=9, va="center")

    for yi, row in zip(y, rows):                # zip pairs the two lists item by item
        # The MLIP chain's range: best guess -> worst plausible gamma error
        if not np.isnan(row["stress"]):
            ax.plot([row["mlip"], row["stress"]], [yi, yi], color=BLUE, lw=2.2, alpha=0.35,
                    solid_capstyle="round")
            ax.plot(row["stress"], yi, "|", color=BLUE, ms=11, mew=2)
        ax.plot(row["mlip"], yi, "o", color=BLUE, ms=7)
        ax.plot(row["poisson"], yi, "D", color=AMBER, ms=6)
        ax.plot(row["direct"], yi, "s", color=GREEN, ms=6.5)

    # Legend entries drawn once, off-screen, so each symbol appears exactly once
    ax.plot([], [], "o", color=BLUE, ms=7, label="Slack + MLIP phonon $\\gamma$ (max of 3 models)")
    ax.plot([], [], "|", color=BLUE, ms=11, mew=2, label="... with $\\gamma$ cut by 32% (stress test)")
    ax.plot([], [], "D", color=AMBER, ms=6, label="Slack + Poisson $\\gamma$ (max of 3 models)")
    ax.plot([], [], "s", color=GREEN, ms=6.5, label="direct ALIGNN, trained on 6,641 phonon-DFT $\\kappa_L$")

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


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    m, c = figure_audit_status(os.path.join(FIGS, "audit_status.png"))
    print("  metric rows by status:", dict(m), "| checks:", dict(c))
    table_moduli(os.path.join(FIGS, "audit_moduli_table.tex"))
    table_truth(os.path.join(FIGS, "audit_truth_table.tex"))
    d, ctrl, extra = read_shortlist()
    table_shortlist(d, extra, os.path.join(FIGS, "shortlist_table.tex"))
    figure_shortlist(d, ctrl, os.path.join(FIGS, "shortlist_kappa.png"))
