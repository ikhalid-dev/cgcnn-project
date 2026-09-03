#!/usr/bin/env python3
"""
STEP 18 - Element frequency across the GNoME screen.
================================================================================

    python scripts/cgcnn/18_element_frequency.py

WHAT THIS IS
-------------
13_screen_gnome.py scored 33,323 candidates; 15_filter_oxides.py already asked
"does this contain oxygen" as a yes/no split. This script generalises that to
EVERY element: for each element that appears anywhere in the screen, how many
candidates contain it, and - the more interesting number - is it OVER- or
UNDER-represented among the low-kappa_L (<=1 W/m/K) candidates relative to
its share of the whole screen. An element in exactly its background
proportion of low-kappa hits has enrichment = 1.0; enrichment > 1 means that
element's compounds skew toward low kappa_L more than average, < 1 means the
opposite.

    enrichment(element) = (fraction of low-kappa candidates containing it)
                         / (fraction of ALL candidates containing it)

WHY PYMATGEN, NOT A STRING SPLIT
------------------------------------
Same reasoning as 15_filter_oxides.py's has_oxygen(): reduced formulas like
"Ho7Er(OsBr4)2" have nested parentheses and multi-letter symbols, so element
sets are read via pymatgen's Composition parser, not a regex.

Writes one CSV (element, n_total, n_low_kappa, frac_of_all, frac_of_low_kappa,
enrichment - sorted by n_total descending) and one bar chart: the top 30
elements by how many candidates contain them, all-candidates vs low-kappa
overlaid, in the same style as the space-group histograms.
"""

import os                            # path joining
import warnings                      # suppress a noisy, harmless pymatgen warning below

import matplotlib
matplotlib.use("Agg")                # non-interactive backend - this script only saves PNGs, never shows a window
import matplotlib.pyplot as plt      # figure/axes creation and saving
import numpy as np                   # np.arange for the bar chart's y-positions
import pandas as pd                  # CSV I/O and DataFrame aggregation
from pymatgen.core import Composition  # parses a chemical formula into its constituent elements

warnings.filterwarnings("ignore")    # silence pymatgen's formula-parsing warnings for the whole script

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory
# input: 13_screen_gnome.py's full, unfiltered scored-candidate table
SCREEN_CSV = os.path.join(RESULTS, "13_gnome_screen_all.csv")
# outputs: this script's own element-frequency table and chart, "18_" prefix (both terminal)
OUT_CSV = os.path.join(RESULTS, "18_gnome_screen_elements.csv")
OUT_PNG = os.path.join(RESULTS, "18_gnome_screen_elements.png")
KAPPA_THRESHOLD = 1.0  # W/m/K cutoff used throughout this project to flag a "low-kappa candidate"
TOP_N = 30             # how many of the most common elements to plot

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({  # global matplotlib style overrides, applied to every figure created after this point
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,  # drop the top/right plot border for a cleaner look
})


def elements_of(formula):
    try:
        return {str(el) for el in Composition(formula).elements}  # parse into a set of element symbols
    except Exception:
        return set()  # unparseable formula: treat as containing no elements rather than crashing


def element_table(df):
    """One row per element that appears anywhere in df, with total/low-kappa
    counts and the enrichment ratio defined in the module docstring."""
    total_n = len(df)                                     # total number of candidates in this screen
    low_n = int(df["is_low_kappa_candidate"].sum())        # how many of them are low-kappa

    elem_sets = df["formula"].apply(elements_of)           # one set of element symbols per row
    low_mask = df["is_low_kappa_candidate"].values         # parallel boolean array, same row order as elem_sets

    counts_total, counts_low = {}, {}                      # {element symbol: candidate count}, overall and low-kappa-only
    for elems, is_low in zip(elem_sets, low_mask):          # walk both arrays together, one candidate at a time
        for el in elems:                                    # every distinct element this candidate's formula contains
            counts_total[el] = counts_total.get(el, 0) + 1  # increment (or initialise at 1) this element's total count
            if is_low:
                counts_low[el] = counts_low.get(el, 0) + 1  # also increment its low-kappa count if this candidate qualifies

    rows = []
    for el, n_total in counts_total.items():                # one row per distinct element seen
        n_low = counts_low.get(el, 0)                       # 0 if this element never appeared in a low-kappa candidate
        frac_all = n_total / total_n                        # this element's share of ALL candidates
        frac_low = n_low / low_n if low_n else 0.0          # this element's share of LOW-KAPPA candidates (guard divide-by-zero)
        enrichment = (frac_low / frac_all) if frac_all > 0 else float("nan")  # ratio defined in the module docstring
        rows.append({
            "element": el, "n_total": n_total, "n_low_kappa": n_low,
            "frac_of_all": frac_all, "frac_of_low_kappa": frac_low,
            "enrichment": enrichment,
        })
    table = pd.DataFrame(rows).sort_values("n_total", ascending=False).reset_index(drop=True)  # most common element first
    return table, total_n, low_n


def plot_top_elements(table, total_n, low_n, path, top_n=TOP_N):
    top = table.head(top_n).iloc[::-1]  # reverse so the most common element is at the top of the barh
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * top_n + 1.5))  # figure height scales with how many bars are drawn
    y = np.arange(len(top))  # integer y-position for each bar, 0..len(top)-1
    ax.barh(y, top["n_total"], color=BLUE, alpha=0.6,
           label=f"All screened candidates (n={total_n})")   # background bar: how common this element is overall
    ax.barh(y, top["n_low_kappa"], color=ORANGE, alpha=0.9,
           label=f"Low-κ candidates (n={low_n})")             # overlaid bar: how common it is among low-kappa candidates only
    ax.set_yticks(y)
    ax.set_yticklabels(top["element"], fontsize=9)  # label each bar with its element symbol
    ax.set_xlabel("Number of candidates containing this element")
    ax.set_title(f"Most common elements in the GNoME screen (top {top_n})")
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.tight_layout()       # shrink margins so labels don't get clipped
    fig.savefig(path, dpi=160)
    plt.close(fig)           # free the figure's memory - matters when a script makes many plots in one run


def main():
    print("=== Element frequency across the GNoME screen ===\n")
    df = pd.read_csv(SCREEN_CSV)
    df["is_low_kappa_candidate"] = df["Kappa_cal (W m-1 K-1)"] <= KAPPA_THRESHOLD  # per-row boolean: clears the threshold
    print(f"Loaded {len(df)} scored candidates from {SCREEN_CSV} "
         f"({int(df.is_low_kappa_candidate.sum())} clear kappa_L <= {KAPPA_THRESHOLD} W/m/K)")

    table, total_n, low_n = element_table(df)
    print(f"\n{len(table)} distinct elements appear across the screen")

    print(f"\nTop 10 most common elements (by candidate count):")
    for _, row in table.head(10).iterrows():  # ten most common elements overall
        print(f"  {row.element:3s} {row.n_total:6d} candidates, "
             f"{row.n_low_kappa:5d} low-kappa, enrichment {row.enrichment:.2f}x")

    enriched = table[table.n_total >= 30].sort_values("enrichment", ascending=False)  # drop rare elements first - their enrichment ratio is too noisy to trust
    print(f"\nMost enriched among low-kappa candidates (n_total >= 30, to avoid noisy small counts):")
    for _, row in enriched.head(10).iterrows():  # highest enrichment first
        print(f"  {row.element:3s} enrichment {row.enrichment:.2f}x "
             f"({row.n_total:5d} total, {row.n_low_kappa:5d} low-kappa)")
    print(f"\nMost depleted among low-kappa candidates (n_total >= 30):")
    for _, row in enriched.tail(10).iloc[::-1].iterrows():  # lowest enrichment first (tail, then reversed)
        print(f"  {row.element:3s} enrichment {row.enrichment:.2f}x "
             f"({row.n_total:5d} total, {row.n_low_kappa:5d} low-kappa)")

    table.to_csv(OUT_CSV, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {OUT_CSV} ({len(table)} rows, sorted by n_total descending)")

    plot_top_elements(table, total_n, low_n, OUT_PNG)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
