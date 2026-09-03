#!/usr/bin/env python3
"""
STEP 19 - Pull out the small-cell candidates from the GNoME screen.
================================================================================

    python scripts/cgcnn/19_small_cells.py

WHAT THIS IS
-------------
13_screen_gnome.py scored all 33,323 filtered GNoME candidates for K, G and
kappa_L, and recorded each one's primitive-cell atom count along the way
(structure_to_graph()'s primitive-cell reduction, not whatever supercell
GNoME's own CIF happened to use). This script does not re-run any of that -
it just reads the already-scored CSV and pulls out the candidates with a
SMALL primitive cell (< 10 atoms), which matters for two practical reasons a
materials scientist would actually care about:

  - DFT verification cost scales steeply with atom count, so a small cell is
    far cheaper to actually check computationally before trusting it.
  - A small primitive cell usually means a simple, highly symmetric
    structure - the kind that is easier to synthesize and characterize than
    a sprawling 80-atom disordered one.

WHY < 10, NOT SOME OTHER CUTOFF
-----------------------------------
10 is a round, commonly-used threshold in materials screening for "cheap to
verify" (a single-point DFT calculation on a <10-atom cell is routine on a
laptop-scale cluster; the screen's own atom counts range 3-84, median 22,
so this is a genuine minority, not most of the data - see the printed
distribution below).

WHAT'S KEPT FROM THE SCREEN, AND WHAT'S ADDED
-----------------------------------------------
Every column from 13_screen_gnome.py's output rides along unchanged. One
column added:

    is_low_kappa_candidate   True if this candidate also cleared the
                              screen's own kappa_L <= 1 W/m/K threshold -
                              kept explicit rather than silently restricting
                              to only the sub-threshold rows, so this file
                              answers "which small cells did we screen" AND
                              "which of those are actual low-kappa finds" at
                              once, same convention as 15_filter_oxides.py.

Writes two files: every scored small cell (gnome_screen_small_cells.csv),
and a second, smaller one containing only the ones that also cleared the
kappa_L <= 1 threshold (gnome_screen_small_cells_low_kappa.csv) - the actual
low-kappa small-cell finds, not just "small cells that got scored at all".
"""

import argparse                      # command-line flag parsing for the file paths/thresholds below
import os                            # path joining/checking

import matplotlib
matplotlib.use("Agg")                # non-interactive backend - this script only saves a PNG, never shows a window
import matplotlib.pyplot as plt      # figure/axes creation and saving
import numpy as np                   # np.arange for the histogram bin edges
import pandas as pd                  # CSV I/O and DataFrame filtering/sorting

# walk up 3 levels from this file (scripts/cgcnn/<this file>) to the repo root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared results directory

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


def plot_atom_count(df, small, max_atoms_shown, path):
    """Atom-count histogram, full screen vs. the small-cell subset, with a
    vertical line marking the < 10 cutoff. Bins span the FULL 3-84 range
    (real per-atom-count heights, nothing accumulated into an overflow bin)
    - max_atoms_shown only crops the x-axis window so the long tail doesn't
    crush the interesting region near the cutoff; it does not distort any
    bar's height."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    max_atoms = int(df["Number of Atoms"].max())              # largest primitive cell in the full screen
    bins = np.arange(3, max_atoms + 2) - 0.5                  # one bin per integer atom count, edges offset so each bar centers on its count
    ax.hist(df["Number of Atoms"], bins=bins, color=BLUE, alpha=0.55,
           label=f"All screened candidates (n={len(df)})", edgecolor="none")   # background histogram: full screen
    ax.hist(small["Number of Atoms"], bins=bins, color=ORANGE, alpha=0.85,
           label=f"< 10 atoms (n={len(small)})", edgecolor="none")            # overlaid histogram: small-cell subset only
    ax.axvline(9.5, color=INK_SOFT, linewidth=1.0, linestyle="--")  # dashed line marking the <10-atom cutoff
    ax.set_xlim(2, max_atoms_shown)  # crop the visible x-range only - does not affect the underlying bin heights
    ax.text(9.5, ax.get_ylim()[1] * 0.95, " < 10 atoms", fontsize=8.5, color=INK_SOFT, va="top")  # label the cutoff line near the top of the plot
    ax.set_xlabel(f"Primitive-cell atom count (view cropped at {max_atoms_shown}; "
                 f"full data range 3-{max_atoms})")
    ax.set_ylabel("Number of candidates")
    ax.set_title("Primitive-cell size across the GNoME screen")
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.tight_layout()       # shrink margins so labels don't get clipped
    fig.savefig(path, dpi=160)
    plt.close(fig)           # free the figure's memory


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # input: 13_screen_gnome.py's full, unfiltered scored-candidate table
    parser.add_argument("--screen-csv", default=os.path.join(RESULTS, "13_gnome_screen_all.csv"))
    parser.add_argument("--max-atoms", type=int, default=10,
                        help="exclusive upper bound - candidates with fewer atoms than this are kept")
    parser.add_argument("--kappa-threshold", type=float, default=1.0)
    # output: every scored small cell, "19_" prefix (read downstream by 28_alignn_small_cell_filter.py)
    parser.add_argument("--out", default=os.path.join(RESULTS, "19_gnome_screen_small_cells.csv"))
    # output: just the low-kappa small-cell subset, "19_" prefix (read downstream by 22_best_dft_candidate.py)
    parser.add_argument("--out-low-kappa",
                        default=os.path.join(RESULTS, "19_gnome_screen_small_cells_low_kappa.csv"),
                        help="second file, the is_low_kappa_candidate subset only - the actual "
                             "low-kappa small-cell finds, not every small cell scored")
    # output: the atom-count histogram, "19_" prefix (terminal)
    parser.add_argument("--out-png", default=os.path.join(RESULTS, "19_gnome_screen_atom_count.png"))
    args = parser.parse_args()  # reads sys.argv; every flag above falls back to its default if not passed

    print("=== Filtering the GNoME screen for small-cell candidates ===\n")
    if not os.path.exists(args.screen_csv):
        raise SystemExit(f"{args.screen_csv} not found - run scripts/cgcnn/13_screen_gnome.py first")

    df = pd.read_csv(args.screen_csv)
    print(f"Loaded {len(df)} scored candidates from {args.screen_csv}")
    print(f"  Number of Atoms: min={df['Number of Atoms'].min()}, "
         f"median={df['Number of Atoms'].median():.0f}, max={df['Number of Atoms'].max()}")

    small = df[df["Number of Atoms"] < args.max_atoms].copy()  # boolean-mask filter to small cells, then copy to avoid a pandas view/copy warning
    print(f"\n{len(small)} of {len(df)} have fewer than {args.max_atoms} atoms "
         f"({len(small) / len(df) * 100:.1f}%)")

    small["is_low_kappa_candidate"] = small["Kappa_cal (W m-1 K-1)"] <= args.kappa_threshold  # per-row boolean: clears the threshold
    small = small.sort_values(["Number of Atoms", "Kappa_cal (W m-1 K-1)"]).reset_index(drop=True)  # smallest cells first, then lowest kappa within a tie

    n_candidates = int(small["is_low_kappa_candidate"].sum())  # True/False column summed as 1/0
    print(f"  {n_candidates} of those also clear the kappa_L <= {args.kappa_threshold} "
         f"W/m/K threshold (real low-kappa, small-cell candidates)")

    print("\nBy atom count:")
    for n_atoms, count in small["Number of Atoms"].value_counts().sort_index().items():  # iterate atom counts in ascending order
        n_low = int(small[(small["Number of Atoms"] == n_atoms) &
                          small.is_low_kappa_candidate].shape[0])  # low-kappa count at this exact atom count
        print(f"  {n_atoms:3d} atoms   {count:5d} total, {n_low:5d} low-kappa candidates")

    small.to_csv(args.out, index=False)  # index=False: don't write the pandas row-number column
    print(f"\nWrote {args.out} ({len(small)} rows, sorted by atom count then Kappa_cal ascending)")

    low_kappa = small[small["is_low_kappa_candidate"]].drop(columns=["is_low_kappa_candidate"])  # subset rows, then drop the now-constant-True column
    low_kappa.to_csv(args.out_low_kappa, index=False)
    print(f"Wrote {args.out_low_kappa} ({len(low_kappa)} rows - just the small cells that "
         f"cleared kappa_L <= {args.kappa_threshold})")

    plot_atom_count(df, small, max_atoms_shown=30, path=args.out_png)
    print(f"Wrote {args.out_png}")


if __name__ == "__main__":  # only run main() when executed directly, not when imported
    main()
