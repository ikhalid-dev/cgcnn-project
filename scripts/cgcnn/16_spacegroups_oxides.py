#!/usr/bin/env python3
"""
STEP 16 - Space groups of the oxide candidates.
================================================================================

    python scripts/cgcnn/16_spacegroups_oxides.py

WHAT THIS IS
-------------
15_filter_oxides.py pulled the oxygen-containing rows out of the GNoME screen
into gnome_oxide_candidates.csv. This script tags each of those oxides with
its crystallographic space group. That is NOT a new structure analysis -
GNoME's own summary CSV (gnome_data/stable_materials_summary.csv) already
carries "Space Group", "Space Group Number" and "Crystal System" per
material, computed by GNoME itself when the structures were generated. So
this is a join on MaterialId, reusing data that already exists on disk,
not a re-parse of the CIFs.

Writes one CSV (every oxide candidate, all of gnome_oxide_candidates.csv's
own columns plus the three space-group columns, SORTED by crystal system in
symmetry order - triclinic through cubic, matching CRYSTAL_SYSTEM_RANGES,
not alphabetical - then by space group number, then by predicted Kappa_cal
ascending within each space group) and one histogram: the distribution of
Space Group Number (1-230, International Tables numbering) across all oxide
candidates, with the low-kappa subset overlaid and the seven crystal-system
windows marked along the top axis.

Also writes two "best candidate" CSVs - the single lowest-predicted-Kappa_cal
oxide in each crystal system (7 rows), and the single lowest-predicted-Kappa_cal
oxide in each space group (one row per space group actually present in the
oxide screen). These are not restricted to rows that already cleared the
kappa_L <= 1 W/m/K threshold: a crystal system/space group with zero confirmed
low-kappa hits still gets its closest candidate shown, with
is_low_kappa_candidate visible so it is always clear whether that "best"
candidate actually cleared the threshold or is merely the least-bad option in
an otherwise unpromising group.

Also breaks that down by 15_filter_oxides.py's own stoichiometry_pattern tag
(ABO3-type, AB2O4-type, A2B2O7-type, ternary, complex-N-element) - a second,
long-format CSV (one row per pattern x crystal-system pair, with both the
total and low-kappa counts) and a small-multiples bar chart, one panel per
pattern, so a rare pattern (e.g. the 8 AB2O4-type oxides) is still legible
next to the two dominant complex-oxide patterns (11,035 of 11,612 rows)
instead of being crushed flat on a single shared axis.

WHY THE JOIN DROPS ~1% OF ROWS
--------------------------------
2,986 of the 554,054 rows in the GNoME summary CSV have no MaterialId at all
(a gap in the upstream file itself, not something this project introduces).
A handful of this screen's oxide candidates happen to be among them and
simply have no space-group row to join against - reported explicitly by this
script (count printed, columns left blank), not silently dropped from the
output.
"""

import os  # path joining

import matplotlib  # plotting library, backend selected below before pyplot is imported
matplotlib.use("Agg")  # non-interactive backend - writes image files, no display needed
import matplotlib.pyplot as plt  # the plotting API used throughout this file
import numpy as np  # array math (bin edges, x-axis positions)
import pandas as pd  # DataFrame I/O, joins, and grouping

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, 3 levels above this file
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")  # shared directory for this pipeline's CSV/PNG output
GNOME_SUMMARY = os.path.join(PROJECT_ROOT, "gnome_data", "stable_materials_summary.csv")  # GNoME's own per-material summary table (external, unrenamed)
OXIDE_CSV = os.path.join(RESULTS, "15_gnome_oxide_candidates.csv")  # 15_filter_oxides.py's output - this script's input
OUT_CSV = os.path.join(RESULTS, "16_gnome_oxide_spacegroups.csv")  # this script's main output table
OUT_PNG = os.path.join(RESULTS, "16_gnome_oxide_spacegroups.png")  # the space-group histogram
STOICH_CSV = os.path.join(RESULTS, "16_gnome_oxide_by_stoichiometry.csv")  # stoichiometry x crystal-system breakdown table
STOICH_PNG = os.path.join(RESULTS, "16_gnome_oxide_by_stoichiometry.png")  # the small-multiples bar chart for that breakdown
BEST_BY_CS_CSV = os.path.join(RESULTS, "16_gnome_oxide_best_by_crystal_system.csv")  # one best row per crystal system
BEST_BY_SG_CSV = os.path.join(RESULTS, "16_gnome_oxide_best_by_spacegroup.csv")  # one best row per space group

BLUE = "#2a78d6"  # primary series color (all candidates)
ORANGE = "#eb6834"  # highlight series color (low-kappa subset)
INK = "#0b0b0b"  # main text/title color
INK_SOFT = "#52514e"  # secondary text color (axis labels)
MUTED = "#898781"  # tertiary color (tick labels)
GRID = "#e1e0d9"  # gridline / divider color
SURFACE = "#fcfcfb"  # figure/axes background color

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})  # global matplotlib style overrides applied to every figure this script makes

# International Tables space-group ranges, used to mark crystal-system
# windows on the histogram (boundaries per the standard 230-group numbering).
CRYSTAL_SYSTEM_RANGES = [
    ("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142), ("trigonal", 143, 167), ("hexagonal", 168, 194),
    ("cubic", 195, 230),
]  # (system name, first space-group number, last space-group number) tuples, in symmetry order


def load_space_groups():
    """The GNoME summary CSV's own space-group columns, deduped to one row
    per MaterialId (a small number of rows share no MaterialId at all - see
    module docstring - those are dropped here since they cannot be joined
    against anyway)."""
    sg = pd.read_csv(GNOME_SUMMARY,
                     usecols=["MaterialId", "Space Group", "Space Group Number", "Crystal System"])  # only the 4 columns needed, for speed
    sg = sg.dropna(subset=["MaterialId"]).drop_duplicates("MaterialId")  # drop unjoinable rows, then one row per material
    return sg.rename(columns={
        "Space Group": "space_group",
        "Space Group Number": "space_group_number",
        "Crystal System": "crystal_system",
    })  # snake_case column names, matching this project's convention


def sort_by_crystal_structure(df):
    """Crystal system in symmetry order (triclinic -> cubic, per
    CRYSTAL_SYSTEM_RANGES - NOT alphabetical), then space group number, then
    predicted Kappa_cal ascending as a tiebreaker within each space group.
    Rows with no matched space-group row (see module docstring) sort to the
    end, since pandas puts NaN categories last by default."""
    system_order = [name for name, _, _ in CRYSTAL_SYSTEM_RANGES]  # just the names, in symmetry order
    out = df.copy()  # avoid mutating the caller's DataFrame
    out["crystal_system"] = pd.Categorical(out["crystal_system"], categories=system_order, ordered=True)  # ordered categorical so sort_values uses symmetry order, not alphabetical
    return out.sort_values(
        ["crystal_system", "space_group_number", "Kappa_cal (W m-1 K-1)"]
    ).reset_index(drop=True)  # three-key sort, then a fresh 0..n-1 index


def best_by_group(df, group_col, rank_col="Kappa_cal_p95"):
    """The single best row within each value of group_col ("crystal_system"
    or "space_group"), ranked by rank_col ascending - best-first. Not
    restricted to rows that already cleared the low-kappa threshold - see
    module docstring for why.

    RANKED BY p95, NOT THE POINT ESTIMATE - AND THAT CHANGE IS DELIBERATE.
    Ranking 30k+ candidates by lowest point-estimate Kappa_cal and taking the
    minimum is a textbook winner's-curse setup: whichever candidate happens
    to have both a low prediction AND high ensemble disagreement gets
    over-represented at the top, because a wide, noisy interval has more
    chance of a low point estimate than a tight, confident one does. Checked
    directly on this exact data: ranking by the point estimate, the
    "best" oxide (ReAsOF10) and 2nd-best general candidate (HgWCl10) both
    carry p95/p05 ratios of ~19-22x - solidly in the "worth a DFT check
    before trusting" bucket by this project's own convention (see
    docs/oxide_screen_columns.tex), not the confident end of the range.
    Ranking by Kappa_cal_p95 instead asks a different, more conservative
    question - "which candidate is still predicted low-kappa even in this
    model's own pessimistic (95th percentile) scenario" - and rewards
    confident, tightly-bounded predictions over lucky, noisy ones."""
    valid = df.dropna(subset=[group_col])  # rows that actually have a group value to key on
    best_idx = valid.groupby(group_col, observed=True)[rank_col].idxmin()  # index label of the lowest rank_col within each group
    return valid.loc[best_idx].sort_values(rank_col).reset_index(drop=True)  # those rows, re-sorted best-first, fresh index


def plot_histogram(df, path, title="Space groups of the GNoME oxide screen",
                   all_label="All oxide candidates", low_label="Low-κ oxide candidates"):
    """Space-group-number histogram, all candidates vs. the low-kappa subset
    overlaid, with the seven crystal-system windows marked along the top
    axis. title/all_label/low_label are parametrised so 17_spacegroups_general.py
    can reuse this unchanged for the full (non-oxide-filtered) screen."""
    all_sgn = df["space_group_number"].dropna()  # every candidate's space-group number, NaNs excluded
    low_sgn = df.loc[df["is_low_kappa_candidate"], "space_group_number"].dropna()  # same, restricted to the low-kappa subset

    fig, ax = plt.subplots(figsize=(9.5, 5.2))  # one figure, one axes, fixed size in inches
    bins = np.arange(1, 232) - 0.5  # bin edges centered on each integer 1..230
    ax.hist(all_sgn, bins=bins, color=BLUE, alpha=0.55,
           label=f"{all_label} (n={len(all_sgn)})", edgecolor="none")  # background histogram, all candidates
    ax.hist(low_sgn, bins=bins, color=ORANGE, alpha=0.85,
           label=f"{low_label} (n={len(low_sgn)})", edgecolor="none")  # overlaid histogram, low-kappa subset only

    for name, lo, hi in CRYSTAL_SYSTEM_RANGES[1:]:  # skip the first system - no left boundary line needed before it
        ax.axvline(lo - 0.5, color=GRID, linewidth=0.8, linestyle=":")  # dotted vertical divider at each system boundary

    ax2 = ax.twiny()  # a second x-axis sharing the same y-axis, for the crystal-system labels on top
    ax2.set_xlim(ax.get_xlim())  # keep it aligned with the main axis's x-range
    mids = [(lo + hi) / 2 for _, lo, hi in CRYSTAL_SYSTEM_RANGES]  # midpoint of each system's space-group range
    ax2.set_xticks(mids)  # place one tick at each system's midpoint
    ax2.set_xticklabels([name for name, _, _ in CRYSTAL_SYSTEM_RANGES],
                        fontsize=8, color=MUTED, rotation=20, ha="left")  # label each tick with the system name
    ax2.tick_params(length=0)  # hide the tick marks themselves, keep only the labels
    ax2.spines["top"].set_visible(False)  # hide the top axis border line

    ax.set_xlabel("Space group number (International Tables, 1-230)")
    ax.set_ylabel("Number of candidates")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.tight_layout()  # adjust spacing so labels/titles do not get clipped
    fig.savefig(path, dpi=160)  # write the PNG to disk
    plt.close(fig)  # free the figure's memory now that it is saved


def stoichiometry_breakdown(df):
    """Long-format table: one row per (stoichiometry_pattern, crystal_system)
    pair, with the total count and the low-kappa-candidate count. Patterns
    ordered by descending total size, crystal systems by the fixed
    triclinic->cubic symmetry order (not alphabetical) so both the console
    print and the CSV read in a consistent, physically meaningful order."""
    system_order = [name for name, _, _ in CRYSTAL_SYSTEM_RANGES]  # symmetry order for crystal systems
    pattern_order = df["stoichiometry_pattern"].value_counts().index.tolist()  # pattern names, most common first

    rows = []
    for pattern in pattern_order:  # one outer iteration per stoichiometry pattern
        sub = df[df.stoichiometry_pattern == pattern]  # rows belonging to this pattern
        for system in system_order:  # one inner iteration per crystal system, in symmetry order
            in_system = sub[sub.crystal_system == system]  # this pattern's rows that are also this crystal system
            rows.append({
                "stoichiometry_pattern": pattern,
                "crystal_system": system,
                "n_total": len(in_system),  # row count for this (pattern, system) pair
                "n_low_kappa": int(in_system["is_low_kappa_candidate"].sum()),  # count of those that also cleared the threshold
            })
    return pd.DataFrame(rows), pattern_order, system_order  # the long table plus both orderings, reused by the plot function


def plot_by_stoichiometry(df, pattern_order, system_order, path):
    """Small multiples, one bar-chart panel per stoichiometry pattern - a
    single shared axis would crush the 8-row AB2O4-type pattern to invisible
    next to the >4,000-row complex-oxide patterns, so each panel gets its own
    y-scale and only the crystal-system shape is being compared across
    panels, not the absolute magnitude."""
    ncols = 3  # fixed grid width
    nrows = -(-len(pattern_order) // ncols)  # ceiling division: enough rows to fit every pattern at 3 per row
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)  # grid of subplots, always 2-D indexable
    x = np.arange(len(system_order))  # bar x-positions, one per crystal system

    for i, pattern in enumerate(pattern_order):  # i = panel index, pattern = which stoichiometry pattern this panel shows
        ax = axes[i // ncols][i % ncols]  # row/column of this panel within the grid
        sub = df[df.stoichiometry_pattern == pattern]  # rows for this pattern only
        totals = sub["crystal_system"].value_counts().reindex(system_order, fill_value=0)  # counts per system, in symmetry order, missing systems filled with 0
        lows = sub.loc[sub.is_low_kappa_candidate, "crystal_system"] \
                  .value_counts().reindex(system_order, fill_value=0)  # same, restricted to the low-kappa subset
        ax.bar(x, totals.values, color=BLUE, alpha=0.6,
              label="All oxide candidates" if i == 0 else None)  # background bars; label only once, for the shared legend
        ax.bar(x, lows.values, color=ORANGE, alpha=0.9,
              label="Low-κ oxide candidates" if i == 0 else None)  # overlaid bars for the low-kappa subset
        ax.set_xticks(x)
        ax.set_xticklabels(system_order, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(f"{pattern}\n(n={len(sub)})", fontsize=9, color=INK)
        ax.tick_params(axis="y", labelsize=7.5)

    for j in range(len(pattern_order), nrows * ncols):  # any leftover grid cells beyond the last real pattern
        axes[j // ncols][j % ncols].axis("off")  # hide unused panels entirely

    handles, labels = axes[0][0].get_legend_handles_labels()  # reuse the first panel's legend entries for the whole figure
    fig.legend(handles, labels, loc="upper right", frameon=True,
              facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.suptitle("Oxide screen: crystal systems by stoichiometry pattern", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # leave room at the top for the suptitle
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    print("=== Space groups of the GNoME oxide screen ===\n")
    oxides = pd.read_csv(OXIDE_CSV)  # 15_filter_oxides.py's oxide-candidate table
    print(f"Loaded {len(oxides)} oxide candidates from {OXIDE_CSV}")

    sg = load_space_groups()  # GNoME's own space-group table, deduped and renamed
    merged = oxides.merge(sg, left_on="material_id", right_on="MaterialId", how="left") \
                   .drop(columns=["MaterialId"])  # left join keeps every oxide row even if unmatched; drop the now-redundant join key
    missing = merged["space_group"].isna().sum()  # count of oxide rows that found no matching GNoME summary row
    print(f"Joined against {GNOME_SUMMARY} on MaterialId")
    if missing:
        print(f"  {missing}/{len(merged)} oxide rows have no matching GNoME summary "
             f"row (upstream data gap, not this project's - see module docstring) "
             f"- space_group columns left blank for those")

    print("\nBy crystal system (all oxide candidates):")
    for system, count in merged["crystal_system"].value_counts().items():  # system = crystal system name, count = number of rows
        n_low = int(merged[(merged.crystal_system == system) &
                           merged.is_low_kappa_candidate].shape[0])  # rows in this system that also cleared the threshold
        print(f"  {system:14s} {count:6d} total, {n_low:6d} low-kappa candidates")

    sorted_df = sort_by_crystal_structure(merged)  # full table, three-key sorted
    sorted_df.to_csv(OUT_CSV, index=False)  # write without the pandas row-index column
    print(f"\nWrote {OUT_CSV} ({len(sorted_df)} rows, sorted by crystal system "
         f"-> space group number -> Kappa_cal ascending)")

    plot_histogram(merged, OUT_PNG)  # unsorted `merged` is fine here - the histogram does its own binning
    print(f"Wrote {OUT_PNG}")

    best_cs = best_by_group(merged, "crystal_system")  # one best-p95 row per crystal system
    best_cs.to_csv(BEST_BY_CS_CSV, index=False)
    print(f"\nWrote {BEST_BY_CS_CSV} ({len(best_cs)} rows - best oxide candidate per crystal system, "
         f"ranked by lowest Kappa_cal_p95, not the point estimate - see best_by_group() docstring)")
    print("Best oxide candidate per crystal system (lowest Kappa_cal_p95):")
    for _, row in best_cs.iterrows():  # _ = row index (unused), row = a pandas Series for that row
        flag = "cleared threshold" if row.is_low_kappa_candidate else "did NOT clear threshold"
        ratio = row.Kappa_cal_p95 / row.Kappa_cal_p05  # this row's Monte Carlo interval width
        print(f"  {row.crystal_system:14s} {row.formula:16s} point={row['Kappa_cal (W m-1 K-1)']:.4f} "
             f"p95={row.Kappa_cal_p95:.4f} (p95/p05={ratio:.1f}x) (sg {row.space_group}, {flag})")

    best_sg = best_by_group(merged, "space_group")  # one best-p95 row per space group
    best_sg.to_csv(BEST_BY_SG_CSV, index=False)
    print(f"\nWrote {BEST_BY_SG_CSV} ({len(best_sg)} rows - best oxide candidate per space group, "
         f"ranked by lowest Kappa_cal_p95)")
    print("Top 10 overall (best oxide candidate per space group, ranked by Kappa_cal_p95):")
    for _, row in best_sg.head(10).iterrows():  # the 10 lowest-p95 rows, since best_by_group() already sorted ascending
        flag = "cleared threshold" if row.is_low_kappa_candidate else "did NOT clear threshold"
        ratio = row.Kappa_cal_p95 / row.Kappa_cal_p05
        print(f"  {row.formula:16s} point={row['Kappa_cal (W m-1 K-1)']:.4f} p95={row.Kappa_cal_p95:.4f} "
             f"(p95/p05={ratio:.1f}x) (sg {row.space_group}, {row.crystal_system}, {flag})")

    stoich_table, pattern_order, system_order = stoichiometry_breakdown(merged)  # long-format counts, plus both display orderings
    print("\nBy stoichiometry pattern x crystal system:")
    for pattern in pattern_order:  # one block per pattern, most common pattern first
        sub = stoich_table[stoich_table.stoichiometry_pattern == pattern]  # this pattern's rows across all 7 crystal systems
        total = int(sub.n_total.sum())  # total candidates in this pattern, across all systems
        print(f"  {pattern} (n={total}):")
        for _, row in sub[sub.n_total > 0].sort_values("n_total", ascending=False).iterrows():  # skip empty (pattern, system) pairs, largest first
            print(f"      {row.crystal_system:14s} {row.n_total:6d} total, "
                 f"{row.n_low_kappa:6d} low-kappa candidates")

    stoich_table.to_csv(STOICH_CSV, index=False)
    print(f"\nWrote {STOICH_CSV} ({len(stoich_table)} rows)")

    plot_by_stoichiometry(merged, pattern_order, system_order, STOICH_PNG)
    print(f"Wrote {STOICH_PNG}")


if __name__ == "__main__":  # only run main() when executed as a script, not when imported
    main()
