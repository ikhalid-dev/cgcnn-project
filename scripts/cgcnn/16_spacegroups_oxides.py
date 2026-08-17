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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "results", "cgcnn")
GNOME_SUMMARY = os.path.join(PROJECT_ROOT, "gnome_data", "stable_materials_summary.csv")
OXIDE_CSV = os.path.join(RESULTS, "gnome_oxide_candidates.csv")
OUT_CSV = os.path.join(RESULTS, "gnome_oxide_spacegroups.csv")
OUT_PNG = os.path.join(RESULTS, "gnome_oxide_spacegroups.png")
STOICH_CSV = os.path.join(RESULTS, "gnome_oxide_by_stoichiometry.csv")
STOICH_PNG = os.path.join(RESULTS, "gnome_oxide_by_stoichiometry.png")
BEST_BY_CS_CSV = os.path.join(RESULTS, "gnome_oxide_best_by_crystal_system.csv")
BEST_BY_SG_CSV = os.path.join(RESULTS, "gnome_oxide_best_by_spacegroup.csv")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# International Tables space-group ranges, used to mark crystal-system
# windows on the histogram (boundaries per the standard 230-group numbering).
CRYSTAL_SYSTEM_RANGES = [
    ("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142), ("trigonal", 143, 167), ("hexagonal", 168, 194),
    ("cubic", 195, 230),
]


def load_space_groups():
    """The GNoME summary CSV's own space-group columns, deduped to one row
    per MaterialId (a small number of rows share no MaterialId at all - see
    module docstring - those are dropped here since they cannot be joined
    against anyway)."""
    sg = pd.read_csv(GNOME_SUMMARY,
                     usecols=["MaterialId", "Space Group", "Space Group Number", "Crystal System"])
    sg = sg.dropna(subset=["MaterialId"]).drop_duplicates("MaterialId")
    return sg.rename(columns={
        "Space Group": "space_group",
        "Space Group Number": "space_group_number",
        "Crystal System": "crystal_system",
    })


def sort_by_crystal_structure(df):
    """Crystal system in symmetry order (triclinic -> cubic, per
    CRYSTAL_SYSTEM_RANGES - NOT alphabetical), then space group number, then
    predicted Kappa_cal ascending as a tiebreaker within each space group.
    Rows with no matched space-group row (see module docstring) sort to the
    end, since pandas puts NaN categories last by default."""
    system_order = [name for name, _, _ in CRYSTAL_SYSTEM_RANGES]
    out = df.copy()
    out["crystal_system"] = pd.Categorical(out["crystal_system"], categories=system_order, ordered=True)
    return out.sort_values(
        ["crystal_system", "space_group_number", "Kappa_cal (W m-1 K-1)"]
    ).reset_index(drop=True)


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
    valid = df.dropna(subset=[group_col])
    best_idx = valid.groupby(group_col, observed=True)[rank_col].idxmin()
    return valid.loc[best_idx].sort_values(rank_col).reset_index(drop=True)


def plot_histogram(df, path, title="Space groups of the GNoME oxide screen",
                   all_label="All oxide candidates", low_label="Low-κ oxide candidates"):
    """Space-group-number histogram, all candidates vs. the low-kappa subset
    overlaid, with the seven crystal-system windows marked along the top
    axis. title/all_label/low_label are parametrised so 17_spacegroups_general.py
    can reuse this unchanged for the full (non-oxide-filtered) screen."""
    all_sgn = df["space_group_number"].dropna()
    low_sgn = df.loc[df["is_low_kappa_candidate"], "space_group_number"].dropna()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bins = np.arange(1, 232) - 0.5
    ax.hist(all_sgn, bins=bins, color=BLUE, alpha=0.55,
           label=f"{all_label} (n={len(all_sgn)})", edgecolor="none")
    ax.hist(low_sgn, bins=bins, color=ORANGE, alpha=0.85,
           label=f"{low_label} (n={len(low_sgn)})", edgecolor="none")

    for name, lo, hi in CRYSTAL_SYSTEM_RANGES[1:]:
        ax.axvline(lo - 0.5, color=GRID, linewidth=0.8, linestyle=":")

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    mids = [(lo + hi) / 2 for _, lo, hi in CRYSTAL_SYSTEM_RANGES]
    ax2.set_xticks(mids)
    ax2.set_xticklabels([name for name, _, _ in CRYSTAL_SYSTEM_RANGES],
                        fontsize=8, color=MUTED, rotation=20, ha="left")
    ax2.tick_params(length=0)
    ax2.spines["top"].set_visible(False)

    ax.set_xlabel("Space group number (International Tables, 1-230)")
    ax.set_ylabel("Number of candidates")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stoichiometry_breakdown(df):
    """Long-format table: one row per (stoichiometry_pattern, crystal_system)
    pair, with the total count and the low-kappa-candidate count. Patterns
    ordered by descending total size, crystal systems by the fixed
    triclinic->cubic symmetry order (not alphabetical) so both the console
    print and the CSV read in a consistent, physically meaningful order."""
    system_order = [name for name, _, _ in CRYSTAL_SYSTEM_RANGES]
    pattern_order = df["stoichiometry_pattern"].value_counts().index.tolist()

    rows = []
    for pattern in pattern_order:
        sub = df[df.stoichiometry_pattern == pattern]
        for system in system_order:
            in_system = sub[sub.crystal_system == system]
            rows.append({
                "stoichiometry_pattern": pattern,
                "crystal_system": system,
                "n_total": len(in_system),
                "n_low_kappa": int(in_system["is_low_kappa_candidate"].sum()),
            })
    return pd.DataFrame(rows), pattern_order, system_order


def plot_by_stoichiometry(df, pattern_order, system_order, path):
    """Small multiples, one bar-chart panel per stoichiometry pattern - a
    single shared axis would crush the 8-row AB2O4-type pattern to invisible
    next to the >4,000-row complex-oxide patterns, so each panel gets its own
    y-scale and only the crystal-system shape is being compared across
    panels, not the absolute magnitude."""
    ncols = 3
    nrows = -(-len(pattern_order) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(system_order))

    for i, pattern in enumerate(pattern_order):
        ax = axes[i // ncols][i % ncols]
        sub = df[df.stoichiometry_pattern == pattern]
        totals = sub["crystal_system"].value_counts().reindex(system_order, fill_value=0)
        lows = sub.loc[sub.is_low_kappa_candidate, "crystal_system"] \
                  .value_counts().reindex(system_order, fill_value=0)
        ax.bar(x, totals.values, color=BLUE, alpha=0.6,
              label="All oxide candidates" if i == 0 else None)
        ax.bar(x, lows.values, color=ORANGE, alpha=0.9,
              label="Low-κ oxide candidates" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(system_order, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(f"{pattern}\n(n={len(sub)})", fontsize=9, color=INK)
        ax.tick_params(axis="y", labelsize=7.5)

    for j in range(len(pattern_order), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True,
              facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    fig.suptitle("Oxide screen: crystal systems by stoichiometry pattern", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    print("=== Space groups of the GNoME oxide screen ===\n")
    oxides = pd.read_csv(OXIDE_CSV)
    print(f"Loaded {len(oxides)} oxide candidates from {OXIDE_CSV}")

    sg = load_space_groups()
    merged = oxides.merge(sg, left_on="material_id", right_on="MaterialId", how="left") \
                   .drop(columns=["MaterialId"])
    missing = merged["space_group"].isna().sum()
    print(f"Joined against {GNOME_SUMMARY} on MaterialId")
    if missing:
        print(f"  {missing}/{len(merged)} oxide rows have no matching GNoME summary "
             f"row (upstream data gap, not this project's - see module docstring) "
             f"- space_group columns left blank for those")

    print("\nBy crystal system (all oxide candidates):")
    for system, count in merged["crystal_system"].value_counts().items():
        n_low = int(merged[(merged.crystal_system == system) &
                           merged.is_low_kappa_candidate].shape[0])
        print(f"  {system:14s} {count:6d} total, {n_low:6d} low-kappa candidates")

    sorted_df = sort_by_crystal_structure(merged)
    sorted_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(sorted_df)} rows, sorted by crystal system "
         f"-> space group number -> Kappa_cal ascending)")

    plot_histogram(merged, OUT_PNG)
    print(f"Wrote {OUT_PNG}")

    best_cs = best_by_group(merged, "crystal_system")
    best_cs.to_csv(BEST_BY_CS_CSV, index=False)
    print(f"\nWrote {BEST_BY_CS_CSV} ({len(best_cs)} rows - best oxide candidate per crystal system, "
         f"ranked by lowest Kappa_cal_p95, not the point estimate - see best_by_group() docstring)")
    print("Best oxide candidate per crystal system (lowest Kappa_cal_p95):")
    for _, row in best_cs.iterrows():
        flag = "cleared threshold" if row.is_low_kappa_candidate else "did NOT clear threshold"
        ratio = row.Kappa_cal_p95 / row.Kappa_cal_p05
        print(f"  {row.crystal_system:14s} {row.formula:16s} point={row['Kappa_cal (W m-1 K-1)']:.4f} "
             f"p95={row.Kappa_cal_p95:.4f} (p95/p05={ratio:.1f}x) (sg {row.space_group}, {flag})")

    best_sg = best_by_group(merged, "space_group")
    best_sg.to_csv(BEST_BY_SG_CSV, index=False)
    print(f"\nWrote {BEST_BY_SG_CSV} ({len(best_sg)} rows - best oxide candidate per space group, "
         f"ranked by lowest Kappa_cal_p95)")
    print("Top 10 overall (best oxide candidate per space group, ranked by Kappa_cal_p95):")
    for _, row in best_sg.head(10).iterrows():
        flag = "cleared threshold" if row.is_low_kappa_candidate else "did NOT clear threshold"
        ratio = row.Kappa_cal_p95 / row.Kappa_cal_p05
        print(f"  {row.formula:16s} point={row['Kappa_cal (W m-1 K-1)']:.4f} p95={row.Kappa_cal_p95:.4f} "
             f"(p95/p05={ratio:.1f}x) (sg {row.space_group}, {row.crystal_system}, {flag})")

    stoich_table, pattern_order, system_order = stoichiometry_breakdown(merged)
    print("\nBy stoichiometry pattern x crystal system:")
    for pattern in pattern_order:
        sub = stoich_table[stoich_table.stoichiometry_pattern == pattern]
        total = int(sub.n_total.sum())
        print(f"  {pattern} (n={total}):")
        for _, row in sub[sub.n_total > 0].sort_values("n_total", ascending=False).iterrows():
            print(f"      {row.crystal_system:14s} {row.n_total:6d} total, "
                 f"{row.n_low_kappa:6d} low-kappa candidates")

    stoich_table.to_csv(STOICH_CSV, index=False)
    print(f"\nWrote {STOICH_CSV} ({len(stoich_table)} rows)")

    plot_by_stoichiometry(merged, pattern_order, system_order, STOICH_PNG)
    print(f"Wrote {STOICH_PNG}")


if __name__ == "__main__":
    main()
