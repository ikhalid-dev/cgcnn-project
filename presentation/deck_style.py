#!/usr/bin/env python3
"""
deck_style.py - ONE shared look for every chart in the ALIGNN/PINK deck.

Why this file exists
--------------------
Before, every figure script picked its own colours, so "CGCNN" was blue on one
slide and green on the next. Now every script does

    import deck_style as ds          # "import X as Y" = load file X, call it Y here
    ds.apply()                       # set fonts and sizes once, for all charts
    ax.plot(x, y, color=ds.MODEL_COLOURS["ALIGNN"])

and a model has the same colour on every slide.

To change the look of the WHOLE deck, edit only the CONFIG block below and
re-run presentation/build_alignn.sh.
"""

# =============================================================================
#  CONFIG - every setting a student might want to change lives here
# =============================================================================
# A Python "dict" ({key: value, ...}) maps a name to a value.
# Look a value up with square brackets: MODEL_COLOURS["ALIGNN"] -> "#1E8E5A".

# One fixed colour per model. These hex codes are the SAME colours the slides
# use (theme.tex: SlideBlue, SlideGreen, SlideAmber), so chart and text agree.
MODEL_COLOURS = {
    "CGCNN ensemble":  "#1450AA",   # blue   (theme.tex SlideBlue)
    "CGCNN single":    "#6FA8FF",   # light blue - the same model family, one seed
    "ALIGNN":          "#1E8E5A",   # green  (theme.tex SlideGreen)
    "ALIGNN single":   "#7CC4A0",   # light green
    "tree":            "#D97B12",   # amber  (theme.tex SlideAmber) - descriptor/composition trees
    "direct ALIGNN":   "#7A3FA0",   # purple - structure -> kappa directly, no Slack chain
    "measured":        "#1F2937",   # near-black - experiment
    "PINK paper":      "#8A96A8",   # grey - the published paper's own numbers
}

# Colours that are not models
INK = "#1F2937"        # main text colour (theme.tex SlideInk)
MUTED = "#5B6B82"      # secondary text (theme.tex Muted)
GRID = "#E1E6EF"       # light grid lines
THRESHOLD = "#B02A2A"  # red - the kappa_L <= 1 screening line (theme.tex SlideRed)
BACKGROUND = "white"   # figure background; slides are white too

# Fonts: the slides use Avenir Next. The list is tried left to right, so if a
# machine does not have Avenir Next, matplotlib falls back to the next name.
FONT_FAMILY = ["Avenir Next", "Helvetica Neue", "Helvetica", "DejaVu Sans"]
FONT_SIZE = 11          # axis labels and legend
TICK_SIZE = 10          # numbers on the axes
TITLE_SIZE = 12         # panel titles

DPI = 200               # dots per inch of the saved PNG; 200 is sharp on a projector
# =============================================================================


import matplotlib            # the plotting library
matplotlib.use("Agg")        # "Agg" = draw to a file, never open a window
import matplotlib.pyplot as plt


def apply():
    """Set the shared fonts and sizes. Call once, before drawing anything."""
    # plt.rcParams is matplotlib's global settings dict; .update() changes
    # several keys at once.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "mathtext.fontset": "dejavusans",   # how $math$ in labels is drawn
    })


def tidy(ax, grid="y"):
    """Remove the top/right box lines and add a light grid.

    ax   = one panel of a figure (matplotlib calls a panel an "Axes")
    grid = "y", "x", "both", or None for no grid
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)      # spines = the 4 border lines
    if grid:
        ax.grid(color=GRID, lw=0.7, axis=grid)
    ax.set_axisbelow(True)                      # grid behind the data, not on top


def panel_title(ax, letter, text):
    """Left-aligned title like 'A  atoms per cell (matbench, n = 10,987)'."""
    ax.set_title(f"{letter}   {text}", loc="left", color=INK)


def save(fig, path):
    """Save at the shared DPI, trim white margins, close the figure."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)                               # free the memory
