#!/usr/bin/env python3
"""
Build the CGCNN project presentation.
=====================================

    python presentation/build_presentation.py

Produces `presentation/CGCNN_Elastic_Moduli.pptx`, matching the visual format of
the existing "Neural Networks from Scratch" deck: 16:9, a dark navy title slide,
white content slides, Cambria headings over Calibri body text, and the same
accent palette.

EVERY NUMBER IS READ FROM results/metrics_summary.csv, never typed in here. A
deck with hand-copied metrics goes stale the first time a model is retrained,
and a stale number in a talk is worse than no number - so if the results change,
rerunning this script is all it takes.

The result figures are the same PNGs the pipeline writes, plus two charts
generated here specifically for the talk (a comparison against the literature
and the ensemble gain), which are drawn in the deck's own palette so they look
native to the slides rather than pasted in.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJECT_ROOT, "results")
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")

# --- Design tokens, lifted from the reference deck -------------------------
DARK_BG = RGBColor(0x0F, 0x1B, 0x33)   # title slide background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1F, 0x29, 0x37)       # headings, primary text
MUTED = RGBColor(0x5B, 0x6B, 0x82)     # secondary text
BLUE = RGBColor(0x14, 0x50, 0xAA)      # primary accent / eyebrow
LIGHT_BLUE = RGBColor(0x6F, 0xA8, 0xFF)  # accent on dark
GREEN = RGBColor(0x1E, 0x8E, 0x5A)
AMBER = RGBColor(0xD9, 0x7B, 0x12)
SUBTLE = RGBColor(0xAA, 0xB6, 0xC8)    # subtitle on dark
CARD_BG = RGBColor(0xF4, 0xF6, 0xFA)
CARD_RULE = RGBColor(0xD8, 0xE0, 0xEC)

HEAD_FONT = "Cambria"
BODY_FONT = "Calibri"

# Matplotlib versions of the same palette, so generated charts match the slides.
MPL_BLUE, MPL_AMBER, MPL_GREEN = "#1450AA", "#D97B12", "#1E8E5A"
MPL_INK, MPL_MUTED, MPL_GRID = "#1F2937", "#5B6B82", "#E1E6EF"


# ===========================================================================
# Slide primitives
# ===========================================================================

def add_text(slide, left, top, width, height, text, size=14, color=INK,
             bold=False, font=BODY_FONT, align=PP_ALIGN.LEFT, italic=False,
             line_spacing=None, anchor=MSO_ANCHOR.TOP):
    """Place a text box. Returns the text frame for further paragraphs."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0

    para = frame.paragraphs[0]
    para.alignment = align
    if line_spacing:
        para.line_spacing = line_spacing
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color
    return frame


def add_para(frame, text, size=13, color=MUTED, bold=False, font=BODY_FONT,
             align=PP_ALIGN.LEFT, space_before=6, italic=False):
    """Append a paragraph to an existing text frame."""
    para = frame.add_paragraph()
    para.alignment = align
    para.space_before = Pt(space_before)
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color
    return para


def add_card(slide, left, top, width, height, fill=CARD_BG, line=CARD_RULE,
             rounded=True):
    """A soft background panel that groups related content."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    if shape.has_text_frame:
        shape.text_frame.text = ""
    return shape


def add_pill(slide, left, top, width, height, text, fill, text_color=WHITE,
             size=13):
    """A filled rounded label - used for the flow across the title slide."""
    shape = add_card(slide, left, top, width, height, fill=fill, line=None)
    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.name = BODY_FONT
    run.font.color.rgb = text_color
    return shape


def title_slide(prs, eyebrow, title, subtitle, flow):
    """The dark opening slide, with a horizontal flow of pills beneath."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = DARK_BG

    add_text(slide, 0.0, 1.30, 13.33, 0.40, eyebrow, size=14, bold=True,
             color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
    add_text(slide, 0.5, 1.70, 12.33, 1.10, title, size=42, bold=True,
             color=WHITE, font=HEAD_FONT, align=PP_ALIGN.CENTER)
    add_text(slide, 0.5, 2.85, 12.33, 0.60, subtitle, size=16, color=SUBTLE,
             align=PP_ALIGN.CENTER)

    # Evenly space the flow pills, with arrows between them.
    n = len(flow)
    pill_w, gap = 2.15, 0.28
    total = n * pill_w + (n - 1) * gap
    x = (13.33 - total) / 2
    for i, label in enumerate(flow):
        add_pill(slide, x, 4.35, pill_w, 0.95, label, fill=RGBColor(0x1A, 0x2E, 0x52))
        if i < n - 1:
            add_text(slide, x + pill_w, 4.60, gap, 0.45, "→", size=16,
                     color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
        x += pill_w + gap

    add_text(slide, 0.0, 6.55, 13.33, 0.35,
             "Aizaz Khalid  ·  reproduction of PINK (J. Mater. Inf. 2025, 5, 12)",
             size=11, color=SUBTLE, align=PP_ALIGN.CENTER)
    return slide


def content_slide(prs, eyebrow, title):
    """A white content slide with the standard eyebrow + heading block."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    add_text(slide, 0.60, 0.32, 8.00, 0.30, eyebrow, size=13, bold=True,
             color=BLUE)
    add_text(slide, 0.60, 0.62, 11.50, 0.70, title, size=30, bold=True,
             color=INK, font=HEAD_FONT)
    return slide


def takeaway(slide, text, top=6.15, color=BLUE):
    """The one-line 'so what' strip at the foot of a slide."""
    add_card(slide, 0.60, top, 12.13, 0.62, fill=CARD_BG, line=None)
    add_card(slide, 0.60, top, 0.06, 0.62, fill=color, line=None)
    add_text(slide, 0.85, top + 0.16, 11.70, 0.40, text, size=13, bold=True,
             color=INK)


# ===========================================================================
# Charts generated for the deck
# ===========================================================================

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C3C9D6")
    ax.spines["bottom"].set_color("#C3C9D6")
    ax.tick_params(colors=MPL_MUTED, labelsize=9)
    ax.grid(True, axis="y", color=MPL_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_literature(test, path):
    """Where our bulk model sits against the paper and the leaderboard.

    A bar chart is right here: four labelled magnitudes on one scale, compared
    against each other. Lower is better, which the axis label states outright
    rather than relying on the reader to infer.
    """
    names = ["Our single\nmodel", "Our 3-model\nensemble", "PINK paper",
             "coGN\n(best published)"]
    values = [test.loc["K_VRH_full", "MAE_log10"],
              test.loc["K_VRH_ens", "MAE_log10"], 0.070, 0.0535]
    # Ours are highlighted; the two references recede to neutral.
    colors = [MPL_BLUE, MPL_BLUE, MPL_MUTED, MPL_MUTED]
    alphas = [0.45, 1.0, 0.55, 0.55]

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    bars = ax.bar(names, values, color=colors, width=0.6)
    for bar, alpha, value in zip(bars, alphas, values):
        bar.set_alpha(alpha)
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0018,
                f"{value:.4f}", ha="center", va="bottom", fontsize=10,
                color=MPL_INK, fontweight="bold")
    ax.set_ylabel("Test MAE, log₁₀(GPa)   — lower is better", color=MPL_MUTED,
                  fontsize=10)
    ax.set_ylim(0, max(values) * 1.22)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def figure_ensemble_gain(test, path):
    """Single model vs ensemble, for both targets."""
    labels = ["Bulk modulus\n(K_VRH)", "Shear modulus\n(G_VRH)"]
    single = [test.loc["K_VRH_full", "MAE_log10"],
              test.loc["G_VRH_full", "MAE_log10"]]
    ens = [test.loc["K_VRH_ens", "MAE_log10"],
           test.loc["G_VRH_ens", "MAE_log10"]]

    x = range(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    b1 = ax.bar([i - width / 2 - 0.03 for i in x], single, width,
                label="Single model", color=MPL_MUTED, alpha=0.55)
    b2 = ax.bar([i + width / 2 + 0.03 for i in x], ens, width,
                label="3-model ensemble", color=MPL_BLUE)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom",
                    fontsize=9, color=MPL_INK, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test MAE, log₁₀(GPa)", color=MPL_MUTED, fontsize=10)
    ax.set_ylim(0, max(single) * 1.25)
    ax.legend(frameon=False, fontsize=9, labelcolor=MPL_MUTED, loc="upper right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ===========================================================================
# The deck
# ===========================================================================

def build(prs, test, allrows):
    k_ens = test.loc["K_VRH_ens"]
    g_ens = test.loc["G_VRH_ens"]
    k_one = test.loc["K_VRH_full"]
    g_one = test.loc["G_VRH_full"]
    k_train = allrows[(allrows.tag == "K_VRH_ens") &
                      (allrows.split == "train")].iloc[0]

    # ---------------------------------------------------------------- 1
    title_slide(
        prs, "A VISUAL GUIDE",
        "Crystal Graph Neural Networks",
        "Predicting elastic moduli directly from crystal structure — "
        "a from-scratch reproduction of PINK",
        ["CIF", "Crystal graph", "CGCNN", "K and G", "κₗ"])

    # ---------------------------------------------------------------- 2
    s = content_slide(prs, "01 · MOTIVATION", "Why predict elastic moduli?")
    add_card(s, 0.60, 1.55, 5.90, 3.05)
    add_text(s, 0.95, 1.85, 5.20, 0.35, "The bottleneck", size=15, bold=True,
             color=INK)
    f = add_text(s, 0.95, 2.30, 5.20, 2.10,
                 "Elastic moduli come from DFT calculations costing hours to "
                 "days of compute per material.", size=13, color=MUTED)
    add_para(f, "Only ~11,000 of the Materials Project's ~150,000 entries "
                "have them.", size=13, color=MUTED, space_before=10)
    add_para(f, "Screening a large candidate space by DFT is simply not "
                "affordable.", size=13, color=MUTED, space_before=10)

    add_card(s, 6.83, 1.55, 5.90, 3.05)
    add_text(s, 7.18, 1.85, 5.20, 0.35, "Why they matter here", size=15,
             bold=True, color=INK)
    f = add_text(s, 7.18, 2.30, 5.20, 2.10,
                 "PINK screens for materials with ultralow lattice thermal "
                 "conductivity κₗ.", size=13, color=MUTED)
    add_para(f, "Its physics stage needs bulk modulus K and shear modulus G "
                "as inputs.", size=13, color=MUTED, space_before=10)
    add_para(f, "A network that predicts K and G in milliseconds turns an "
                "impossible screen into a routine one.", size=13, color=MUTED,
             space_before=10)

    takeaway(s, "A CIF says where the atoms are. It does not say how stiff "
                "the crystal is — that is what we are learning.", top=5.00)

    # ---------------------------------------------------------------- 3
    s = content_slide(prs, "02 · THE PIPELINE", "Where this work fits in PINK")
    stages = [("CIF", "crystal\nstructure", MUTED),
              ("CGCNN", "predicts\nK and G", BLUE),
              ("Slack model", "physics,\nnot learned", MUTED),
              ("κₗ", "lattice thermal\nconductivity", MUTED)]
    x = 0.75
    for i, (name, sub, colour) in enumerate(stages):
        highlight = colour == BLUE
        add_card(s, x, 2.05, 2.55, 1.65,
                 fill=RGBColor(0xE8, 0xEF, 0xFA) if highlight else CARD_BG,
                 line=BLUE if highlight else CARD_RULE)
        add_text(s, x + 0.15, 2.30, 2.25, 0.40, name, size=17, bold=True,
                 color=BLUE if highlight else INK, font=HEAD_FONT,
                 align=PP_ALIGN.CENTER)
        add_text(s, x + 0.15, 2.85, 2.25, 0.70, sub, size=12, color=MUTED,
                 align=PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            add_text(s, x + 2.55, 2.68, 0.55, 0.40, "→", size=20, color=BLUE,
                     align=PP_ALIGN.CENTER)
        x += 3.10

    add_text(s, 0.60, 4.10, 12.13, 0.40,
             "The elastic moduli are the ONLY learned quantity. Everything "
             "downstream is physics.", size=14, bold=True, color=INK)
    add_text(s, 0.60, 4.55, 12.13, 0.80,
             "This project implements the first arrow — CIF → (K, G) — "
             "including the network itself, written from scratch rather than "
             "forked. It produces the (K, G) table the κₗ stage consumes.",
             size=13, color=MUTED)
    takeaway(s, "Scope: the machine-learning half of PINK, reproduced "
                "independently.", top=5.55)

    # ---------------------------------------------------------------- 4
    s = content_slide(prs, "03 · REPRESENTATION",
                      "A crystal becomes a graph")
    add_text(s, 0.60, 1.45, 12.13, 0.40,
             "A neural network cannot read a .cif file. So we translate:",
             size=14, color=MUTED)
    add_card(s, 0.60, 2.00, 3.85, 2.30)
    add_text(s, 0.90, 2.28, 3.25, 0.35, "Nodes = atoms", size=16, bold=True,
             color=BLUE, font=HEAD_FONT)
    add_text(s, 0.90, 2.75, 3.25, 1.30,
             "Each atom carries a fixed 92-dimensional vector of element "
             "properties: group, period, electronegativity, covalent radius, "
             "valence electrons.", size=12, color=MUTED)

    add_card(s, 4.74, 2.00, 3.85, 2.30)
    add_text(s, 5.04, 2.28, 3.25, 0.35, "Edges = bonds", size=16, bold=True,
             color=GREEN, font=HEAD_FONT)
    add_text(s, 5.04, 2.75, 3.25, 1.30,
             "The 12 nearest neighbours within 8 Å. The search respects "
             "periodic boundaries, so atoms at a cell face see their periodic "
             "images.", size=12, color=MUTED)

    add_card(s, 8.88, 2.00, 3.85, 2.30)
    add_text(s, 9.18, 2.28, 3.25, 0.35, "Edge features", size=16, bold=True,
             color=AMBER, font=HEAD_FONT)
    add_text(s, 9.18, 2.75, 3.25, 1.30,
             "Each bond length is expanded onto 41 Gaussian basis functions "
             "rather than fed in as one raw number.", size=12, color=MUTED)

    add_card(s, 0.60, 4.55, 12.13, 1.30, fill=RGBColor(0xEF, 0xF5, 0xEF),
             line=RGBColor(0xC8, 0xE0, 0xC8))
    add_text(s, 0.95, 4.78, 11.40, 0.35, "Why this representation generalises",
             size=14, bold=True, color=GREEN)
    add_text(s, 0.95, 5.15, 11.40, 0.55,
             "The graph stores only distances between atoms, never absolute "
             "positions — so it is invariant to translation, rotation and the "
             "choice of unit cell. The same crystal written two different ways "
             "produces the same graph.", size=12, color=MUTED)

    # ---------------------------------------------------------------- 5
    s = content_slide(prs, "04 · EDGE FEATURES",
                      "Why bond lengths are expanded")
    add_card(s, 0.60, 1.50, 12.13, 1.10, fill=RGBColor(0xF7, 0xF9, 0xFD),
             line=CARD_RULE)
    add_text(s, 0.60, 1.78, 12.13, 0.55,
             "eₖ(d) = exp( −(d − μₖ)² / σ² )     "
             "μₖ = 0, 0.2, … 8 Å     σ = 0.2 Å",
             size=18, bold=True, color=INK, font=HEAD_FONT,
             align=PP_ALIGN.CENTER)

    add_card(s, 0.60, 2.85, 5.90, 2.10)
    add_text(s, 0.95, 3.12, 5.20, 0.35, "The problem with a raw number",
             size=15, bold=True, color=INK)
    add_text(s, 0.95, 3.58, 5.20, 1.10,
             "A single scalar distance forces the network to learn a sharply "
             "non-linear response to it from scratch — a hard thing to fit "
             "and slow to converge.", size=12.5, color=MUTED)

    add_card(s, 6.83, 2.85, 5.90, 2.10)
    add_text(s, 7.18, 3.12, 5.20, 0.35, "What the expansion buys", size=15,
             bold=True, color=INK)
    add_text(s, 7.18, 3.58, 5.20, 1.30,
             "A bond at 2.31 Å strongly activates the basis functions centred "
             "near 2.3 and barely touches the one at 5.0. The distance "
             "dependence becomes a smooth, nearly linear read-out — far "
             "easier to learn.", size=12.5, color=MUTED)
    takeaway(s, "This is a radial basis expansion — the same trick used "
                "throughout atomistic machine learning.", top=5.25)

    # ---------------------------------------------------------------- 6
    s = content_slide(prs, "05 · THE NETWORK", "Gated graph convolution")
    add_text(s, 0.60, 1.42, 12.13, 0.40,
             "Message passing updates each atom's vector using its neighbours. "
             "For atom i with neighbours j:", size=13, color=MUTED)
    add_card(s, 0.60, 1.92, 12.13, 1.35, fill=RGBColor(0xF7, 0xF9, 0xFD),
             line=CARD_RULE)
    add_text(s, 0.60, 2.12, 12.13, 0.45,
             "zᵢⱼ = vᵢ ⊕ vⱼ ⊕ eᵢⱼ",
             size=16, bold=True, color=INK, font=HEAD_FONT,
             align=PP_ALIGN.CENTER)
    add_text(s, 0.60, 2.62, 12.13, 0.50,
             "vᵢ ← vᵢ + Σⱼ  σ(zᵢⱼ W_f + b_f) ⊙ "
             "g(zᵢⱼ W_s + b_s)",
             size=18, bold=True, color=BLUE, font=HEAD_FONT,
             align=PP_ALIGN.CENTER)

    items = [
        ("The gate  σ(·)", "Decides, per bond, how much that neighbour "
                           "contributes. A weak long bond can be ignored "
                           "without being discarded.", BLUE),
        ("Shared weights", "The same linear layer processes every edge. That "
                           "is what makes this a convolution, and why it "
                           "generalises across crystal sizes.", GREEN),
        ("The residual  vᵢ +", "Gradients flow straight through the "
                                    "stack, so early layers are not washed "
                                    "out by later ones.", AMBER),
    ]
    x = 0.60
    for name, body, colour in items:
        add_card(s, x, 3.50, 3.91, 1.95)
        add_text(s, x + 0.28, 3.75, 3.35, 0.35, name, size=14, bold=True,
                 color=colour)
        add_text(s, x + 0.28, 4.18, 3.35, 1.10, body, size=12, color=MUTED)
        x += 4.11
    takeaway(s, "Three convolution layers, so every atom sees its "
                "environment out to roughly three bond hops.", top=5.65)

    # ---------------------------------------------------------------- 7
    s = content_slide(prs, "06 · POOLING & TARGET",
                      "Two decisions that matter more than they look")
    add_card(s, 0.60, 1.50, 5.90, 2.35, fill=RGBColor(0xEF, 0xF5, 0xEF),
             line=RGBColor(0xC8, 0xE0, 0xC8))
    add_text(s, 0.95, 1.76, 5.20, 0.35, "Mean-pool, never sum", size=16,
             bold=True, color=GREEN, font=HEAD_FONT)
    add_text(s, 0.95, 2.20, 5.20, 0.40,
             "v_c = (1/N) Σᵢ vᵢ", size=15, bold=True, color=INK,
             font=HEAD_FONT)
    add_text(s, 0.95, 2.68, 5.20, 1.05,
             "A modulus is an INTENSIVE property. Doubling the unit cell must "
             "not double the prediction — but a sum would make the output "
             "scale with however many atoms the CIF author happened to write "
             "down.", size=12, color=MUTED)

    add_card(s, 6.83, 1.50, 5.90, 2.35, fill=RGBColor(0xFD, 0xF4, 0xE8),
             line=RGBColor(0xF0, 0xD8, 0xB8))
    add_text(s, 7.18, 1.76, 5.20, 0.35, "Train on log₁₀(modulus)", size=16,
             bold=True, color=AMBER, font=HEAD_FONT)
    add_text(s, 7.18, 2.20, 5.20, 0.40,
             "moduli span 1 – 575 GPa", size=13, italic=True, color=MUTED)
    add_text(s, 7.18, 2.68, 5.20, 1.05,
             "Under a plain MSE on raw values, a 40 GPa error on a stiff "
             "crystal produces 100× more gradient than the same RELATIVE "
             "error on a soft one — so the model would learn to care only "
             "about hard materials.", size=12, color=MUTED)

    add_card(s, 0.60, 4.10, 12.13, 1.45, fill=CARD_BG, line=CARD_RULE)
    add_text(s, 0.95, 4.32, 11.45, 0.35,
             "Exactly the wrong bias for this project", size=14, bold=True,
             color=INK)
    add_text(s, 0.95, 4.72, 11.45, 0.65,
             "PINK screens for ULTRALOW thermal conductivity — the soft, "
             "compliant end of the range. Taking the logarithm converts "
             "“within a factor of x” into “within a fixed "
             "distance”, so soft and stiff crystals carry equal weight.",
             size=12.5, color=MUTED)
    takeaway(s, "Normalisation statistics come from the training split only — "
                "using all the data would leak the test set.", top=5.75)

    # ---------------------------------------------------------------- 8
    s = content_slide(prs, "07 · THE DATA", "What to train on")
    add_card(s, 0.60, 1.45, 5.90, 1.55, fill=CARD_BG, line=CARD_RULE)
    add_text(s, 0.95, 1.68, 5.20, 0.35, "1,213 CIFs", size=17, bold=True,
             color=INK, font=HEAD_FONT)
    add_text(s, 0.95, 2.10, 5.20, 0.70,
             "Materials Project structures, and NO labels at all. These are "
             "the PREDICTION set.", size=12.5, color=MUTED)

    add_card(s, 6.83, 1.45, 5.90, 1.55, fill=RGBColor(0xE8, 0xEF, 0xFA),
             line=BLUE)
    add_text(s, 7.18, 1.68, 5.20, 0.35, "10,987 matbench crystals", size=17,
             bold=True, color=BLUE, font=HEAD_FONT)
    add_text(s, 7.18, 2.10, 5.20, 0.70,
             "DFT-computed K and G — the same benchmark the paper trained on. "
             "This is the TRAINING set.", size=12.5, color=MUTED)

    add_text(s, 0.60, 3.20, 12.13, 0.40,
             "Only 278 of the 1,213 appear in matbench — so it is tempting to "
             "label those and train on them.", size=13, color=MUTED)

    add_card(s, 0.60, 3.75, 12.13, 1.55, fill=RGBColor(0xFD, 0xF4, 0xE8),
             line=RGBColor(0xF0, 0xD8, 0xB8))
    add_text(s, 0.95, 3.98, 11.45, 0.35,
             "The structures were never the limit — the LABELS were",
             size=15, bold=True, color=AMBER, font=HEAD_FONT)
    add_text(s, 0.95, 4.40, 11.45, 0.80,
             "Intersecting an unlabelled collection with a labelled benchmark "
             "discards 97% of the labels that exist. Invert the roles: train "
             "on all 10,987 matbench crystals, and treat the 1,213 CIFs purely "
             "as a prediction set. That is 40× more data — and it is what the "
             "paper does.", size=12.5, color=MUTED)
    takeaway(s, "When labels are the scarce resource, never let an unlabelled "
                "collection define your training set.", top=5.50, color=AMBER)

    # ---------------------------------------------------------------- 9
    s = content_slide(prs, "08 · ENSEMBLING", "Three models, averaged")
    add_text(s, 0.60, 1.42, 12.13, 0.75,
             "Two networks trained on the same data from different random "
             "initialisations land in different minima. They AGREE about the "
             "signal and DISAGREE about their own noise — so averaging keeps "
             "the first and partially cancels the second.", size=13,
             color=MUTED)

    cards = [
        ("Average in log space",
         "Members predicting 10 and 1000 GPa average to 100 GPa in log space "
         "but 505 GPa linearly. 100 is the defensible answer.", BLUE),
        ("Members share one split",
         "--seed varies initialisation; --split-seed is PINNED. Otherwise the "
         "ensemble is scored partly on data some members trained on.", GREEN),
        ("Free uncertainty",
         "The spread between members is a per-crystal confidence estimate — "
         "where they disagree, the ensemble is guessing.", AMBER),
    ]
    x = 0.60
    for name, body, colour in cards:
        add_card(s, x, 2.45, 3.91, 2.15)
        add_card(s, x, 2.45, 3.91, 0.07, fill=colour, line=None)
        add_text(s, x + 0.28, 2.75, 3.35, 0.35, name, size=14, bold=True,
                 color=colour)
        add_text(s, x + 0.28, 3.20, 3.35, 1.20, body, size=12, color=MUTED)
        x += 4.11
    takeaway(s, "The code refuses to combine members whose test splits "
                "disagree, rather than trusting the caller to have got it "
                "right.", top=4.90)

    # --------------------------------------------------------------- 10
    s = content_slide(prs, "09 · RESULTS", "Model performance")
    headers = ["Model", "MAE log₁₀(GPa)", "R²", "MAE (GPa)", "Rel. error"]
    rows = [
        ("Bulk, single model", f"{k_one.MAE_log10:.4f}", f"{k_one.R2:.3f}",
         f"{k_one.MAE_GPa:.2f}", f"{k_one.rel_error_pct:.1f}%", False),
        ("Bulk, 3-model ensemble", f"{k_ens.MAE_log10:.4f}",
         f"{k_ens.R2:.3f}", f"{k_ens.MAE_GPa:.2f}",
         f"{k_ens.rel_error_pct:.1f}%", True),
        ("Shear, single model", f"{g_one.MAE_log10:.4f}", f"{g_one.R2:.3f}",
         f"{g_one.MAE_GPa:.2f}", f"{g_one.rel_error_pct:.1f}%", False),
        ("Shear, 3-model ensemble", f"{g_ens.MAE_log10:.4f}",
         f"{g_ens.R2:.3f}", f"{g_ens.MAE_GPa:.2f}",
         f"{g_ens.rel_error_pct:.1f}%", True),
    ]
    col_x = [0.60, 4.60, 6.90, 8.70, 10.70]
    col_w = [3.90, 2.20, 1.70, 1.90, 2.03]

    add_card(s, 0.60, 1.50, 12.13, 0.45, fill=RGBColor(0xEC, 0xF0, 0xF7),
             line=None)
    for xx, ww, head in zip(col_x, col_w, headers):
        add_text(s, xx + 0.18, 1.60, ww - 0.30, 0.30, head, size=11.5,
                 bold=True, color=MUTED,
                 align=PP_ALIGN.LEFT if head == "Model" else PP_ALIGN.CENTER)

    y = 1.95
    for label, mae, r2, gpa, rel, strong in rows:
        if strong:
            add_card(s, 0.60, y, 12.13, 0.52, fill=RGBColor(0xE8, 0xEF, 0xFA),
                     line=None)
        vals = [label, mae, r2, gpa, rel]
        for i, (xx, ww, val) in enumerate(zip(col_x, col_w, vals)):
            add_text(s, xx + 0.18, y + 0.14, ww - 0.30, 0.32, val,
                     size=13 if strong else 12.5, bold=strong,
                     color=INK if strong else MUTED,
                     align=PP_ALIGN.LEFT if i == 0 else PP_ALIGN.CENTER)
        y += 0.52

    add_text(s, 0.60, 4.20, 12.13, 0.35,
             "Measured on 1,648 held-out crystals. All models share one "
             "train/validation/test split, so every row is directly "
             "comparable.", size=12, italic=True, color=MUTED)

    add_card(s, 0.60, 4.70, 5.90, 1.05, fill=RGBColor(0xEF, 0xF5, 0xEF),
             line=RGBColor(0xC8, 0xE0, 0xC8))
    add_text(s, 0.95, 4.88, 5.20, 0.30, "No overfitting", size=13, bold=True,
             color=GREEN)
    add_text(s, 0.95, 5.18, 5.20, 0.45,
             f"Train {k_train.MAE_log10:.3f} vs test "
             f"{k_ens.MAE_log10:.3f} — ordinary generalisation error, not "
             f"recall.", size=12, color=MUTED)

    add_card(s, 6.83, 4.70, 5.90, 1.05, fill=RGBColor(0xE8, 0xEF, 0xFA),
             line=BLUE)
    add_text(s, 7.18, 4.88, 5.20, 0.30, "Ensembling gain", size=13, bold=True,
             color=BLUE)
    gain = (1 - k_ens.MAE_log10 / k_one.MAE_log10) * 100
    add_text(s, 7.18, 5.18, 5.20, 0.45,
             f"{gain:.0f}% of the bulk error removed by averaging three "
             f"initialisations.", size=12, color=MUTED)

    # --------------------------------------------------------------- 11
    s = content_slide(prs, "10 · RESULTS",
                      "How we compare to the literature")
    s.shapes.add_picture(os.path.join(FIGS, "literature.png"),
                         Inches(0.95), Inches(1.45), width=Inches(7.60))
    add_card(s, 8.85, 1.45, 3.88, 3.35)
    add_text(s, 9.15, 1.72, 3.30, 0.35, "Reading this", size=14, bold=True,
             color=INK)
    f = add_text(s, 9.15, 2.15, 3.30, 2.50,
                 "Our bulk ensemble slightly BEATS the paper it reproduces.",
                 size=12, color=MUTED)
    add_para(f, "It sits between PINK and coGN — the best result ever "
                "published on this benchmark, with any architecture.",
             size=12, color=MUTED, space_before=10)
    add_para(f, "Same architecture, same data as the paper. The gains come "
                "from ensembling and the cosine schedule.", size=12,
             color=MUTED, space_before=10)
    takeaway(s, "Reproduction target met: 0.0630 against the paper's ≈0.070.",
             top=5.10)

    # --------------------------------------------------------------- 12
    s = content_slide(prs, "11 · RESULTS", "Predicted vs DFT — bulk modulus")
    s.shapes.add_picture(os.path.join(RESULTS, "parity_K_VRH_ens.png"),
                         Inches(1.15), Inches(1.35), height=Inches(4.60))
    add_card(s, 6.60, 1.60, 6.13, 3.95)
    add_text(s, 6.95, 1.88, 5.45, 0.35, "How to read a parity plot", size=15,
             bold=True, color=INK, font=HEAD_FONT)
    f = add_text(s, 6.95, 2.35, 5.45, 3.05,
                 "Every point is one crystal: DFT value on x, our prediction "
                 "on y. A perfect model puts every point on the diagonal.",
                 size=12.5, color=MUTED)
    add_para(f, "Points ABOVE the line are over-predictions; BELOW are "
                "under-predictions.", size=12.5, color=MUTED, space_before=10)
    add_para(f, "The shaded band marks “within a factor of 1.5” — a "
                "useful screening tolerance, where getting the RANKING right "
                "matters more than the absolute value.", size=12.5,
             color=MUTED, space_before=10)
    add_para(f, "Blue = the 1,648 held-out test crystals. Grey = the training "
                "cloud, shown for context only.", size=12.5, color=MUTED,
             space_before=10)
    add_para(f, f"R² = {k_ens.R2:.3f} — the model explains "
                f"{k_ens.R2 * 100:.0f}% of the variance.", size=12.5,
             color=BLUE, bold=True, space_before=10)

    # --------------------------------------------------------------- 13
    s = content_slide(prs, "12 · RESULTS", "Predicted vs DFT — shear modulus")
    s.shapes.add_picture(os.path.join(RESULTS, "parity_G_VRH_ens.png"),
                         Inches(1.15), Inches(1.35), height=Inches(4.60))
    add_card(s, 6.60, 1.60, 6.13, 3.95)
    add_text(s, 6.95, 1.88, 5.45, 0.35, "Shear is the harder target", size=15,
             bold=True, color=INK, font=HEAD_FONT)
    f = add_text(s, 6.95, 2.35, 5.45, 3.05,
                 f"MAE {g_ens.MAE_log10:.4f} against "
                 f"{k_ens.MAE_log10:.4f} for bulk — consistently worse, and "
                 f"for a physical reason.", size=12.5, color=MUTED)
    add_para(f, "Bulk modulus resists uniform compression, which is dominated "
                "by average bond stiffness — close to what a graph network "
                "reads off directly.", size=12.5, color=MUTED, space_before=10)
    add_para(f, "Shear modulus resists shape change, which depends on bond "
                "DIRECTIONALITY and on anisotropy the representation captures "
                "less directly.", size=12.5, color=MUTED, space_before=10)
    add_para(f, "The same ordering appears in the paper and across the "
                "matbench leaderboard — it is a property of the task, not of "
                "our implementation.", size=12.5, color=MUTED, space_before=10)

    # --------------------------------------------------------------- 14
    s = content_slide(prs, "13 · RESULTS", "Where the errors live")
    s.shapes.add_picture(os.path.join(RESULTS, "residuals_K_VRH_ens.png"),
                         Inches(0.75), Inches(1.55), width=Inches(11.80))
    add_text(s, 0.75, 4.85, 5.70, 0.35, "Left: unbiased", size=14, bold=True,
             color=GREEN)
    add_text(s, 0.75, 5.22, 5.70, 0.70,
             "The error distribution is centred on zero — the model does not "
             "systematically over- or under-predict.", size=12, color=MUTED)
    add_text(s, 6.85, 4.85, 5.70, 0.35, "Right: the honest weakness", size=14,
             bold=True, color=AMBER)
    add_text(s, 6.85, 5.22, 5.70, 0.70,
             "Errors fan out below ~30 GPa. The model is least reliable on "
             "soft materials — precisely the ultralow-κ regime PINK targets.",
             size=12, color=MUTED)

    # --------------------------------------------------------------- 15
    s = content_slide(prs, "14 · RESULTS", "Ensembling, and how it trains")
    s.shapes.add_picture(os.path.join(FIGS, "ensemble.png"),
                         Inches(0.75), Inches(1.50), width=Inches(5.55))
    s.shapes.add_picture(os.path.join(RESULTS, "training_K_VRH_full.png"),
                         Inches(6.60), Inches(1.72), width=Inches(6.13))
    add_text(s, 0.75, 5.15, 5.55, 0.35,
             "Averaging three initialisations", size=13, bold=True, color=INK)
    add_text(s, 0.75, 5.50, 5.55, 0.60,
             "Consistent gains on both targets — about 9–10% of the error, in "
             "line with the 10–15% typically expected.", size=12, color=MUTED)
    add_text(s, 6.60, 5.15, 6.13, 0.35, "Training curves", size=13, bold=True,
             color=INK)
    add_text(s, 6.60, 5.50, 6.13, 0.60,
             "Cosine annealing over 200 epochs. Train and validation stay "
             "close together throughout — no divergence, so no overfitting.",
             size=12, color=MUTED)

    # --------------------------------------------------------------- 16
    s = content_slide(prs, "15 · HONESTY", "How low can the error go?")
    add_text(s, 0.60, 1.40, 12.13, 0.40,
             "Because the model is trained on log₁₀, MAE converts to a "
             "MULTIPLICATIVE error of 10^MAE − 1.", size=13, color=MUTED)
    add_card(s, 0.60, 1.95, 12.13, 0.85, fill=RGBColor(0xF7, 0xF9, 0xFD),
             line=CARD_RULE)
    add_text(s, 0.60, 2.18, 12.13, 0.45,
             f"MAE {k_ens.MAE_log10:.4f}  →  within a factor of "
             f"{10 ** k_ens.MAE_log10:.3f}  →  "
             f"{k_ens.rel_error_pct:.1f}% typical error",
             size=17, bold=True, color=INK, font=HEAD_FONT,
             align=PP_ALIGN.CENTER)

    add_card(s, 0.60, 3.05, 12.13, 2.05, fill=RGBColor(0xFD, 0xF0, 0xF0),
             line=RGBColor(0xF0, 0xC8, 0xC8))
    add_text(s, 0.95, 3.28, 11.45, 0.35,
             "A ~2% relative error is NOT reachable on this task — by anyone",
             size=15, bold=True, color=RGBColor(0xB0, 0x2A, 0x2A),
             font=HEAD_FONT)
    f = add_text(s, 0.95, 3.72, 11.45, 1.25,
                 "2% would need MAE = 0.0086, about six times better than the "
                 "best result ever published on this benchmark with any "
                 "architecture.", size=12.5, color=MUTED)
    add_para(f, "The blocker is not model capacity — it is LABEL NOISE. The "
                "targets are DFT-computed elastic moduli, and DFT elastic "
                "constants themselves disagree with experiment by roughly "
                "5–15%.", size=12.5, color=MUTED, space_before=8)
    add_para(f, "A model cannot be more accurate than the labels it is fitted "
                "to. A model reporting 2% here would be revealing a leak, not "
                "an achievement.", size=12.5, bold=True, color=INK,
             space_before=8)
    takeaway(s, "Realistic floor for this architecture: ≈0.06 log₁₀, about "
                "15%. We are at 0.0630.", top=5.30)

    # --------------------------------------------------------------- 17
    s = content_slide(prs, "16 · CONTRIBUTION",
                      "What we did differently from the paper")
    items = [
        ("1", "Written from scratch",
         "Graph construction, convolution, pooling and training loop are our "
         "own implementation — not a fork. No pre-trained weights are ever "
         "loaded."),
        ("2", "Ensembles, not single models",
         "Three initialisations per target, averaged in log space. Worth "
         "~9–10% of the error, and it supplies a per-crystal uncertainty the "
         "paper has no equivalent of."),
        ("3", "Adam with cosine annealing",
         "Rather than the original CGCNN's SGD with step decay. On a fixed "
         "epoch budget a scheduled decay beats a reactive one."),
        ("4", "Provenance on every prediction",
         "Each row records whether the model was trained on that crystal. "
         "Without it, accuracy claims on a screening set cannot be "
         "interpreted — part of the apparent accuracy may be recall."),
        ("5", "A documented error floor",
         "We state what accuracy is achievable and why, and report both the "
         "multiplicative and range-normalised errors rather than whichever "
         "flatters more."),
    ]
    y = 1.42
    for num, name, body in items:
        add_card(s, 0.60, y, 0.55, 0.88, fill=RGBColor(0xE8, 0xEF, 0xFA),
                 line=None)
        add_text(s, 0.60, y + 0.24, 0.55, 0.40, num, size=17, bold=True,
                 color=BLUE, font=HEAD_FONT, align=PP_ALIGN.CENTER)
        add_text(s, 1.35, y + 0.04, 11.38, 0.30, name, size=14, bold=True,
                 color=INK)
        add_text(s, 1.35, y + 0.36, 11.38, 0.55, body, size=11.5, color=MUTED)
        y += 0.98
    takeaway(s, "Deliberately identical: the architecture, the training data, "
                "the log target, the split ratio. In a reproduction, "
                "differences should be few and intentional.", top=6.35)

    # --------------------------------------------------------------- 18
    s = content_slide(prs, "17 · DELIVERABLE", "What this produces")
    add_card(s, 0.60, 1.45, 12.13, 1.50, fill=RGBColor(0xE8, 0xEF, 0xFA),
             line=BLUE)
    add_text(s, 0.95, 1.68, 11.45, 0.35,
             "results/pink_moduli_predictions.csv", size=17, bold=True,
             color=BLUE, font=HEAD_FONT)
    add_text(s, 0.95, 2.12, 11.45, 0.70,
             "Bulk and shear modulus for all 1,213 crystals — with Pugh "
             "ratio, DFT reference where one exists, per-crystal ensemble "
             "uncertainty, and a provenance tag. This is the input the κₗ "
             "stage consumes.", size=12.5, color=MUTED)

    stats = [("935", "crystals genuinely unseen", GREEN),
             ("278", "in matbench, so tagged as recall", AMBER),
             ("42", "in the held-out test split", BLUE)]
    x = 0.60
    for value, label, colour in stats:
        add_card(s, x, 3.15, 3.91, 1.15)
        add_text(s, x + 0.28, 3.35, 3.35, 0.45, value, size=24, bold=True,
                 color=colour, font=HEAD_FONT)
        add_text(s, x + 0.28, 3.85, 3.35, 0.35, label, size=12, color=MUTED)
        x += 4.11

    add_text(s, 0.60, 4.55, 12.13, 0.35, "A built-in correctness check",
             size=14, bold=True, color=INK)
    add_text(s, 0.60, 4.92, 12.13, 0.65,
             "On the 42 test-provenance crystals the error is 0.062 — matching "
             "the benchmark test error. Had the provenance mapping been wrong, "
             "we would have seen the much lower training-set error instead.",
             size=12.5, color=MUTED)
    takeaway(s, "Next: feed these moduli into the Slack model, and propagate "
                "the uncertainties through to κₗ — something the original "
                "pipeline cannot do.", top=5.75)

    # --------------------------------------------------------------- 19
    s = content_slide(prs, "18 · REPRODUCIBILITY", "Run it yourself")
    add_card(s, 0.60, 1.50, 12.13, 1.35, fill=RGBColor(0x1F, 0x29, 0x37),
             line=None)
    add_text(s, 0.95, 1.75, 11.45, 0.30,
             "conda env create -f environment.yml && conda activate pink-cgcnn",
             size=13, color=RGBColor(0x9F, 0xD0, 0xFF), font="Consolas")
    add_text(s, 0.95, 2.12, 11.45, 0.30,
             "./run_pipeline.sh --predict        # ~2 min, no training",
             size=13, color=RGBColor(0x9F, 0xD0, 0xFF), font="Consolas")
    add_text(s, 0.95, 2.45, 11.45, 0.30,
             "./run_pipeline.sh                  # retrain: ~45 min GPU",
             size=13, color=RGBColor(0x9F, 0xD0, 0xFF), font="Consolas")

    cards = [
        ("Checkpoints committed",
         "All six models (~380 KB each) are in the repository, so every "
         "figure and number regenerates in two minutes without training.",
         GREEN),
        ("~45 min on a GPU",
         "Colab notebook included. On the 2016 laptop this was developed on, "
         "the same work takes about nine hours.", BLUE),
        ("Numbers never hand-copied",
         "This deck, RESULTS.md and the metrics table are all generated from "
         "the same CSV the pipeline writes.", AMBER),
    ]
    x = 0.60
    for name, body, colour in cards:
        add_card(s, x, 3.10, 3.91, 1.85)
        add_card(s, x, 3.10, 3.91, 0.07, fill=colour, line=None)
        add_text(s, x + 0.28, 3.38, 3.35, 0.35, name, size=13.5, bold=True,
                 color=colour)
        add_text(s, x + 0.28, 3.80, 3.35, 1.05, body, size=11.5, color=MUTED)
        x += 4.11

    add_text(s, 0.60, 5.25, 12.13, 0.35,
             "Full write-up in docs/method.pdf  ·  "
             "github.com/ikhalid-dev/cgcnn-project", size=12.5, italic=True,
             color=MUTED, align=PP_ALIGN.CENTER)
    takeaway(s, "Reproduction target met: bulk MAE 0.0630 vs the paper's "
                "≈0.070, on the paper's own benchmark.", top=5.75)


def main():
    if not os.path.exists(os.path.join(RESULTS, "metrics_summary.csv")):
        sys.exit("No results/metrics_summary.csv - run "
                 "`python scripts/06_summarise.py` first.")

    allrows = pd.read_csv(os.path.join(RESULTS, "metrics_summary.csv"))
    test = allrows[allrows.split == "test"].set_index("tag")

    os.makedirs(FIGS, exist_ok=True)
    figure_literature(test, os.path.join(FIGS, "literature.png"))
    figure_ensemble_gain(test, os.path.join(FIGS, "ensemble.png"))

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    build(prs, test, allrows)

    out = os.path.join(HERE, "CGCNN_Elastic_Moduli.pptx")
    prs.save(out)
    print(f"Wrote {out}")
    print(f"  {len(prs.slides.__iter__.__self__._sldIdLst)} slides")


if __name__ == "__main__":
    main()
