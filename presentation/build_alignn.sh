#!/usr/bin/env bash
# Regenerate this deck's own figures, then build the PDF.
#
#     ./build_alignn.sh
#
# xelatex, not pdflatex - theme.tex loads system fonts through fontspec.
#
# Uses ml_env's python explicitly: the system python3 has neither pymatgen nor
# the plotting stack, and `python3 make_figures.py` fails there on a missing
# networkx. Only this deck's scripts are run here - the CGCNN deck's figures
# are already in figures/ and are that deck's to regenerate.
#
# The three figure scripts, and what they feed:
#   make_alignn_figures.py  the two-graphs diagram (slide 7)
#   make_deck_figures.py    every model chart on slides 11-24, plus
#                           figures/deck_numbers.tex (the \D... number macros)
#   make_audit_figures.py   the audit and DFT-list slides (25-30); reads
#                           results/cgcnn/95_final_all_metrics.csv and
#                           dft/final_shortlist_15/, so re-run after either changes
# All charts share one look and one colour per model through deck_style.py.
# Slides that were cut live in removed_slides.tex, which is NOT compiled.
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/mac/miniconda3/envs/ml_env/bin/python

echo "Regenerating this deck's figures..."
"$PY" make_alignn_figures.py
OMP_NUM_THREADS=1 "$PY" make_deck_figures.py
OMP_NUM_THREADS=1 "$PY" make_audit_figures.py

echo "Compiling (pass 1/2)..."
xelatex -interaction=nonstopmode -halt-on-error ALIGNN_PINK_presentation.tex > build_alignn.log 2>&1 \
  || { echo "FAILED - see presentation/build_alignn.log"; tail -40 build_alignn.log; exit 1; }
echo "Compiling (pass 2/2)..."
xelatex -interaction=nonstopmode -halt-on-error ALIGNN_PINK_presentation.tex >> build_alignn.log 2>&1 \
  || { echo "FAILED - see presentation/build_alignn.log"; tail -40 build_alignn.log; exit 1; }

rm -f ALIGNN_PINK_presentation.aux ALIGNN_PINK_presentation.log \
      ALIGNN_PINK_presentation.nav ALIGNN_PINK_presentation.out \
      ALIGNN_PINK_presentation.snm ALIGNN_PINK_presentation.toc \
      ALIGNN_PINK_presentation.vrb build_alignn.log

echo "Built presentation/ALIGNN_PINK_presentation.pdf"
