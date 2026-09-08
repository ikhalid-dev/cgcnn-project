#!/usr/bin/env bash
# Regenerate this deck's own figures, then build the PDF.
#
#     ./build_alignn.sh
#
# xelatex, not pdflatex - theme.tex loads system fonts through fontspec.
#
# Uses ml_env's python explicitly: the system python3 has neither pymatgen nor
# the plotting stack, and `python3 make_figures.py` fails there on a missing
# networkx. Only make_alignn_figures.py is run here - the CGCNN deck's figures
# are already in figures/ and are that deck's to regenerate.
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/mac/miniconda3/envs/ml_env/bin/python

echo "Regenerating this deck's figures..."
"$PY" make_alignn_figures.py && "$PY" make_pink_figures.py

echo "Compiling (pass 1/2)..."
xelatex -interaction=nonstopmode -halt-on-error PINK_REPRODUCTION_presentation.tex > build_alignn.log 2>&1 \
  || { echo "FAILED - see presentation/build_alignn.log"; tail -40 build_alignn.log; exit 1; }
echo "Compiling (pass 2/2)..."
xelatex -interaction=nonstopmode -halt-on-error PINK_REPRODUCTION_presentation.tex >> build_alignn.log 2>&1 \
  || { echo "FAILED - see presentation/build_alignn.log"; tail -40 build_alignn.log; exit 1; }

rm -f PINK_REPRODUCTION_presentation.aux PINK_REPRODUCTION_presentation.log \
      PINK_REPRODUCTION_presentation.nav PINK_REPRODUCTION_presentation.out \
      PINK_REPRODUCTION_presentation.snm PINK_REPRODUCTION_presentation.toc \
      PINK_REPRODUCTION_presentation.vrb build_alignn.log

echo "Built presentation/PINK_REPRODUCTION_presentation.pdf"
