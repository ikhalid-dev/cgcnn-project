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
# make_audit_figures.py (Parts 8-9) reads results/cgcnn/95_final_all_metrics.csv
# and dft/final_shortlist_15/, so re-run this script after either changes.
# make_hidden_features.py (slides 12b/12c) is NOT run here: it needs torch, which
# lives in infer_env, and the trained checkpoint. Run it by hand when the K model
# changes:
#     ~/miniconda3/envs/infer_env/bin/python make_hidden_features.py
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/mac/miniconda3/envs/ml_env/bin/python

echo "Regenerating this deck's figures..."
"$PY" make_alignn_figures.py
"$PY" make_audit_figures.py

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
