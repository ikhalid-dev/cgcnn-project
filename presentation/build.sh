#!/usr/bin/env bash
# Regenerate figures/metrics, then build the presentation PDF.
#
#     ./build.sh
#
# Must use xelatex, not pdflatex - the theme loads system fonts (Georgia,
# Avenir Next) through fontspec, which pdflatex cannot use.
set -euo pipefail
cd "$(dirname "$0")"

echo "Regenerating figures and metrics.tex from results/metrics_summary.csv..."
PY=/Users/mac/miniconda3/envs/ml_env/bin/python
"$PY" make_figures.py \
  && "$PY" make_alignn_figures.py \
  && "$PY" make_pink_figures.py

echo "Compiling (pass 1/2)..."
xelatex -interaction=nonstopmode -halt-on-error CGCNN_presentation.tex > build.log 2>&1 \
  || { echo "FAILED - see presentation/build.log"; tail -40 build.log; exit 1; }
echo "Compiling (pass 2/2)..."
xelatex -interaction=nonstopmode -halt-on-error CGCNN_presentation.tex >> build.log 2>&1 \
  || { echo "FAILED - see presentation/build.log"; tail -40 build.log; exit 1; }

rm -f CGCNN_presentation.aux CGCNN_presentation.log CGCNN_presentation.nav \
      CGCNN_presentation.out CGCNN_presentation.snm CGCNN_presentation.toc \
      CGCNN_presentation.vrb build.log

echo "Built presentation/CGCNN_presentation.pdf"
