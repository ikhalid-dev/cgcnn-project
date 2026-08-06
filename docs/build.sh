#!/usr/bin/env bash
# Rebuild method.pdf from method.tex.
#
# Run twice: the first pass writes the .aux that the table of contents and
# cross-references are read from, the second pass resolves them.
#
# Needs a LaTeX install with tikz, booktabs, hyperref, geometry, xcolor and
# lmodern. The document deliberately avoids `caption` and `enumitem`, which are
# absent from minimal TeX distributions - so a basic MacTeX/TeX Live install is
# enough and no package downloads are required.
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode method.tex >/dev/null
pdflatex -interaction=nonstopmode method.tex >/dev/null
rm -f method.aux method.log method.out method.toc
echo "Built docs/method.pdf"
