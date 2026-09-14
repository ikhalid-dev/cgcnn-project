"""
THE AUDIT: is every result reported, and is every reported number still true?

    python scripts/audit_results.py            # report
    python scripts/audit_results.py --strict   # exit 1 if anything fails

WHY
---
An audit on 2026-09-15 found the same two failures in this project and both of
its siblings:

    ORPHAN   a result computed, saved as a CSV, never folded into README.md
    DRIFT    a number written correctly, then the thing it described re-run

Here the orphans were step 18's ALIGNN ensembles - including a finding worth
keeping, that ensembling ALIGNN barely helps - and steps 57/58's rho
decomposition with its population control. All three sat in results/alignn/ for
weeks without reaching the document that is this project's deliverable.

Neither failure breaks anything, so no test catches them. This does.

THE ORPHAN RULE, AND WHY IT IS BY STEP
--------------------------------------
The first version of this check looked for FILENAMES and reported nearly every
artefact as an orphan, because a README cites results the way a document does -
"step 18", "scripts/alignn/12_train_alignn.py", "round 9" - not by CSV name. A
check that does not match how the document is written measures the check.

So artefacts are grouped by their leading step number and the question asked is
whether that STEP is mentioned anywhere. One step writes several files; what
matters is whether its result was reported.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOC = ROOT / "README.md"

# (claimed in README.md, csv relative to results/, (index col, row, value col))
CLAIMS = [
    (0.063015, "alignn/18_alignn_ensemble_scores.csv", ("K", "CGCNN 3-ens")),
    (0.053912, "alignn/18_alignn_ensemble_scores.csv", ("K", "ALIGNN single (existing)")),
    (0.078097, "alignn/18_alignn_ensemble_scores.csv", ("G", "CGCNN 3-ens")),
    (0.072503, "alignn/18_alignn_ensemble_scores.csv", ("G", "ALIGNN single (existing)")),
    (0.053361, "alignn/18_alignn_ensemble_scores.csv", ("K", "ALIGNN 3-ens")),
    (0.070755, "alignn/18_alignn_ensemble_scores.csv", ("G", "ALIGNN 3-ens")),
]


def check_drift(text):
    bad = 0
    for claimed, rel, (target, model) in CLAIMS:
        path = RESULTS / rel
        if not path.exists():
            print(f"  MISSING FILE   {rel}")
            bad += 1
            continue
        df = pd.read_csv(path)
        row = df[(df.target == target) & (df.model == model)]
        if not len(row):
            print(f"  MISSING ROW    {rel}: {target}/{model}")
            bad += 1
            continue
        actual = float(row.mae_log10.iloc[0])
        drifted = abs(actual - claimed) > 0.0006
        # the number must still be PRESENT in the README, at the precision the
        # README uses - a claim silently deleted is also a problem
        shown = any(f"{claimed:.{d}f}" in text for d in (3, 4))
        if drifted or not shown:
            print(f"  {'DRIFTED' if drifted else 'NOT IN README':<14} "
                  f"{target}/{model:<26} readme {claimed:.4f}  file {actual:.4f}")
            bad += 1
    if not bad:
        print(f"  all {len(CLAIMS)} tracked numbers match their source and "
              "appear in README.md")
    return bad


def check_orphans(text):
    steps = set()
    for pat in (r'step\s+(\d+)', r'round\s+(\d+)', r'/(\d+)_', r'fig(\d+)',
                r'\b(\d+)_[a-z]'):
        steps.update(int(m) for m in re.findall(pat, text, re.I))
    by_step, unnumbered = {}, []
    for f in sorted(RESULTS.rglob("*")):
        if f.suffix not in (".csv", ".png"):
            continue
        m = re.match(r'^(?:fig)?(\d+)_', f.name)
        if m:
            by_step.setdefault(int(m.group(1)), []).append(f.name)
        else:
            unnumbered.append(str(f.relative_to(RESULTS)))
    missing = {k: v for k, v in by_step.items() if k not in steps}
    print(f"  {sum(len(v) for v in by_step.values())} numbered artefacts from "
          f"{len(by_step)} steps; {len(steps)} steps cited in README.md")
    if missing:
        print(f"  {len(missing)} step(s) never mentioned:")
        for k in sorted(missing):
            print(f"      step {k}: {', '.join(missing[k][:4])}"
                  + (" ..." if len(missing[k]) > 4 else ""))
    else:
        print("  every numbered step that produced output is written up")
    print(f"  ({len(unnumbered)} artefacts have no step prefix and are not "
          "checked)")
    return len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    text = DOC.read_text()
    print("=" * 78)
    print("AUDIT")
    print("=" * 78)
    print("\n1. ORPHANS - results computed but never reported")
    print("-" * 78)
    a = check_orphans(text)
    print("\n2. DRIFT - numbers in README.md vs the files that produced them")
    print("-" * 78)
    b = check_drift(text)
    print("\n" + "=" * 78)
    print(f"{'PASS' if a + b == 0 else 'FAIL'}: {a + b} problem(s)")
    print("=" * 78)
    if args.strict and a + b:
        sys.exit(1)


if __name__ == "__main__":
    main()
