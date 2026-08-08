#!/usr/bin/env python3
"""
Build the Google Colab notebook for training ALIGNN on a GPU.
===============================================================

    python colab/PINK_ALIGNN_colab.py        # writes colab/PINK_ALIGNN.ipynb

WHY THIS EXISTS
----------------
Phase 2 asks whether a more expressive architecture (ALIGNN adds a line graph
of bond angles on top of CGCNN's bond graph) does better on the exact same
data our CGCNN ensemble trained on. scripts/11_prepare_alignn_data.py and
scripts/12_train_alignn.py already do this correctly and were verified with a
local CPU smoke test (2 epochs, 60 crystals) - but a real run (~7,700
training crystals, enough epochs to converge, the full 4-layer/256-hidden
ALIGNN) is far too slow on a CPU: ALIGNN builds a line graph of bond angles on
top of the bond graph, which is real extra work per crystal per epoch that
CGCNN's single graph never does. Exactly the same reason the CGCNN ensemble
itself needed colab/PINK_CGCNN_colab.py rather than training locally.

WHY A GENERATOR RATHER THAN A CHECKED-IN .ipynb
-------------------------------------------------
Same reason as PINK_CGCNN_colab.py: the notebook embeds scripts/11 and
scripts/12 as a zip so it always matches the current source. Edit either
script and a checked-in notebook would silently train with stale code.
Re-run this generator after changing either script.

WHAT COLAB INSTALLS THAT THE LAPTOP ALREADY HAS
--------------------------------------------------
`alignn` (which pulls in jarvis-tools) is NOT part of this project's regular
environment.yml - it's specific to this one experiment. Colab installs it
fresh each run, same as it does for pymatgen/matminer in the CGCNN notebook.

A NOTE ON WHAT scripts/12 HAD TO WORK AROUND
------------------------------------------------
This installed ALIGNN version's default code path (model="alignn_atomwise",
neighbor_strategy="k-nearest") calls literal dgl.graph(...) regardless of
what its own release notes say about dropping the DGL dependency - confirmed
by actually running it, not by re-reading the release notes a second time.
scripts/12_train_alignn.py works around this with the model's own DGL-free
variant (model="alignn_atomwise_pure", neighbor_strategy="pure_torch") - see
that script's docstring for the full story. Colab's environment is a fresh
pip install, same package version, so this applies there too; nothing
different needs doing on the Colab side, which is exactly why it is worth
recording here rather than rediscovering it if a future Colab pip install
resolves to a newer ALIGNN release.
"""

import base64
import io
import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Deliberately code only. data_full/alignn_data.pkl (the converted matbench
# structures) is rebuilt on Colab from matminer directly - faster than
# uploading it, and it's a few hundred MB.
BUNDLE = [
    "scripts/01b_prepare_full_dataset.py",
    "scripts/02_train.py",
    "scripts/11_prepare_alignn_data.py",
    "scripts/12_train_alignn.py",
]


def build_bundle_b64():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in BUNDLE:
            path = os.path.join(PROJECT_ROOT, relative)
            if not os.path.exists(path):
                raise SystemExit(f"missing bundle file: {relative}")
            archive.write(path, relative)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def build_cells(bundle_b64):
    return [
        markdown("""
# PINK / ALIGNN — a second architecture, same data, same split

Trains ALIGNN (Choudhary & DeCost 2021 - a line graph of bond *angles* on top
of the usual bond graph, which is exactly the geometric information CGCNN's
convolution cannot see) to predict bulk (`bulk_modulus_kv`) and shear
(`shear_modulus_gv`) modulus on the **same 10,987-crystal matbench benchmark**,
using the **exact same train/val/test split** the CGCNN ensemble used -
computed once with our own `split_indices()`, not re-derived from ALIGNN's own
(different-RNG) split logic. See scripts/11_prepare_alignn_data.py's
docstring for why that distinction matters for a fair comparison.

**Before you run anything: Runtime → Change runtime type → T4 GPU (or better).**
ALIGNN builds a line graph of bond angles on top of the bond graph - real
extra work per crystal, per epoch, that CGCNN never does - so a CPU run here
would be considerably slower than the CGCNN notebook's already-long CPU
estimate. Budget on the order of a few hours per target on a T4; there is no
laptop CPU timing to compare against since this was only ever smoke-tested
locally (2 epochs, 60 crystals, seconds) to verify correctness, not to
benchmark full-run speed.
"""),
        markdown("## 1. Check the GPU\n\nIf this prints `cpu`, stop and switch the runtime type."),
        code("""
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("\\n*** NO GPU. Runtime -> Change runtime type -> T4 GPU, then rerun. ***")
"""),
        markdown("""
## 2. Install dependencies

`alignn` pulls in `jarvis-tools` automatically. This is a bigger install than
the CGCNN notebook's - budget a couple of minutes, and some dependency-resolver
noise is expected and harmless.
"""),
        code("""
%pip install -q pymatgen matminer alignn
print("done")
"""),
        markdown("## 3. Unpack the project code\n\nEmbedded as a zip, so this notebook always matches the repo it was generated from."),
        code(f'''
import base64, io, zipfile, os

BUNDLE_B64 = "{bundle_b64}"

os.makedirs("/content/pink_alignn", exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall("/content/pink_alignn")

os.chdir("/content/pink_alignn")
print("\\n".join(sorted(
    os.path.join(root, f).replace("/content/pink_alignn/", "")
    for root, _, files in os.walk(".") for f in files
    if not root.startswith("./."))))
'''),
        markdown("""
## 4. Convert matbench to ALIGNN's format

Downloads both matbench elastic datasets (~100 MB, same download the CGCNN
notebook does) and converts all 10,987 structures to JARVIS `Atoms` dicts -
this conversion itself is fast (seconds); the slow part (line-graph
construction) happens per-run in step 5, not here. Also computes the exact
same train/val/test split the CGCNN ensemble used.
"""),
        code("""
!python scripts/11_prepare_alignn_data.py
"""),
        markdown("""
## 5. Train both targets

One model per target - the plan here is "does a different architecture beat
ours," not another ensemble. 4 ALIGNN layers + 4 GCN layers, 256 hidden
features are the published ALIGNN defaults; `--n-early-stopping 30` stops a
target early if validation loss hasn't improved in 30 epochs rather than
burning the rest of the epoch budget once it's clearly converged.
"""),
        code("""
import subprocess, sys, time

COMMON = ["--epochs", "150", "--batch-size", "64", "--learning-rate", "0.001",
          "--alignn-layers", "4", "--gcn-layers", "4", "--hidden-features", "256",
          "--embedding-features", "64", "--n-early-stopping", "30",
          "--device", "cuda"]

def run(args):
    \"\"\"Stream output live - see PINK_CGCNN_colab.py for why -u matters here.\"\"\"
    print("$", " ".join(args), flush=True)
    result = subprocess.run([sys.executable, "-u"] + args)
    if result.returncode:
        raise SystemExit(f"FAILED (exit {result.returncode}): {' '.join(args)}")

start = time.time()
for target in ("bulk_modulus_kv", "shear_modulus_gv"):
    t0 = time.time()
    print(f"\\n{'=' * 62}\\n{target}   [{(time.time() - start) / 60:.0f} min elapsed]\\n{'=' * 62}",
          flush=True)
    run(["scripts/12_train_alignn.py", "--target", target,
         "--out-dir", f"results/alignn_{target}"] + COMMON)
    print(f">>> {target} done in {(time.time() - t0) / 60:.1f} min", flush=True)

print(f"\\nBoth targets finished in {(time.time() - start) / 60:.0f} min")
"""),
        markdown("""
## 6. Quick look at test-set accuracy

Same log10 convention as the rest of this project - matbench's targets are
already log10(GPa), and scripts/12 passes them through unchanged, so this MAE
is directly comparable to the CGCNN ensemble's 0.0630 / 0.0781.
"""),
        code("""
import json

for target in ("bulk_modulus_kv", "shear_modulus_gv"):
    with open(f"results/alignn_{target}/Test_results.json") as fh:
        test_results = json.load(fh)
    print(target, "->", test_results if isinstance(test_results, dict) else test_results[:1])
"""),
        markdown("""
## 7. Download the results

Brings back both checkpoints, configs, and test-set predictions.
"""),
        code("""
!cd /content/pink_alignn && zip -qr /content/alignn_results.zip results
from google.colab import files
files.download("/content/alignn_results.zip")
"""),
        markdown("""
### Back on the laptop

```bash
unzip -o ~/Downloads/alignn_results.zip -d "/Users/mac/Desktop/Cgcnn project"
```

`results/alignn_bulk_modulus_kv/` and `results/alignn_shear_modulus_gv/` each
hold `best_model.pt`, `config.json`, and `prediction_results_test_set.csv` -
the last of these is enough on its own to add ALIGNN's row to the metrics
comparison table (it already has both predicted and true values for the held-
out test set, in the same units).

Feeding ALIGNN's moduli through `slack_physics()` (scripts/07_predict_kappa.py)
the way the CGCNN ensemble's are is a natural follow-up, not done by this
notebook - it needs a small adapter script to run the downloaded ALIGNN
checkpoint on complete-data/'s 1,213 CIFs the way scripts/04_predict_moduli.py
does for CGCNN, since ALIGNN's checkpoint format and inference call are
different from CGCNN's.
"""),
    ]


def main():
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": build_cells(build_bundle_b64()),
    }

    out_path = os.path.join(PROJECT_ROOT, "colab", "PINK_ALIGNN.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(notebook, fh, indent=1)

    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    print("Upload it to https://colab.research.google.com and run top to bottom.")


if __name__ == "__main__":
    main()
