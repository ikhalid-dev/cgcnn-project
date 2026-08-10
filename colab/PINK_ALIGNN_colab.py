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
#
# cgcnn_scratch/ is bundled too even though ALIGNN never uses it: scripts/11
# and scripts/12 both import 01b_prepare_full_dataset.py / 02_train.py via
# importlib to reuse load_benchmark()/split_indices()/pick_device() rather
# than duplicate them - but importing a module runs ALL of its top-level
# code, including 01b's and 02_train's own `from cgcnn_scratch.data import
# ...`. Without cgcnn_scratch/ physically present, that import fails with
# ModuleNotFoundError before either helper function is ever reached - caught
# by actually running this notebook on Colab, not predicted in advance.
# (local_path, archive_path) pairs - local follows this project's own
# scripts/cgcnn|alignn/ split, archive stays flat (what the code embedded
# below, which runs unpacked on Colab, refers to it as). See
# PINK_CGCNN_colab.py's identical note for why this means zero changes to
# any remotely-executed code.
BUNDLE = [
    ("cgcnn_scratch/__init__.py", "cgcnn_scratch/__init__.py"),
    ("cgcnn_scratch/data.py", "cgcnn_scratch/data.py"),
    ("cgcnn_scratch/model.py", "cgcnn_scratch/model.py"),
    ("cgcnn_scratch/atom_init.json", "cgcnn_scratch/atom_init.json"),
    ("scripts/cgcnn/01b_prepare_full_dataset.py", "scripts/01b_prepare_full_dataset.py"),
    ("scripts/cgcnn/02_train.py", "scripts/02_train.py"),
    ("scripts/alignn/11_prepare_alignn_data.py", "scripts/11_prepare_alignn_data.py"),
    ("scripts/alignn/12_train_alignn.py", "scripts/12_train_alignn.py"),
    # alignn_gpu_driver.py needs to exist as a real file on disk too, not
    # just live in this notebook's embedded cell - its own worker mode
    # re-launches itself via a fixed on-disk path (colab/alignn_gpu_driver.py),
    # since colab exec -f / a notebook cell has no reliable __file__ to
    # re-derive that path from. See that script's own docstring.
    ("colab/alignn_gpu_driver.py", "colab/alignn_gpu_driver.py"),
]

DRIVER_PATH = os.path.join(PROJECT_ROOT, "colab", "alignn_gpu_driver.py")


def build_bundle_b64():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for local, arcname in BUNDLE:
            path = os.path.join(PROJECT_ROOT, local)
            if not os.path.exists(path):
                raise SystemExit(f"missing bundle file: {local}")
            archive.write(path, arcname)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


def read_driver_source():
    with open(DRIVER_PATH) as fh:
        return fh.read()


def build_cells(bundle_b64, driver_source):
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
## 4. Convert matbench to ALIGNN's format, then train both targets

This cell embeds `colab/alignn_gpu_driver.py`'s actual source (the same file
`colab/run_alignn_cli.py` sends via `colab exec -f` for the CLI-driven path -
one driver, two ways to launch it, so notebook and CLI can never drift
apart). `check_gpu()` fails loudly if this session isn't actually a GPU
runtime; `run_data_prep()` downloads matbench (~100 MB) and converts all
10,987 structures to JARVIS `Atoms` dicts, computing the exact same
train/val/test split the CGCNN ensemble used; `train_all_targets()` then
trains `bulk_modulus_kv` and `shear_modulus_gv` in turn (4 ALIGNN layers + 4
GCN layers + 256 hidden features are the published ALIGNN defaults;
`--n-early-stopping 30` stops a target early once validation loss plateaus)
and zips both results directories when done.

**If one target fails, this cell does NOT stop** - it logs the failure and
moves on to the next target, then reports which of the two actually
succeeded. That's a deliberate change from an earlier version of this
notebook, which raised `SystemExit` on the first failure and hid the actual
Python traceback that explained why - if a target does fail here, the real
traceback prints above, in this same cell's output, not hidden behind a
generic "FAILED (exit 1)".
"""),
        # Plain concatenation, NOT an f-string: driver_source is a full
        # Python script full of its own {braces} (f-strings, dict/set
        # literals) that an outer f-string would misinterpret as format
        # placeholders and fail to parse - caught by checking, not assumed
        # safe by analogy with step 3's bundle_b64 (which has no braces).
        code(driver_source + "\n\ncheck_gpu()\nrun_data_prep()\ntrain_all_targets()\n"),
        markdown("""
## 5. Quick look at test-set accuracy

Same log10 convention as the rest of this project - matbench's targets are
already log10(GPa), and scripts/12 passes them through unchanged, so this MAE
is directly comparable to the CGCNN ensemble's 0.0630 / 0.0781.
"""),
        code("""
import json

for target in ("bulk_modulus_kv", "shear_modulus_gv"):
    path = f"results/alignn_{target}/Test_results.json"
    if not os.path.exists(path):
        print(target, "-> no Test_results.json (this target failed - see cell 4's output above)")
        continue
    with open(path) as fh:
        test_results = json.load(fh)
    print(target, "->", test_results if isinstance(test_results, dict) else test_results[:1])
"""),
        markdown("""
## 6. Download the results

Downloads the zip cell 4 already built (`colab/alignn_results.zip`) - both
checkpoints, configs, and test-set predictions for whichever target(s)
succeeded.
"""),
        code("""
from google.colab import files
files.download("colab/alignn_results.zip")
"""),
        markdown("""
### Back on the laptop

```bash
unzip -o ~/Downloads/alignn_results.zip -d "/Users/mac/Desktop/Cgcnn project"
```

`results/alignn/alignn_bulk_modulus_kv/` and `results/alignn/alignn_shear_modulus_gv/`
each hold `best_model.pt`, `config.json`, and `prediction_results_test_set.csv` -
the last of these is enough on its own to add ALIGNN's row to the metrics
comparison table (it already has both predicted and true values for the held-
out test set, in the same units).

Feeding ALIGNN's moduli through `slack_physics()` the way the CGCNN
ensemble's are: `python scripts/alignn/15_alignn_predict_moduli.py` then
`python scripts/alignn/16_alignn_predict_kappa.py` - both already built (they
were the natural follow-up mentioned in an earlier version of this notebook,
not hypothetical anymore), producing
`results/alignn/alignn_kappa_predictions.csv` and a comparison against the
CGCNN ensemble's own kappa_L.
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
        "cells": build_cells(build_bundle_b64(), read_driver_source()),
    }

    out_path = os.path.join(PROJECT_ROOT, "colab", "PINK_ALIGNN.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(notebook, fh, indent=1)

    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    print("Upload it to https://colab.research.google.com and run top to bottom.")


if __name__ == "__main__":
    main()
