#!/usr/bin/env python3
"""
Build the Google Colab notebook for training the PINK CGCNN on a GPU.
=====================================================================

    python colab/PINK_CGCNN_colab.py        # writes colab/PINK_CGCNN.ipynb

WHY A GENERATOR RATHER THAN A CHECKED-IN .ipynb
-----------------------------------------------
The notebook needs to carry copies of `cgcnn_scratch/data.py`, `model.py` and
the training scripts, because Colab has no access to this private repo. Copies
rot: edit data.py here and a checked-in notebook would silently train with the
old featuriser, which is exactly the kind of bug that produces a model whose
predictions look fine and are wrong.

So the notebook is GENERATED from the current source files every time. Re-run
this script after changing anything in cgcnn_scratch/ or scripts/.
"""

import base64
import io
import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The files Colab needs. Deliberately code only - no data. The 240 MB graph
# cache is NOT uploaded; the notebook rebuilds it from matbench in about a
# minute, which is far quicker than pushing it over a network.
#
# (local_path, archive_path) pairs, not a flat list: local paths follow this
# project's own scripts/cgcnn/ split, but the archive path - what the path is
# CALLED inside the zip, and therefore what the code embedded below (which
# runs unpacked on Colab) refers to it as - stays flat, exactly as it always
# has. That means the remote-side run() calls two cells down needed zero
# changes for the cgcnn/alignn reorganisation; only this list did.
BUNDLE = [
    ("cgcnn_scratch/__init__.py", "cgcnn_scratch/__init__.py"),
    ("cgcnn_scratch/data.py", "cgcnn_scratch/data.py"),
    ("cgcnn_scratch/model.py", "cgcnn_scratch/model.py"),
    ("scripts/cgcnn/01b_prepare_full_dataset.py", "scripts/01b_prepare_full_dataset.py"),
    ("scripts/cgcnn/02_train.py", "scripts/02_train.py"),
    ("scripts/cgcnn/03_evaluate.py", "scripts/03_evaluate.py"),
    ("scripts/cgcnn/04_predict_moduli.py", "scripts/04_predict_moduli.py"),
    ("scripts/cgcnn/05_ensemble.py", "scripts/05_ensemble.py"),
    ("cgcnn_scratch/atom_init.json", "cgcnn_scratch/atom_init.json"),
]


def build_bundle_b64():
    """Zip the source files and base64 them, so the notebook is self-contained.

    Embedding beats "upload these files yourself": there is no way to run the
    notebook against a stale or partial copy of the code.
    """
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


def build_cells(bundle_b64):
    return [
        markdown("""
# PINK / CGCNN — train elastic-modulus models on a GPU

Trains a CGCNN to predict bulk (`K_VRH`) and shear (`G_VRH`) modulus on the
**full matbench elastic benchmark — 10,987 DFT-labelled crystals**, the same
training set the PINK paper used.

**Before you run anything: Runtime → Change runtime type → T4 GPU.**
On a T4 the whole thing takes roughly 30–60 minutes. On the CPU runtime it
takes about 9 hours, which defeats the point.

The code is embedded in this notebook, so there is nothing to upload.
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

`pymatgen` supplies the crystal handling and `matminer` fetches the matbench
datasets. This takes a couple of minutes and prints some dependency-resolver
noise, which is expected and harmless.
"""),
        code("""
%pip install -q pymatgen matminer
print("done")
"""),
        markdown("## 3. Unpack the project code\n\nEmbedded as a zip, so this notebook always matches the repo it was generated from."),
        code(f'''
import base64, io, zipfile, os

BUNDLE_B64 = "{bundle_b64}"

os.makedirs("/content/pink", exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall("/content/pink")

os.chdir("/content/pink")
print("\\n".join(sorted(
    os.path.join(root, f).replace("/content/pink/", "")
    for root, _, files in os.walk(".") for f in files
    if not root.startswith("./."))))
'''),
        markdown("""
## 4. Build the training set

Downloads both matbench elastic datasets (~100 MB) and converts all 10,987
crystals into cached graphs. Takes about 2 minutes.

`--skip-match` is passed because the provenance matching needs the 1,213 local
CIFs, which are not uploaded — that step runs back on the laptop.
"""),
        code("""
!python scripts/01b_prepare_full_dataset.py --skip-match
"""),
        markdown("_If the cell above fails on a pymatgen/numpy import, use "
                 "**Runtime → Restart session** and rerun from step 3 — the "
                 "pip install in step 2 replaces packages Colab preloaded._"),
        markdown("""
## 5. Train

Six runs — three ensemble members per target. Each member varies `--seed`
(weight initialisation) while `--split-seed 42` is **pinned**, so all members
share one train/val/test split. Without that the ensemble's test score would be
measured partly on data some members trained on.

`--num-workers 0` is deliberate: the graphs are already in RAM, so worker
processes would only add per-batch pickling across a process boundary. Workers
help when a dataset reads files; they cost here.
"""),
        # subprocess rather than a ! magic: ! inside a loop, with interpolated
        # variables and line continuations, is exactly where IPython's shell
        # parsing gets fragile. subprocess takes an argv list, so there is no
        # quoting or continuation to get wrong.
        code("""
import subprocess, sys, time

COMMON = ["--data-dir", "data_full", "--batch-size", "128", "--lr", "0.01",
          "--atom-fea-len", "64", "--h-fea-len", "128", "--n-h", "1",
          "--split-seed", "42", "--scheduler", "cosine", "--epochs", "200",
          "--device", "cuda", "--num-workers", "0"]

def run(args):
    \"\"\"Run a pipeline step, streaming its output, and stop on failure.

    The -u matters. Python block-buffers stdout at 8 KB when it is not a
    terminal, and a whole 200-epoch run prints only ~2 KB - so without it the
    cell shows NOTHING until each model finishes, and a healthy run is
    indistinguishable from a hung one.
    \"\"\"
    print("$", " ".join(args), flush=True)
    result = subprocess.run([sys.executable, "-u"] + args)
    if result.returncode:
        raise SystemExit(f"FAILED (exit {result.returncode}): {' '.join(args)}")

start = time.time()
for target in ("K_VRH", "G_VRH"):
    for tag, seed, n_conv in ((f"{target}_full", "42", "3"),
                              (f"{target}_s1",   "1", "4"),
                              (f"{target}_s2",   "2", "3")):
        t0 = time.time()
        print(f"\\n{'=' * 62}\\n{tag}  (seed {seed}, n_conv {n_conv})"
              f"   [{(time.time() - start) / 60:.0f} min elapsed]\\n{'=' * 62}",
              flush=True)
        run(["scripts/02_train.py", "--target", target, "--tag", tag,
             "--seed", seed, "--n-conv", n_conv] + COMMON)
        print(f">>> {tag} done in {(time.time() - t0) / 60:.1f} min", flush=True)

print(f"\\nAll six runs finished in {(time.time() - start) / 60:.0f} min")
"""),
        markdown("## 6. Score each target, alone and as an ensemble"),
        code("""
for target in ("K_VRH", "G_VRH"):
    run(["scripts/03_evaluate.py", "--target", target,
         "--data-dir", "data_full", "--tag", f"{target}_full"])
    run(["scripts/05_ensemble.py", "--target", target, "--data-dir", "data_full",
         "--tags", f"{target}_full,{target}_s1,{target}_s2"])
"""),
        markdown("""
## 7. Download the results

Brings back the six checkpoints, the metrics and the figures. Unzip this into
the project root on the laptop, then run **`scripts/cgcnn/04_predict_moduli.py`**
there to produce `pink_moduli_predictions.csv` — that step needs the 1,213
local CIFs, which never left the laptop.
"""),
        code("""
!cd /content/pink && zip -qr /content/pink_results.zip results
from google.colab import files
files.download("/content/pink_results.zip")
"""),
        markdown("""
### Back on the laptop

```bash
unzip -o ~/Downloads/pink_results.zip -d "/Users/mac/Desktop/Cgcnn project"
python scripts/cgcnn/04_predict_moduli.py \\
    --k-tag K_VRH_full,K_VRH_s1,K_VRH_s2 \\
    --g-tag G_VRH_full,G_VRH_s1,G_VRH_s2
```

Checkpoints are always serialised on CPU, so GPU-trained weights load on a
machine with no CUDA.
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

    out_path = os.path.join(PROJECT_ROOT, "colab", "PINK_CGCNN.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(notebook, fh, indent=1)

    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    print("Upload it to https://colab.research.google.com and run top to bottom.")


if __name__ == "__main__":
    main()
