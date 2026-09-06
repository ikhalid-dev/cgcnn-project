#!/usr/bin/env python3
"""
Build the Kaggle GPU kernel for the MATBENCH SUBSET SIZE TEST.
================================================================================

    python kaggle/build_matbench_subset_kernel.py
    python kaggle/run_matbench_subset_kernel.py

THE QUESTION
--------------
Step 53 established that the CGCNN still underfits on AFLOW even with every
brake released - its TRAIN error (0.088 on K) is still above a composition-only
tree's TEST error (0.059). Dropout, huber and multi-head competition are ruled
out. So what is left?

The clue is that the SAME architecture with the SAME recipe BEATS the tree on
matbench and loses to it on AFLOW:

    matbench K   CGCNN 0.0630   tree 0.0868    CGCNN wins by 27%
    AFLOW    K   CGCNN 0.1154   tree 0.0592    TREE  wins by 49%

Three things differ between those datasets:

    training crystals   7,691 (matbench) vs 3,894 (AFLOW)
    graph source        matminer structures vs AFLOW POSCARs
    label source        matbench VRH vs AFLOW AEL

This run separates the FIRST from the other two. Train the CGCNN on a random
3,894-crystal subset of matbench - AFLOW's exact training-set size - and score
it on the UNCHANGED matbench test set.

    if it degrades to tree level    -> the cause is DATA QUANTITY. AFLOW is
                                       simply too small for a graph network,
                                       and no amount of tuning fixes that.
    if it still beats the tree      -> the cause is the AFLOW DATA ITSELF -
                                       its graphs or its labels - and the
                                       moduli models trained on it should not
                                       be trusted for screening.

THE DESIGN
------------
`02_train.py` gained ONE optional flag, `--train-subsample N`, which shrinks the
TRAINING set only and leaves val and test untouched. That last part is the whole
point: shrinking via --train-ratio would move the test set too, and the result
would be confounded with a different set of held-out crystals. The flag defaults
to None, in which case the script behaves bit-identically to every baseline it
has ever produced.

Twelve runs: {K_VRH, G_VRH} x 3 seeds x {3,894 subset, 7,691 full}.

The full-size arm is the IN-SESSION CONTROL, and it does double duty. It should
reproduce this project's published single-model numbers (K about 0.070, G about
0.084). If it does, the subset comparison is sound AND the new flag is verified
end to end. If it does not, nothing else in this run can be trusted.

The tree control at both sizes is fitted LOCALLY by
scripts/cgcnn/54_matbench_subset_trees.py - it is sklearn on composition
features and needs no GPU.
"""

import base64
import io
import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build_matbench_subset")

# =============================================================================
#  CONFIG - edit these to change what the GPU run actually does
# =============================================================================

# Kaggle derives the real URL slug from the TITLE, not from the id we send.
# Getting this wrong makes every later status/output call 404 against a kernel
# that exists under a different name.
KERNEL_TITLE = "PINK matbench subset size test"
KERNEL_SLUG = KERNEL_TITLE.lower().replace(" ", "-")

# T4 is sm_75 and inside the supported range. See the docstring before
# touching this.
MACHINE_SHAPE = "NvidiaTeslaT4"

# AFLOW POSCARs + aflow_agl.csv, uploaded once with `kaggle datasets create`.
# Mounted read-only at /kaggle/input/aflow-agl-kappa/. This is the same
# dataset build_joint_kernel.py uses, so nothing needs re-uploading.
# No Kaggle Dataset needed: matbench arrives via matminer at runtime.
DATASET_SOURCES = []

# Epoch budget. Both arms use it, so the cosine schedule anneals over the same
# trajectory in each - changing it changes both, which is what keeps this a
# one-variable experiment.
EPOCHS = 200          # 02_train.py's default - the recipe every baseline used
SEEDS = [42, 1, 2]    # three, so the seed spread is measurable
SPLIT_SEED = 42       # the split every matbench number in this project uses
# AFLOW's training-set size. Matching it exactly is what makes the comparison a
# size test rather than a vague "less data" test.
SUBSET_N = 3894

# Flags shared by every run. Per the argparse ordering rule these are placed
# BEFORE the per-run flags when the command is assembled, so a per-run
# override always wins - the trap that silently reverted four of six sweeps.
COMMON_FLAGS = ["--epochs", str(EPOCHS), "--device", "cuda"]

# The matbench recipe, spelled out. These are 02_train.py's own defaults and
# are passed explicitly so the kernel log records what actually ran.
MB_RECIPE = ["--batch-size", "32", "--lr", "0.02", "--weight-decay", "1e-5",
             "--atom-fea-len", "64", "--h-fea-len", "128", "--n-conv", "3",
             "--n-h", "1", "--scheduler", "plateau"]

# Files shipped inside the kernel source, at their real repo paths. Path
# shape matters: 01_train_direct_kappa.py computes PROJECT_ROOT by walking up
# from __file__, so flattening it would resolve one level too high and
# `import cgcnn_scratch` would fail.
BUNDLE = [
    ("cgcnn_scratch/__init__.py", "cgcnn_scratch/__init__.py"),
    ("cgcnn_scratch/data.py", "cgcnn_scratch/data.py"),
    ("cgcnn_scratch/model.py", "cgcnn_scratch/model.py"),
    ("cgcnn_scratch/joint.py", "cgcnn_scratch/joint.py"),
    ("cgcnn_scratch/atom_init.json", "cgcnn_scratch/atom_init.json"),
    # Paths are PRESERVED, not flattened: 02_train.py computes PROJECT_ROOT by
    # walking up three directories from __file__, so scripts/cgcnn/ must stay.
    ("scripts/cgcnn/01b_prepare_full_dataset.py", "scripts/cgcnn/01b_prepare_full_dataset.py"),
    ("scripts/cgcnn/02_train.py", "scripts/cgcnn/02_train.py"),
]

# =============================================================================
#  END OF CONFIG
# =============================================================================


def build_run_plan():
    """(tag, script, flags) for all twelve runs.

    Subset arm and full arm differ by exactly one flag. The full arm is the
    in-session control and should reproduce the published single-model numbers.
    """
    plan = []
    for target in ("K_VRH", "G_VRH"):
        for size, extra in (("sub", ["--train-subsample", str(SUBSET_N)]),
                            ("full", [])):
            for s_ in SEEDS:
                plan.append((
                    f"{target}_{size}_s{s_}", "scripts/cgcnn/02_train.py",
                    ["--data-dir", "data_full", "--target", target,
                     "--seed", str(s_), "--split-seed", str(SPLIT_SEED),
                     "--tag", f"mbsubset_{target}_{size}_s{s_}"] + MB_RECIPE + extra))
    return plan


def build_bundle_b64():
    """Zip the source files and base64 them so the kernel is self-contained.

    Kaggle's kernel-source limit is around 1 MB, which is why the AFLOW
    POSCARs travel as an attached Dataset instead of in here. These few source
    files are a few tens of KB.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for local, arcname in BUNDLE:
            path = os.path.join(PROJECT_ROOT, local)
            if not os.path.exists(path):
                raise SystemExit(f"missing bundle file: {local}")
            archive.write(path, arcname)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


KERNEL_TEMPLATE = '''"""
PINK - direct kappa (no Slack) vs the Slack pipeline, three seeds each.

Generated by kaggle/build_aflow_recipe_kernel.py - do not edit here, edit the
generator. Everything worth keeping is left in /kaggle/working.
"""

import base64
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

# Work in /kaggle/temp: it is scratch and is NOT captured as kernel output, so
# the ~240 MB graph cache is never downloaded. Only what is copied to
# /kaggle/working at the end comes back.
WORK = "/kaggle/temp/pink_mb_subset"
OUT = "/kaggle/working"

BUNDLE_B64 = "{bundle_b64}"
RUN_PLAN = {run_plan!r}
COMMON_FLAGS = {common_flags!r}


def run(args, cwd=WORK):
    """Run a step, streaming output live, and abort the kernel on failure.

    -u because Python block-buffers stdout when it is not a terminal; without
    it the log stays empty until a run finishes and a healthy job looks
    identical to a hung one.

    PYTHONPATH=WORK so `import cgcnn_scratch` resolves regardless of which
    directory the child script thinks it is in.
    """
    env = dict(os.environ, PYTHONPATH=WORK)
    print("$", " ".join(args), flush=True)
    result = subprocess.run([sys.executable, "-u"] + args, cwd=cwd, env=env)
    if result.returncode:
        raise SystemExit(f"FAILED (exit {{result.returncode}}): {{' '.join(args)}}")


print("=" * 70, flush=True)
print("ENVIRONMENT", flush=True)
print("=" * 70, flush=True)
import torch
print("torch      :", torch.__version__, flush=True)
print("cuda avail :", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    # enable_gpu alone is silently ignored by some API versions - machine_shape
    # is the field that actually attaches a device. Fail loudly rather than
    # quietly burning an hour of CPU time.
    raise SystemExit("No GPU. Check machine_shape in kernel-metadata.json.")

print("gpu        :", torch.cuda.get_device_name(0), flush=True)

# is_available() IS NOT ENOUGH - a GPU whose compute capability this torch was
# not compiled for reports available=True and a correct device name, and only
# fails on the first real kernel launch, minutes in. Check the capability
# against the compiled arch list AND do a real op.
capability = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print("capability :", f"sm_{{capability[0]}}{{capability[1]}}", flush=True)
print("torch archs:", arch_list, flush=True)
if f"sm_{{capability[0]}}{{capability[1]}}" not in arch_list:
    raise SystemExit(
        f"GPU is sm_{{capability[0]}}{{capability[1]}} but this torch build only "
        f"supports {{arch_list}}. Set MACHINE_SHAPE to NvidiaTeslaT4 (sm_75) in "
        f"kaggle/build_aflow_recipe_kernel.py.")
try:
    _probe = (torch.ones(64, device="cuda") * 2).sum().item()
    assert _probe == 128.0, _probe
    print("cuda probe : OK", flush=True)
except Exception as exc:
    raise SystemExit(f"GPU present but unusable - a real CUDA op failed: {{exc}}")

# pymatgen is needed by 36_prepare_gamma_dataset.py to read the POSCARs.
# Kaggle preinstalls no usable version. Requires enable_internet.
print("\\ninstalling pymatgen + matminer...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pymatgen", "matminer"], check=True)

# --- Unpack the embedded source -------------------------------------------
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall(WORK)
print("\\nunpacked:", sorted(os.listdir(WORK)), flush=True)

# --- Build the matbench dataset -------------------------------------------
# No Kaggle Dataset is mounted for this run: 01b_prepare_full_dataset.py pulls
# matbench's elastic benchmark through matminer and builds the graph cache
# itself. That needs internet, which is enabled in kernel-metadata.json and
# requires a phone-verified Kaggle account.
#
# --skip-match skips the CIF-to-matbench matching step, which only exists to
# align this set with the paper's own 1,213 crystals and is not needed here.
print("\\n" + "=" * 70, flush=True)
print("BUILDING THE MATBENCH DATASET (10,987 crystals)", flush=True)
print("=" * 70, flush=True)
run(["scripts/cgcnn/01b_prepare_full_dataset.py", "--skip-match"])

# --- Train every run ------------------------------------------------------
print("\\n" + "=" * 70, flush=True)
print(f"TRAINING - {{len(RUN_PLAN)}} runs x {{COMMON_FLAGS}}", flush=True)
print("=" * 70, flush=True)
start = time.time()
failed = []
for i, (tag, script, flags) in enumerate(RUN_PLAN, 1):
    t0 = time.time()
    print("\\n" + "=" * 70, flush=True)
    print(f"[{{i}}/{{len(RUN_PLAN)}}] {{tag}}  ({{script}})", flush=True)
    print("=" * 70, flush=True)
    try:
        # COMMON_FLAGS FIRST, per-run flags AFTER: argparse keeps the LAST
        # occurrence of a repeated option, so this order lets a per-run flag
        # override a common one. The reverse silently ignores the sweep.
        run([script] + COMMON_FLAGS + list(flags))
    except SystemExit as exc:
        print(f"!! {{tag}} FAILED: {{exc}}", flush=True)
        failed.append(tag)
    print(f"-- {{tag}} finished in {{time.time() - t0:.0f}}s", flush=True)

print(f"\\nall runs done in {{time.time() - start:.0f}}s; failures: {{failed or 'none'}}",
      flush=True)

# --- Collect everything worth keeping into /kaggle/working ---------------
# One table assembled HERE rather than by reassembling JSON after download -
# that friction is not worth repeating.
os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
os.makedirs(os.path.join(OUT, "predictions"), exist_ok=True)

for pattern, dest in (
        ("results/cgcnn/model_mbsubset_*.pth", "models"),
        ("results/cgcnn/summary_mbsubset_*.json", "models"),
        ("results/cgcnn/history_mbsubset_*.csv", "predictions")):
    for path in sorted(glob.glob(os.path.join(WORK, pattern))):
        shutil.copy(path, os.path.join(OUT, dest, os.path.basename(path)))

rows = []
for path in sorted(glob.glob(os.path.join(OUT, "models", "*.json"))):
    with open(path) as fh:
        blob = json.load(fh)
    res = blob.get("results", {{}})
    test = res.get("test", {{}})
    name = os.path.basename(path)
    # 02_train.py writes its metrics flat (test_mae, train_mae, ...) while
    # 01_train_direct_kappa.py nests them under results.test. Both shapes are
    # written through as-is; the local comparison script does the like-for-like
    # scoring from the checkpoints, so nothing here has to guess a schema.
    flat = {{f"top_{{k}}": v for k, v in blob.items()
            if isinstance(v, (int, float, str)) and k not in ("tag",)}}
    rows.append({{
        "tag": blob.get("tag"),
        "arm": ("kappa" if "mbrecipe_s" in name
                else "K_VRH" if "K_VRH" in name
                else "G_VRH" if "G_VRH" in name else "?"),
        "best_epoch": blob.get("best_epoch"),
        "seconds": blob.get("seconds"),
        **flat,
        # The direct arm reports kappa error directly; the control's summary
        # carries its own kappa keys. Whatever is present is written through
        # unchanged - 02 does the like-for-like scoring locally from the
        # checkpoints, so nothing here needs to guess at a common schema.
        **{{f"test_{{k}}": v for k, v in test.items() if not isinstance(v, (dict, list))}},
    }})
if rows:
    import csv as _csv
    keys = sorted({{k for r in rows for k in r}})
    with open(os.path.join(OUT, "mbsubset_summary.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("\\nsummary table:", flush=True)
    for r in rows:
        print("   ", r, flush=True)

print("\\noutput:", sorted(os.listdir(OUT)), flush=True)
if failed:
    raise SystemExit(f"{{len(failed)}} run(s) failed: {{failed}}")
'''


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    # The username comes from the authenticated Kaggle client rather than a
    # hardcoded string, so a credential change cannot silently push to a
    # kernel id that does not exist.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    username = api.config_values["username"]

    plan = build_run_plan()
    kernel_path = os.path.join(BUILD_DIR, "pink_mb_subset.py")
    with open(kernel_path, "w") as fh:
        fh.write(KERNEL_TEMPLATE.format(bundle_b64=build_bundle_b64(),
                                        run_plan=plan,
                                        common_flags=COMMON_FLAGS))

    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "pink_mb_subset.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # enable_gpu ALONE IS NOT ENOUGH on this API version - it is accepted
        # and silently ignored, and the kernel lands on the CPU image.
        # machine_shape is the field that actually attaches a device.
        "enable_gpu": True,
        "machine_shape": MACHINE_SHAPE,
        # Un-pin the docker image. Kaggle pins a kernel to the image sha of its
        # FIRST run; if that run was CPU, every later push inherits the pin.
        "docker_image_pinning_type": "latest",
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(BUILD_DIR, "kernel-metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    # Compile the generated kernel before anyone can push it. The template is a
    # format string containing Python source, so one lost backslash turns an
    # escape into a real newline and produces a file that only fails ON KAGGLE,
    # eight minutes and one GPU slot later. That happened once; this makes it
    # impossible to repeat.
    import py_compile
    try:
        py_compile.compile(kernel_path, doraise=True)
    except py_compile.PyCompileError as exc:
        raise SystemExit(f"generated kernel does not compile:\n{exc}")
    print("generated kernel compiles OK")

    size_kb = os.path.getsize(kernel_path) / 1024
    print(f"Built kernel for {username}/{KERNEL_SLUG}")
    print(f"  {kernel_path}  ({size_kb:.0f} KB)")
    print(f"  {BUILD_DIR}/kernel-metadata.json")
    print(f"  accelerator: {MACHINE_SHAPE}")
    print(f"\nRun plan ({len(plan)} runs x {EPOCHS} epochs):")
    for tag, script, flags in plan:
        print(f"    {tag:<14} {os.path.basename(script):<28} {' '.join(flags)}")
    print("\nPush and watch it with:")
    print("    python kaggle/run_matbench_subset_kernel.py")


if __name__ == "__main__":
    main()
