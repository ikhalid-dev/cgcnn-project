#!/usr/bin/env python3
"""
Build the Kaggle GPU kernel for the DIRECT-KAPPA (no Slack) experiment.
================================================================================

    python kaggle/build_direct_kappa_kernel.py
    python kaggle/run_direct_kappa_kernel.py            # push and watch

WHAT THIS RUNS
----------------
The experiment in direct_kappa_no_slack/ - can a CGCNN predict AFLOW's
lattice thermal conductivity DIRECTLY from the crystal graph, better than the
current pipeline predicts moduli and pushes them through Slack's formula?

Both arms run in ONE kernel, on one GPU, off one dataset build:

    direct_s42 / s1 / s2     direct_kappa_no_slack/01_train_direct_kappa.py
    control_s42 / s1 / s2    scripts/cgcnn/37_train_gamma.py  (3 heads + Slack)

The control is re-trained here rather than quoted from the existing round-9
checkpoints, because this project's own rule is that a control belongs in the
same session as the thing it controls for. The existing checkpoints are still
scored locally by 02 as a third reference, but the in-session pair is what any
claim rests on.

Three seeds per arm because an effect smaller than the seed spread is not a
result, and one seed cannot measure the spread.

WHY A GPU AT ALL
------------------
On this laptop the direct model runs ~15 s/epoch, so 300 epochs x 6 runs is
about seven and a half hours. On a T4 the same six runs are well under an
hour, and 37_train_gamma.py's own summary shows ~400 s per run at this size.

THE P100 TRAP - DO NOT CHANGE MACHINE_SHAPE TO P100
------------------------------------------------------
Kaggle's image ships a torch compiled for sm_70..sm_120. The P100 is sm_60,
so every CUDA op dies with "no kernel image is available for execution on the
device" - AFTER torch.cuda.is_available() has returned True and
get_device_name() has cheerfully printed "Tesla P100-PCIE-16GB". A full
12-run grid was lost to this once. The kernel below checks the compute
capability against torch's compiled arch list AND performs a real device op,
so it fails in seconds instead of minutes.
"""

import base64
import io
import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build_direct_kappa")

# =============================================================================
#  CONFIG - edit these to change what the GPU run actually does
# =============================================================================

# Kaggle derives the real URL slug from the TITLE, not from the id we send.
# Getting this wrong makes every later status/output call 404 against a kernel
# that exists under a different name.
KERNEL_TITLE = "PINK direct kappa no slack"
KERNEL_SLUG = KERNEL_TITLE.lower().replace(" ", "-")

# T4 is sm_75 and inside the supported range. See the docstring before
# touching this.
MACHINE_SHAPE = "NvidiaTeslaT4"

# AFLOW POSCARs + aflow_agl.csv, uploaded once with `kaggle datasets create`.
# Mounted read-only at /kaggle/input/aflow-agl-kappa/. This is the same
# dataset build_joint_kernel.py uses, so nothing needs re-uploading.
DATASET_SOURCES = ["aizazkhalidkhan/aflow-agl-kappa"]

# Epoch budget. Both arms use it, so the cosine schedule anneals over the same
# trajectory in each - changing it changes both, which is what keeps this a
# one-variable experiment.
EPOCHS = 300          # matches 37_train_gamma.py and 01_train_direct_kappa.py
SEEDS = [42, 1, 2]    # three, so the seed spread is measurable

# Flags shared by every run. Per the argparse ordering rule these are placed
# BEFORE the per-run flags when the command is assembled, so a per-run
# override always wins - the trap that silently reverted four of six sweeps.
COMMON_FLAGS = ["--epochs", str(EPOCHS), "--device", "cuda"]

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
    ("scripts/cgcnn/36_prepare_gamma_dataset.py", "scripts/cgcnn/36_prepare_gamma_dataset.py"),
    ("scripts/cgcnn/37_train_gamma.py", "scripts/cgcnn/37_train_gamma.py"),
    ("direct_kappa_no_slack/01_train_direct_kappa.py",
     "direct_kappa_no_slack/01_train_direct_kappa.py"),
]

# =============================================================================
#  END OF CONFIG
# =============================================================================


def build_run_plan():
    """(tag, script, flags) for all six runs: three direct, three control.

    The control is 37_train_gamma.py at n_heads=3 - the round-9 configuration
    exactly, so the only difference between the arms is the head target and
    the presence of the Slack formula.
    """
    plan = []
    for s in SEEDS:
        plan.append((f"direct_s{s}", "direct_kappa_no_slack/01_train_direct_kappa.py",
                     ["--seed", str(s), "--tag", f"s{s}"]))
    for s in SEEDS:
        plan.append((f"control_s{s}", "scripts/cgcnn/37_train_gamma.py",
                     ["--init-seed", str(s), "--tag", f"control_s{s}", "--n-heads", "3"]))
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

Generated by kaggle/build_direct_kappa_kernel.py - do not edit here, edit the
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
WORK = "/kaggle/temp/pink_direct"
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
        f"kaggle/build_direct_kappa_kernel.py.")
try:
    _probe = (torch.ones(64, device="cuda") * 2).sum().item()
    assert _probe == 128.0, _probe
    print("cuda probe : OK", flush=True)
except Exception as exc:
    raise SystemExit(f"GPU present but unusable - a real CUDA op failed: {{exc}}")

# pymatgen is needed by 36_prepare_gamma_dataset.py to read the POSCARs.
# Kaggle preinstalls no usable version. Requires enable_internet.
print("\\ninstalling pymatgen...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pymatgen"], check=True)

# --- Unpack the embedded source -------------------------------------------
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall(WORK)
print("\\nunpacked:", sorted(os.listdir(WORK)), flush=True)

# --- Copy the AFLOW dataset in from the mounted Kaggle Dataset ------------
# The mount directory is DISCOVERED, not assumed. Kaggle does not always mount
# a dataset at /kaggle/input/<ref-slug>: the directory can be derived from the
# dataset TITLE instead, and version 1 of this kernel died on a hardcoded
# "/kaggle/input/aflow-agl-kappa" that did not exist. Searching for the file
# that actually matters is robust to whatever Kaggle names the folder, and the
# full listing is printed so a genuine attachment failure is distinguishable
# from a naming mismatch.
INPUT_ROOT = "/kaggle/input"
print("\\n/kaggle/input contains:",
      sorted(os.listdir(INPUT_ROOT)) if os.path.isdir(INPUT_ROOT) else "NOTHING",
      flush=True)

AFLOW_INPUT = None
for candidate in sorted(glob.glob(os.path.join(INPUT_ROOT, "*"))):
    if not os.path.isdir(candidate):
        continue
    # The marker is aflow_agl.csv, the one file every downstream step needs.
    for root, _, files in os.walk(candidate):
        if "aflow_agl.csv" in files:
            # AFLOW_INPUT must be the directory that ACTUALLY HOLDS the csv,
            # not the top-level mount. Kaggle nests these as
            # /kaggle/input/datasets/<owner>/<slug>/, so pointing at the mount
            # root makes the recursive POSCAR walk succeed while the direct
            # os.path.join for aflow_agl.csv silently misses - which is exactly
            # how version 3 copied 5,563 structures and then died claiming the
            # csv was missing.
            AFLOW_INPUT = root
            print(f"found aflow_agl.csv in {{root}}", flush=True)
            break
    if AFLOW_INPUT:
        break

if AFLOW_INPUT is None:
    listing = []
    if os.path.isdir(INPUT_ROOT):
        for root, dirs, files in os.walk(INPUT_ROOT):
            listing.append(f"  {{root}}: {{sorted(dirs)[:8]}} {{sorted(files)[:8]}}")
            if len(listing) > 25:
                break
    raise SystemExit(
        "aflow_agl.csv not found anywhere under /kaggle/input.\\n"
        "The dataset is either not attached or is attached under an unexpected\\n"
        "name. Check dataset_sources in kernel-metadata.json against\\n"
        "`kaggle datasets list -m`. What IS mounted:\\n" + "\\n".join(listing))
os.makedirs(os.path.join(WORK, "data_full", "aflow_structures"), exist_ok=True)
for name in ("aflow_agl.csv", "aflow_soft.csv"):
    src_p = os.path.join(AFLOW_INPUT, name)
    if os.path.exists(src_p):
        shutil.copy(src_p, os.path.join(WORK, "data_full", name))
n_pos = 0
for root, _, files in os.walk(AFLOW_INPUT):
    for fn in files:
        if fn.endswith(".poscar"):
            shutil.copy(os.path.join(root, fn),
                        os.path.join(WORK, "data_full", "aflow_structures", fn))
            n_pos += 1
print(f"\\nAFLOW dataset: {{n_pos}} structures copied", flush=True)

# --- Build the AFLOW graph cache ------------------------------------------
# Produces data_full/gamma_graphs.pt + gamma_labels.csv (5,563 crystals).
# BOTH arms read exactly this one build, which is what makes the comparison
# one-variable: same graphs, same labels, same split_seed.
print("\\n" + "=" * 70, flush=True)
print("BUILDING THE AFLOW GRAPH CACHE (shared by both arms)", flush=True)
print("=" * 70, flush=True)
run(["scripts/cgcnn/36_prepare_gamma_dataset.py"])

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
        ("direct_kappa_no_slack/models/*.pth", "models"),
        ("direct_kappa_no_slack/models/*.json", "models"),
        ("results/cgcnn/model_37_control_s*.pth", "models"),
        ("results/cgcnn/summary_37_control_s*.json", "models"),
        ("direct_kappa_no_slack/results/csv/*.csv", "predictions")):
    for path in sorted(glob.glob(os.path.join(WORK, pattern))):
        shutil.copy(path, os.path.join(OUT, dest, os.path.basename(path)))

rows = []
for path in sorted(glob.glob(os.path.join(OUT, "models", "*.json"))):
    with open(path) as fh:
        blob = json.load(fh)
    res = blob.get("results", {{}})
    test = res.get("test", {{}})
    rows.append({{
        "tag": blob.get("tag"),
        "arm": "direct" if "direct" in os.path.basename(path) else "control",
        "best_epoch": blob.get("best_epoch"),
        "seconds": blob.get("seconds"),
        # The direct arm reports kappa error directly; the control's summary
        # carries its own kappa keys. Whatever is present is written through
        # unchanged - 02 does the like-for-like scoring locally from the
        # checkpoints, so nothing here needs to guess at a common schema.
        **{{f"test_{{k}}": v for k, v in test.items() if not isinstance(v, (dict, list))}},
    }})
if rows:
    import csv as _csv
    keys = sorted({{k for r in rows for k in r}})
    with open(os.path.join(OUT, "direct_vs_control_summary.csv"), "w", newline="") as fh:
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
    kernel_path = os.path.join(BUILD_DIR, "pink_direct.py")
    with open(kernel_path, "w") as fh:
        fh.write(KERNEL_TEMPLATE.format(bundle_b64=build_bundle_b64(),
                                        run_plan=plan,
                                        common_flags=COMMON_FLAGS))

    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "pink_direct.py",
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

    size_kb = os.path.getsize(kernel_path) / 1024
    print(f"Built kernel for {username}/{KERNEL_SLUG}")
    print(f"  {kernel_path}  ({size_kb:.0f} KB)")
    print(f"  {BUILD_DIR}/kernel-metadata.json")
    print(f"  accelerator: {MACHINE_SHAPE}")
    print(f"\nRun plan ({len(plan)} runs x {EPOCHS} epochs):")
    for tag, script, flags in plan:
        print(f"    {tag:<14} {os.path.basename(script):<28} {' '.join(flags)}")
    print("\nPush and watch it with:")
    print("    python kaggle/run_direct_kappa_kernel.py")


if __name__ == "__main__":
    main()
