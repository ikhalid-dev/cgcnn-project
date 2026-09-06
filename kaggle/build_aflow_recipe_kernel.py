#!/usr/bin/env python3
"""
Build the Kaggle GPU kernel for the AFLOW / MATBENCH-RECIPE experiment.
================================================================================

    python kaggle/build_aflow_recipe_kernel.py
    python kaggle/run_aflow_recipe_kernel.py            # push and watch

WHAT THIS RUNS, AND WHY
-------------------------
Steps 43/44 found the composition-only tree beats round 9 on AFLOW, and
diagnosed UNDERFITTING rather than a real advantage: round 9's TRAINING error
on AFLOW K (0.103) is worse than the tree's TEST error (0.059), and its
test/train ratios sit at 1.08-1.23 - the band where a model never learned,
against a healthy baseline of 2.36. Step 51 then showed the tree beats round 9
at the low-kappa call (F1 0.646 vs 0.400, CI excluding zero), and the
direct-kappa experiment lost to the tree too. Three results, one suspect.

This is the test step 44 named and nobody ran: the ORIGINAL single-target
CGCNN on AFLOW with the MATBENCH recipe - no dropout, MSE, one target per
model, batch 32, lr 0.02, 200 epochs, plateau schedule. Same widths
(64/128/3 convs), so this is not a bigger network, it is the same network with
the regularisation removed.

    K_VRH  x 3 seeds     scripts/cgcnn/02_train.py, UNMODIFIED
    G_VRH  x 3 seeds     scripts/cgcnn/02_train.py, UNMODIFIED
    kappa  x 3 seeds     01_train_direct_kappa.py with matbench-recipe flags

02_train.py runs untouched because it is the script every headline baseline in
this project was produced by; editing it to read a second dataset layout would
change the recipe under test. Instead step 52 presents AFLOW in the layout it
already expects (gid -> mb_id, gamma_graphs.pt -> graphs.pt), and its
split_indices was VERIFIED to produce the identical 3894/834/835 split that
round 9, the tree baseline and the direct model were all scored on - so every
number here is directly comparable to them.

ONE RESIDUAL DIFFERENCE, STATED RATHER THAN HIDDEN
----------------------------------------------------
The kappa arm uses 01_train_direct_kappa.py, which offers a cosine schedule
but not 02_train.py's ReduceLROnPlateau. Every other recipe difference that
matters for underfitting - dropout, loss, batch size, learning rate, epoch
budget, single target - is matched exactly. The scheduler is the one thing
that is not, and it should be named in any writeup.

WHAT THE OUTCOMES MEAN
------------------------
  gap closes            round 9 was underfit. Every AFLOW conclusion in this
                        project needs redoing with a properly-fit model,
                        including step 51's shortlist verdict and the
                        direct-vs-Slack non-answer.
  gap persists AND      composition genuinely carries the signal in AFLOW and
  train error drops     structure adds little. A real finding, not a bug.
  below the tree's

Train error is as important as test error here: a model that still cannot fit
its own training data has not answered the question.

THE P100 TRAP - DO NOT CHANGE MACHINE_SHAPE TO P100
------------------------------------------------------
Kaggle's image ships a torch compiled for sm_70..sm_120. The P100 is sm_60, so
every CUDA op dies with "no kernel image is available" AFTER
torch.cuda.is_available() has returned True. The kernel checks the capability
against the compiled arch list and does a real device op, failing in seconds.
"""

import base64
import io
import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build_aflow_recipe")

# =============================================================================
#  CONFIG - edit these to change what the GPU run actually does
# =============================================================================

# Kaggle derives the real URL slug from the TITLE, not from the id we send.
# Getting this wrong makes every later status/output call 404 against a kernel
# that exists under a different name.
KERNEL_TITLE = "PINK AFLOW matbench recipe"
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
EPOCHS = 200          # 02_train.py's default - the matbench recipe, not 37's 300
SEEDS = [42, 1, 2]    # three, so the seed spread is measurable
SPLIT_SEED = 42       # VERIFIED to reproduce round 9 / tree / direct-model split

# Flags shared by every run. Per the argparse ordering rule these are placed
# BEFORE the per-run flags when the command is assembled, so a per-run
# override always wins - the trap that silently reverted four of six sweeps.
COMMON_FLAGS = ["--epochs", str(EPOCHS), "--device", "cuda"]

# The matbench recipe, spelled out. These are 02_train.py's own defaults and
# are passed explicitly so the kernel log records what actually ran.
MB_RECIPE = ["--batch-size", "32", "--lr", "0.02", "--weight-decay", "1e-5",
             "--atom-fea-len", "64", "--h-fea-len", "128", "--n-conv", "3",
             "--n-h", "1", "--scheduler", "plateau"]
AFLOW_DIR = "data_full/aflow_matbench_layout"

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
    ("scripts/cgcnn/02_train.py", "scripts/cgcnn/02_train.py"),
    ("scripts/cgcnn/52_aflow_as_matbench_layout.py",
     "scripts/cgcnn/52_aflow_as_matbench_layout.py"),
    ("direct_kappa_no_slack/01_train_direct_kappa.py",
     "direct_kappa_no_slack/01_train_direct_kappa.py"),
]

# =============================================================================
#  END OF CONFIG
# =============================================================================


def build_run_plan():
    """(tag, script, flags) for all nine runs.

    K and G go through 02_train.py UNMODIFIED - that is the point of the
    experiment, so the recipe cannot be paraphrased. kappa goes through
    01_train_direct_kappa.py with the same recipe expressed in its own flag
    names; see the module docstring for the one residual difference.
    """
    plan = []
    for target in ("K_VRH", "G_VRH"):
        for s_ in SEEDS:
            plan.append((
                f"{target}_s{s_}", "scripts/cgcnn/02_train.py",
                ["--data-dir", AFLOW_DIR, "--target", target,
                 "--seed", str(s_), "--split-seed", str(SPLIT_SEED),
                 "--tag", f"aflow_mbrecipe_{target}_s{s_}"] + MB_RECIPE))
    for s_ in SEEDS:
        plan.append((
            f"kappa_s{s_}", "direct_kappa_no_slack/01_train_direct_kappa.py",
            ["--seed", str(s_), "--tag", f"mbrecipe_s{s_}",
             # the matbench recipe in 01's flag names: brakes off
             "--dropout", "0.0", "--loss-fn", "mse",
             "--batch-size", "32", "--learning-rate", "0.02"]))
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
WORK = "/kaggle/temp/pink_aflow_recipe"
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
print("BUILDING THE AFLOW GRAPH CACHE (shared by every arm)", flush=True)
print("=" * 70, flush=True)
run(["scripts/cgcnn/36_prepare_gamma_dataset.py"])

# Present it in the layout 02_train.py already expects, so that script runs
# UNMODIFIED - editing it would change the very recipe under test. Step 52
# also asserts the cache/label id order matches, which is what guarantees the
# split is the same 3894/834/835 round 9 and the tree were scored on.
print("\\n" + "=" * 70, flush=True)
print("ADAPTING AFLOW TO THE MATBENCH LAYOUT", flush=True)
print("=" * 70, flush=True)
run(["scripts/cgcnn/52_aflow_as_matbench_layout.py"])

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
        # the K/G arm, written by 02_train.py into results/cgcnn/
        ("results/cgcnn/model_aflow_mbrecipe_*.pth", "models"),
        ("results/cgcnn/summary_aflow_mbrecipe_*.json", "models"),
        ("results/cgcnn/history_aflow_mbrecipe_*.csv", "predictions"),
        ("results/cgcnn/predictions_aflow_mbrecipe_*.csv", "predictions"),
        # the kappa arm, written by 01_train_direct_kappa.py
        ("direct_kappa_no_slack/models/*mbrecipe*.pth", "models"),
        ("direct_kappa_no_slack/models/*mbrecipe*.json", "models"),
        ("direct_kappa_no_slack/results/csv/*mbrecipe*.csv", "predictions")):
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
    with open(os.path.join(OUT, "aflow_mbrecipe_summary.csv"), "w", newline="") as fh:
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
    kernel_path = os.path.join(BUILD_DIR, "pink_aflow_recipe.py")
    with open(kernel_path, "w") as fh:
        fh.write(KERNEL_TEMPLATE.format(bundle_b64=build_bundle_b64(),
                                        run_plan=plan,
                                        common_flags=COMMON_FLAGS))

    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "pink_aflow_recipe.py",
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
    print("    python kaggle/run_aflow_recipe_kernel.py")


if __name__ == "__main__":
    main()
