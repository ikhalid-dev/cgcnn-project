#!/usr/bin/env python3
"""
Build the Kaggle kernel that trains the JOINT two-head CGCNN on a GPU.
=======================================================================

    python kaggle/build_joint_kernel.py     # writes kaggle/build_joint/
    python kaggle/run_joint_kernel.py       # push, poll, download

WHAT THIS RUNS AND WHY
----------------------
The joint model exists to fix a measured problem: kappa_L is computed from K
and G through the Slack model, and the Grueneisen parameter depends ONLY on
the ratio K/G. In log space that ratio is a DIFFERENCE, so two separately
trained models contribute two independent errors to it. Measured on the 1,648
crystal test set with the existing separate models:

    residual correlation err_K vs err_G  = +0.263 (single) / +0.297 (ensemble)
    MAE log10(K/G)                       =  0.0993          /  0.0887
      ... which is WORSE than either individual MAE (0.0696 / 0.0836)

    kappa_L MAE attributable to the moduli = 0.2065 / 0.1897 log10
    same modulus error but perfectly correlated = 0.0882  <- 54% is decorrelation

Two changes are meant to fix it, and this kernel measures them SEPARATELY:

    shared trunk  + ratio parameterisation  -> target_mode "G_and_ratio"
    shared trunk  + old parameterisation    -> target_mode "K_and_G"

Running only the first would leave "the trunk did it" and "the ratio did it"
hopelessly confounded. Hence PHASE A below trains both, three seeds each.

WHY KAGGLE
----------
On this laptop the joint model runs ~25 s/epoch on CPU, so one 200-epoch run
is well over an hour and the 12-run grid below would take a full day. On a T4
it is 5-7 minutes per run. Kaggle also has a real API (push / status / output),
so the whole loop is scriptable - Colab needs a human clicking in a browser.

WHY THE SOURCE IS EMBEDDED
--------------------------
Kaggle cannot reach this private repo, so the kernel carries a base64 zip of
cgcnn_scratch/ and the scripts it needs. It is GENERATED from the live files
rather than checked in, so it can never silently go stale and train with an old
featuriser.

Note on layout: unlike build_kernel.py, this bundle PRESERVES the repo's
scripts/cgcnn/ path rather than flattening it. 31_train_joint.py finds the
project root by walking three directories up from itself, so flattening would
make it resolve one level too high and fail to import cgcnn_scratch.
PYTHONPATH is also set in the kernel as belt-and-braces.
"""

import base64
import glob
import io
import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build_joint")

# =============================================================================
#  CONFIG - edit these to change what the GPU run actually does
# =============================================================================

# Kaggle derives the real URL slug from the TITLE, not from the id we send, so
# derive the slug from the title rather than setting it by hand. Getting this
# wrong makes every later status/output call 404 against a kernel that exists
# under a different name.
KERNEL_TITLE = "PINK CGCNN joint two-head"
KERNEL_SLUG = KERNEL_TITLE.lower().replace(" ", "-")

# Accelerator. Allowed values: NvidiaTeslaT4, NvidiaTeslaP100, Tpu1VmV38.
#
# DO NOT USE P100 - it no longer works, whatever the raw FLOPS say.
# Kaggle's current image ships torch 2.10.0+cu128, which is compiled for
# sm_70 through sm_120 only. The P100 is sm_60, so every CUDA kernel launch
# dies with "no kernel image is available for execution on the device".
# The trap is that torch.cuda.is_available() still returns True and
# get_device_name() cheerfully reports "Tesla P100-PCIE-16GB" - the failure
# only surfaces on the first real op. A full 12-run grid burned 6 minutes
# getting through dataset prep before failing 12 times in a row.
# T4 is sm_75, inside the supported range. build_alignn_kernel.py already
# uses T4; build_kernel.py still asks for P100 and will hit this wall.
MACHINE_SHAPE = "NvidiaTeslaT4"

# AFLOW structures + metadata, uploaded once with `kaggle datasets create`.
# Mounted read-only at /kaggle/input/<slug>/. Re-upload with
# `kaggle datasets version -p <dir> -m "..."` when the fetch adds more.
DATASET_SOURCES = ["aizazkhalidkhan/aflow-agl-kappa"]

# Epoch budget for every run in the grid. The cosine schedule anneals over
# exactly this many epochs, so changing it changes the LR trajectory too - it
# is not a pure "train longer" knob.
# 200 matches 02_train.py, which the separate-model baselines were trained
# with. Round 1 used 150 and that is one of the things being controlled for.
# COMMON_FLAGS reads this rather than repeating the number, so the banner
# printed at build time can never disagree with what actually runs.
EPOCHS = 300  # matches 37_train_gamma.py's CONFIG["epochs"]; cosmetic only -
              # COMMON_FLAGS does not pass --epochs, so the script's own
              # default is what actually runs regardless of this constant.

# Seeds for the ensemble members. init_seed varies weight initialisation and
# batch order; split_seed is PINNED at 42 everywhere so all runs (and the
# existing separate-model results) share one held-out test set. Without that
# pin the ensemble score would be measured partly on data some members saw.
# ROUND 5: nine models, grouped into THREE independent ensembles of three.
# Round 4's joint ensemble beat the separate ensemble by 0.0029 (0.1868 vs
# 0.1897) - a real win, but smaller than the seed spread of a single model
# (0.0033-0.0194), so one ensemble cannot establish it. Three independent
# ensembles give a spread on the ensemble itself, which is the number that
# decides whether the margin survives.
SEEDS = [42, 1, 2]

# ---- ROUND 9: single-stage 3-head gamma model, full-scale AFLOW -----------
# Round 8's two-stage transfer made things WORSE than doing nothing: an
# offline recalibration diagnostic on the r8 checkpoints proved the failure
# is not a fixable affine offset - derived-gamma was accidentally cancelling
# the matbench/AFLOW moduli convention bias, and real gamma exposed it
# instead of correcting it. That closes off transfer as an approach.
#
# The actual bottleneck 37_train_gamma.py identified was never the transfer
# scheme - it was data volume: MAE log10(K) was 0.1817 on AFLOW's OWN 1,022
# training crystals, against 0.063 on matbench's 7,691. The AFLOW structure
# fetch that stalled mid-project has since finished: 5,563 structures now
# have matching AGL labels, up from 1,460 (a 3.8x increase in usable
# crystals, 1,022 -> ~3,894 at the same 70% train split). This round re-runs
# the exact same single-stage recipe from 37_train_gamma.py on that full set,
# with nothing else changed, so the effect of data volume alone is isolated
# rather than confounded with an architecture or recipe change.
#
# Three seeds, not one - a single run cannot tell a real gain from seed
# noise (round 5's ensemble work exists for exactly this reason). Reuses
# SEEDS above rather than a new list - same three seeds, same meaning.
PHASE_A = []           # round 9 does not use the 31_train_joint.py plan
PHASE_B = []
ENSEMBLE_GROUPS = {}   # 32_ensemble_joint.py assumes the 31/38 summary schema,
                       # not 37_train_gamma.py's - ensembling the 3 seeds is a
                       # deliberate next step, not done inside this kernel.

# Flags common to every run. These mirror the CONFIG defaults in
# 37_train_gamma.py; anything not listed here uses that file's default
# (n_heads=3, epochs=300, early_stop_patience=80, ...).
COMMON_FLAGS = [
    "--split-seed", "42",
    "--device", "cuda",
    "--num-workers", "0",
    "--print-every", "50",
]

# Files copied into the kernel. (local path, path inside the archive) - the
# archive path MUST mirror the repo layout, see the module docstring.
# Trimmed to only what 37_train_gamma.py and its dataset-builder touch - this
# round never builds the matbench set or the AFLOW-soft merge, so
# 01b_prepare_full_dataset.py / 31_train_joint.py / 32_ensemble_joint.py /
# 35_merge_aflow.py / 38_train_gamma_transfer.py are not needed and would
# only cost GPU minutes downloading/building data nothing here reads.
BUNDLE = [
    ("cgcnn_scratch/__init__.py", "cgcnn_scratch/__init__.py"),
    ("cgcnn_scratch/data.py", "cgcnn_scratch/data.py"),
    ("cgcnn_scratch/model.py", "cgcnn_scratch/model.py"),
    ("cgcnn_scratch/joint.py", "cgcnn_scratch/joint.py"),
    ("cgcnn_scratch/atom_init.json", "cgcnn_scratch/atom_init.json"),
    ("scripts/cgcnn/36_prepare_gamma_dataset.py", "scripts/cgcnn/36_prepare_gamma_dataset.py"),
    ("scripts/cgcnn/37_train_gamma.py", "scripts/cgcnn/37_train_gamma.py"),
]

# =============================================================================
#  END OF CONFIG
# =============================================================================


# Directories shipped wholesale. The 737 AFLOW POSCARs cannot be listed one
# per BUNDLE entry, and the kernel cannot fetch them itself (that took 15
# minutes of polite serial requests). Zipped they are ~0.33 MB base64, which
# is nothing next to a Kaggle kernel's limits.
# Nothing is shipped wholesale any more. The 1,460 AFLOW POSCARs pushed the
# kernel source to 1.38 MB and Kaggle rejected it with a bare
# "400 Bad Request" - the source limit is ~1 MB. They now live in a Kaggle
# DATASET (see DATASET_SOURCES), which has no such limit and does not need
# re-uploading on every kernel edit.
BUNDLE_DIRS = []


def build_bundle_b64():
    """Zip the source files and base64 them so the kernel is self-contained."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for local, arcname in BUNDLE:
            path = os.path.join(PROJECT_ROOT, local)
            if not os.path.exists(path):
                raise SystemExit(f"missing bundle file: {local}")
            archive.write(path, arcname)
        for local_dir, pattern in BUNDLE_DIRS:
            root = os.path.join(PROJECT_ROOT, local_dir)
            if not os.path.isdir(root):
                raise SystemExit(f"missing bundle directory: {local_dir}")
            hits = sorted(glob.glob(os.path.join(root, pattern)))
            if not hits:
                raise SystemExit(f"no {pattern} files in {local_dir}")
            for path in hits:
                archive.write(path, os.path.join(
                    local_dir, os.path.basename(path)))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_full_plan():
    """Expand SEEDS into (tag, flags) for 37_train_gamma.py, round 9.

    No target-mode / freeze-moduli axis this round - one recipe (the
    round-7 default: n_heads=3, huber loss), three seeds, on the full
    5,563-crystal AFLOW set. Tag prefix r9 so the comparison table's
    summary_r9_*.json glob picks these up and nothing older.
    """
    return [(f"r9_full_s{seed}", ["--tag", f"r9_full_s{seed}",
                                  "--init-seed", str(seed)])
            for seed in SEEDS]


def build_run_plan():
    """Expand PHASE_A and PHASE_B into a flat list of (tag, [flags]) runs.

    Done here, on the laptop, rather than inside the kernel: the plan is then
    visible in the generated file and in the printed summary, so what the GPU
    is about to spend an hour on can be checked before pushing it.
    """
    plan = []

    # Phase A - both parameterisations, every seed.
    for entry in PHASE_A:
        suffix, mode = entry[0], entry[1]
        extra = entry[2] if len(entry) > 2 else {}
        for seed in SEEDS:
            tag = f"{suffix}_s{seed}"
            flags = ["--target-mode", mode, "--init-seed", str(seed), "--tag", tag]
            for key, value in extra.items():
                flags += [f"--{key.replace('_', '-')}", str(value)]
            plan.append((tag, flags))

    # Phase B - sweep, single seed only. A sweep does not need three seeds;
    # the point is to rank knob settings, and the winner gets re-run with all
    # three afterwards.
    sweep_mode = PHASE_A[0][1]     # sweep around the primary parameterisation
    for entry in PHASE_B:
        overrides = dict(entry)
        tag = f"sweep_{overrides.pop('tag')}"
        flags = ["--target-mode", sweep_mode,
                 "--init-seed", str(SEEDS[0]), "--tag", tag]
        for key, value in overrides.items():
            flags += [f"--{key.replace('_', '-')}", str(value)]
        plan.append((tag, flags))

    return plan


KERNEL_TEMPLATE = '''#!/usr/bin/env python3
"""
PINK / CGCNN - joint two-head model (shared trunk, K and G together).

Generated by kaggle/build_joint_kernel.py - do not edit here, edit the
generator. Trains the run plan below on the full 10,987-crystal matbench
elastic set and leaves every checkpoint, history and summary in
/kaggle/working/results for download.
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

# Work in /kaggle/temp: it is scratch and is NOT captured as kernel output.
# Only results/ is copied to /kaggle/working at the end, so we do not download
# the 240 MB graph cache for nothing.
WORK = "/kaggle/temp/pink_joint"
OUT = "/kaggle/working"

BUNDLE_B64 = "{bundle_b64}"
RUN_PLAN = {run_plan!r}
COMMON_FLAGS = {common_flags!r}


def run(args, cwd=WORK):
    """Run a step, streaming output live, and abort the kernel on failure.

    -u because Python block-buffers stdout when it is not a terminal; without
    it the log stays empty until a run finishes and a healthy job looks
    identical to a hung one.

    PYTHONPATH=WORK so `import cgcnn_scratch` resolves no matter which
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

# is_available() IS NOT ENOUGH. A GPU whose compute capability the installed
# torch was not compiled for still reports available=True and a correct device
# name; it only fails on the first real kernel launch, which on this kernel is
# ~6 minutes in, after dataset prep. That is exactly what a P100 (sm_60) does
# against torch 2.10+cu128 (built for sm_70..sm_120). So check the capability
# against the compiled arch list AND do a real op, and fail in seconds.
capability = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print("capability :", f"sm_{{capability[0]}}{{capability[1]}}", flush=True)
print("torch archs:", arch_list, flush=True)
if f"sm_{{capability[0]}}{{capability[1]}}" not in arch_list:
    raise SystemExit(
        f"GPU is sm_{{capability[0]}}{{capability[1]}} but this torch build only "
        f"supports {{arch_list}}. Change MACHINE_SHAPE in "
        f"kaggle/build_joint_kernel.py (T4 = sm_75 works).")
try:
    # The definitive test: allocate and reduce on the device for real.
    _probe = (torch.ones(64, device="cuda") * 2).sum().item()
    assert _probe == 128.0, _probe
    print("cuda probe : OK", flush=True)
except Exception as exc:
    raise SystemExit(f"GPU present but unusable - a real CUDA op failed: {{exc}}")

# matminer pulls the matbench datasets; pymatgen does the crystal handling.
# Kaggle preinstalls neither at a usable version. Needs internet enabled on the
# kernel, which requires a phone-verified Kaggle account.
print("\\ninstalling pymatgen + matminer...", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pymatgen", "matminer"], check=True)

# --- Unpack the embedded source -------------------------------------------
os.makedirs(WORK, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall(WORK)
print("\\nunpacked:", sorted(os.listdir(WORK)), flush=True)

# --- Fetch the AFLOW dataset from the mounted Kaggle Dataset --------------
# Round 9 trains ONLY on AFLOW (37_train_gamma.py) - there is no matbench
# build and no AFLOW-into-matbench merge this round, unlike rounds 5-8.
# The POSCARs and aflow_agl.csv arrive via the attached Kaggle dataset, not
# the bundle (1,460 of them alone pushed the kernel source over Kaggle's
# ~1 MB limit - see the module docstring). Copy them into WORK/data_full so
# 36_prepare_gamma_dataset.py finds them exactly where it would locally.
# The mount directory is DISCOVERED, not assumed. Kaggle used to mount a
# dataset at /kaggle/input/<ref-slug> and now nests it as
# /kaggle/input/datasets/<owner>/<slug>, so the hardcoded path below stopped
# existing. Worse, the old code only WARNED and carried on, so the real
# failure surfaced later as a confusing "missing aflow_agl.csv" from
# 36_prepare_gamma_dataset.py. Searching for the file that actually matters is
# robust to whatever Kaggle names the folder next.
INPUT_ROOT = "/kaggle/input"
print("\\n/kaggle/input contains:",
      sorted(os.listdir(INPUT_ROOT)) if os.path.isdir(INPUT_ROOT) else "NOTHING",
      flush=True)

AFLOW_INPUT = None
for candidate in sorted(glob.glob(os.path.join(INPUT_ROOT, "*"))):
    if not os.path.isdir(candidate):
        continue
    for root, _, files in os.walk(candidate):
        if "aflow_agl.csv" in files:
            # The directory that HOLDS the csv, not the top-level mount:
            # pointing at the mount root makes the recursive POSCAR walk
            # succeed while the direct join for aflow_agl.csv silently misses.
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
        "The dataset is either not attached or is mounted under an unexpected\\n"
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
print(f"\\nAFLOW dataset: {{n_pos}} structures copied from {{AFLOW_INPUT}}", flush=True)

# --- Build the AFLOW gamma dataset ----------------------------------------
# 36_prepare_gamma_dataset.py turns the copied POSCARs plus aflow_agl.csv
# into gamma_graphs.pt / gamma_labels.csv - 5,563 crystals this round, up
# from the 1,022 training crystals 37_train_gamma.py had in round 7.
print("\\n" + "=" * 70, flush=True)
print("BUILDING GAMMA DATASET", flush=True)
print("=" * 70, flush=True)
run(["scripts/cgcnn/36_prepare_gamma_dataset.py"])

# --- Train every run in the plan ------------------------------------------
print("\\n" + "=" * 70, flush=True)
print(f"TRAINING - {{len(RUN_PLAN)}} runs", flush=True)
print("=" * 70, flush=True)
start = time.time()
failed = []
for i, (tag, flags) in enumerate(RUN_PLAN, 1):
    t0 = time.time()
    print("\\n" + "=" * 70, flush=True)
    print(f"[{{i}}/{{len(RUN_PLAN)}}] {{tag}}"
          f"   [{{(time.time() - start) / 60:.0f}} min elapsed]", flush=True)
    print("=" * 70, flush=True)
    try:
        # COMMON_FLAGS FIRST, per-run flags LAST. argparse keeps the LAST
        # occurrence of a repeated flag, so this ordering lets a sweep entry
        # override a common setting. Round 2 had these the other way round
        # and COMMON_FLAGS silently shadowed 4 of 6 sweeps - dropout,
        # weight_decay, loss_fn and optimizer all reverted to the defaults
        # and those four runs came back byte-identical to the control.
        run(["scripts/cgcnn/37_train_gamma.py"] + COMMON_FLAGS + flags)
        print(f">>> {{tag}} done in {{(time.time() - t0) / 60:.1f}} min", flush=True)
    except SystemExit as exc:
        # One bad sweep setting must not throw away the runs that already
        # succeeded - record it and keep going.
        print(f">>> {{tag}} FAILED: {{exc}}", flush=True)
        failed.append(tag)

print(f"\\nAll runs finished in {{(time.time() - start) / 60:.0f}} min", flush=True)

# --- Combine the members into one ensemble --------------------------------
# Only meaningful when every member shares one split (--split-seed is pinned in
# COMMON_FLAGS) and every run succeeded. A partial ensemble would quietly be
# scored with fewer members than intended.
ENSEMBLE_GROUPS = {ensemble_groups!r}
if ENSEMBLE_GROUPS and not failed:
    print("\\n" + "=" * 70, flush=True)
    print("ENSEMBLING", flush=True)
    print("=" * 70, flush=True)
    for out_tag, tags in ENSEMBLE_GROUPS.items():
        print("\\n--- " + out_tag + " ---", flush=True)
        run(["scripts/cgcnn/32_ensemble_joint.py", "--results-dir", "results/cgcnn",
             "--tags", ",".join(tags), "--out-tag", out_tag])
elif failed:
    print("\\nskipping ensembles - these members failed:", failed, flush=True)
if failed:
    print(f"FAILED: {{failed}}", flush=True)

# --- Collect every summary into one comparison table ----------------------
# The whole point of the grid is the comparison, so build it here rather than
# making someone reassemble twelve JSON files by hand after downloading.
print("\\n" + "=" * 70, flush=True)
print("COMPARISON", flush=True)
print("=" * 70, flush=True)

# Baseline: 37_train_gamma.py's own round-7 result, same recipe, same script,
# on the 1,022 training crystals available before the AFLOW structure fetch
# finished. The only thing this round changes is data volume, so this row is
# the direct before/after for that one variable.
BASELINES = [
    ("BASELINE 1022-xtal", 0.1817, 0.1898, 0.1923, 0.5662, 0.3408, 0.190),
]

rows = []
results_dir = os.path.join(WORK, "results", "cgcnn")
for name in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
    # 37_train_gamma.py now stamps its own script number into every output
    # filename (summary_37_<tag>.json, not summary_<tag>.json) so results
    # sitting next to other scripts' output in this same directory can be
    # told apart - see scripts/cgcnn/37_train_gamma.py's save block.
    if not (name.startswith("summary_37_r9_") and name.endswith(".json")):
        continue
    with open(os.path.join(results_dir, name)) as fh:
        blob = json.load(fh)
    t = blob["results"]["test"]
    rows.append((blob["tag"], t["mae_log_K"], t["mae_log_G"], t["mae_gamma"],
                 t["gamma_corr"], t["kappa_mae_predicted"], t["recall10_predicted"]))

header = (f"{{'run':<18}} {{'K':>7}} {{'G':>7}} {{'gamma':>7}} {{'g_corr':>8}}"
          f" {{'kappa':>7}} {{'recall10':>9}}")
print(header, flush=True)
print("-" * len(header), flush=True)
for name, k, g, gm, gc, kap, r10 in BASELINES:
    print(f"{{name:<18}} {{k:7.4f}} {{g:7.4f}} {{gm:7.4f}} {{gc:+8.3f}}"
          f" {{kap:7.4f}} {{100 * r10:8.1f}}%", flush=True)
print("-" * len(header), flush=True)
# Sort by predicted-gamma kappa MAE - the actual downstream objective.
for name, k, g, gm, gc, kap, r10 in sorted(rows, key=lambda x: x[5]):
    print(f"{{name:<18}} {{k:7.4f}} {{g:7.4f}} {{gm:7.4f}} {{gc:+8.3f}}"
          f" {{kap:7.4f}} {{100 * r10:8.1f}}%", flush=True)

# Also write it as CSV so the laptop side does not have to scrape the log.
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
with open(os.path.join(OUT, "results", "gamma_comparison.csv"), "w") as fh:
    fh.write("run,mae_log_K,mae_log_G,mae_gamma,gamma_corr,"
             "kappa_mae_predicted,recall10_predicted\\n")
    for name, k, g, gm, gc, kap, r10 in BASELINES + sorted(rows, key=lambda x: x[5]):
        fh.write(f"{{name}},{{k}},{{g}},{{gm}},{{gc}},{{kap}},{{r10}}\\n")

# --- Hand the results back ------------------------------------------------
# Checkpoints are serialised on CPU by the training script, so these load on a
# laptop with no CUDA.
shutil.copytree(os.path.join(WORK, "results"), os.path.join(OUT, "results"),
                dirs_exist_ok=True)

print("\\n" + "=" * 70, flush=True)
print("OUTPUT", flush=True)
print("=" * 70, flush=True)
for root, _, files in os.walk(os.path.join(OUT, "results")):
    for fname in sorted(files):
        path = os.path.join(root, fname)
        print(f"  {{os.path.getsize(path) / 1e3:9.1f}} KB  {{fname}}", flush=True)
print("\\nPIPELINE COMPLETE", flush=True)
'''


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    # Read the authenticated username so the kernel id is right for whoever
    # runs this, rather than hard-coding one account.
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    username = api.config_values.get("username")
    if not username:
        raise SystemExit("Could not determine the Kaggle username.")

    plan = build_full_plan()
    kernel_path = os.path.join(BUILD_DIR, "pink_joint.py")
    with open(kernel_path, "w") as fh:
        # Ensemble members = the Phase A runs (Phase B sweeps vary settings,
        # so they are not interchangeable members of one ensemble).
        # Map each seed group onto the tags build_run_plan() actually emitted,
        # so a rename of the Phase A prefix cannot silently break the grouping.
        # Group names are "<entry prefix>_ens", so stripping the suffix
        # recovers which Phase A entry's tags the group refers to.
        ensemble_groups = {name: [f"{name[:-4]}_s{seed}" for seed in seeds]
                           for name, seeds in ENSEMBLE_GROUPS.items()}
        emitted = {tag for tag, _ in plan}
        for name, tags in ensemble_groups.items():
            missing = [t for t in tags if t not in emitted]
            assert not missing, f"{name} references untrained tags: {missing}"
        fh.write(KERNEL_TEMPLATE.format(bundle_b64=build_bundle_b64(),
                                        run_plan=plan,
                                        common_flags=COMMON_FLAGS,
                                        ensemble_groups=ensemble_groups))

    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": "pink_joint.py",
        "language": "python",
        # Kaggle shows a "script" kernel as a Notebook in the UI, and it is
        # what run_kernel.py's push/poll/pull loop drives. A real .ipynb would
        # need different tooling for no benefit.
        "kernel_type": "script",
        "is_private": True,
        # enable_gpu ALONE IS NOT ENOUGH on this API version - it is accepted
        # and silently ignored, and the kernel lands on the CPU image.
        # machine_shape is the field that actually attaches a device.
        "enable_gpu": True,
        "machine_shape": MACHINE_SHAPE,
        # Un-pin the docker image. Kaggle pins a kernel to the image sha of its
        # FIRST run; if that run was CPU, every later push inherits the pin and
        # comes back CPU-only no matter what accelerator is requested.
        "docker_image_pinning_type": "latest",
        # Needed to pip install pymatgen/matminer and to download matbench.
        # Kaggle only allows internet on kernels for phone-verified accounts.
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(os.path.join(BUILD_DIR, "kernel-metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    # Compile the generated kernel before anyone can push it. The template is a
    # format string containing Python source, so it is parsed twice - once when
    # this builder is read, once on Kaggle - and one lost backslash turns an
    # escape into a real newline. That file is only parsed for the first time on
    # Kaggle, which costs eight minutes and a GPU slot to discover. It happened
    # once, on build_aflow_recipe_kernel.py.
    #
    # This catches SYNTAX errors only - it compiles to bytecode, it does not run
    # the code. A missing dataset or a bad path still fails remotely.
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
    for tag, flags in plan:
        print(f"    {tag:<18} {' '.join(flags)}")
    print("\nPush and watch it with:")
    print("  python kaggle/run_joint_kernel.py")


if __name__ == "__main__":
    main()
