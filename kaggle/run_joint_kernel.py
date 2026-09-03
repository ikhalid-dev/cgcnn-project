#!/usr/bin/env python3
"""
Push the JOINT two-head Kaggle kernel, watch it, and pull the results back.
============================================================================

    python kaggle/run_joint_kernel.py              # push, poll, download
    python kaggle/run_joint_kernel.py --status     # just report where it is
    python kaggle/run_joint_kernel.py --fetch      # just download a finished run
    python kaggle/run_joint_kernel.py --log        # print the running log

Separate from run_kernel.py and run_alignn_kernel.py the same way their build
scripts are separate: its own BUILD_DIR (kaggle/build_joint/) and staging
directory (kaggle/output_joint/), so pushing, polling or fetching this kernel
never touches the other two kernels' state.

WHAT LANDS WHERE
----------------
31_train_joint.py's outputs now carry its own script number rather than a
"joint" tag (script-number-in-filename convention, applied across
scripts/cgcnn/ - see script 31/32/37/38's own headers): the kernel writes
results/cgcnn/model_31_*.pth, history_31_*.csv, predictions_31_*.csv and
summary_31_*.json for each trained member, plus (when 32_ensemble_joint.py
runs) predictions_32_*.csv and summary_32_*.json for the combined ensemble.
Whatever comparison table the run in question builds lands as a top-level
results/*_comparison.csv (e.g. gamma_comparison.csv for a 37_train_gamma.py
round - see build_joint_kernel.py's KERNEL_TEMPLATE for the exact name the
current round writes).

Because the kernel's output already carries the cgcnn/ subdirectory, --fetch
merges it into this project's results/ ROOT - so GPU-trained checkpoints land
in results/cgcnn/ beside the locally trained separate models, and the
comparison CSV lands in results/.

Nothing is overwritten: every file this kernel produces carries its own
script's number (31_, 32_, 37_, ...), so the existing K_VRH_* / G_VRH_*
results from the unrelated 02-08 pipeline are left alone.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build_joint")
STAGING_DIR = os.path.join(PROJECT_ROOT, "kaggle", "output_joint")

TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def parse_state(text):
    """See kaggle/run_kernel.py's identical function for why this can't just
    match the raw string: the enum prefix and case both vary by CLI version."""
    if '"' not in text:
        return text.strip().lower()
    return text.split('"')[1].split(".")[-1].strip().lower()


def kernel_id():
    path = os.path.join(BUILD_DIR, "kernel-metadata.json")
    if not os.path.exists(path):
        sys.exit("No kernel-metadata.json - run "
                 "`python kaggle/build_joint_kernel.py` first.")
    with open(path) as fh:
        return json.load(fh)["id"]


def kaggle(*args, check=True):
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and result.returncode:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return (result.stdout + result.stderr).strip()


def status(kid):
    text = kaggle("kernels", "status", kid, check=False)
    return parse_state(text), text


def push(kid):
    print(f"Pushing {kid}...")
    print(kaggle("kernels", "push", "-p", BUILD_DIR))


def fetch(kid):
    """Download the kernel's output into results/ (it carries cgcnn/ itself)."""
    os.makedirs(STAGING_DIR, exist_ok=True)
    print(f"Downloading output of {kid} -> {STAGING_DIR}")
    print(kaggle("kernels", "output", kid, "-p", STAGING_DIR))

    produced = os.path.join(STAGING_DIR, "results")
    if not os.path.isdir(produced):
        print(f"No results/ in the output. Contents: {sorted(os.listdir(STAGING_DIR))}")
        return

    # results/ itself, NOT results/cgcnn/: the kernel's output already has a
    # cgcnn/ subdirectory inside results/, so walking it below reproduces
    # results/cgcnn/... correctly. Appending "cgcnn" here would nest it twice.
    destination = os.path.join(PROJECT_ROOT, "results")
    copied = 0
    for root, _, files in os.walk(produced):
        rel = os.path.relpath(root, produced)
        dest_root = os.path.join(destination, rel) if rel != "." else destination
        os.makedirs(dest_root, exist_ok=True)
        for name in files:
            shutil.copy2(os.path.join(root, name), os.path.join(dest_root, name))
            copied += 1
    print(f"Copied {copied} files into results/")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="report status and exit")
    parser.add_argument("--fetch", action="store_true", help="download output and exit")
    parser.add_argument("--log", action="store_true", help="print the log and exit")
    parser.add_argument("--poll", type=int, default=120,
                        help="seconds between status checks while waiting")
    # Both targets at full production settings (150 epochs each) on a P100:
    # estimated ~2.5h total, plus a few minutes for deps/data-prep. Sized
    # well above that estimate rather than exactly to it - a --status/--log
    # check costs nothing, a premature timeout just means re-running --fetch
    # by hand once it actually finishes.
    parser.add_argument("--timeout", type=int, default=6 * 3600,
                        help="give up waiting after this many seconds")
    args = parser.parse_args()

    kid = kernel_id()

    if args.status:
        state, raw = status(kid)
        print(raw)
        return
    if args.log:
        print(kaggle("kernels", "output", kid, "-p", STAGING_DIR))
        return
    if args.fetch:
        fetch(kid)
        return

    push(kid)

    start = time.time()
    last = None
    while time.time() - start < args.timeout:
        state, raw = status(kid)
        if state != last:
            print(f"[{(time.time() - start) / 60:5.1f} min] {raw}")
            last = state
        if state in TERMINAL:
            break
        time.sleep(args.poll)
    else:
        sys.exit(f"Still running after {args.timeout / 3600:.1f} h - "
                 f"check with --status.")

    if state == "error":
        sys.exit(f"Kernel failed. Inspect the log:\n"
                 f"  https://www.kaggle.com/code/{kid}")

    fetch(kid)
    print("\nresults/cgcnn/model_37_r9_full_s*.pth and summary_37_r9_full_s*.json "
         "are ready (one triplet per seed), plus results/gamma_comparison.csv "
         "comparing each against the round-7 1,022-crystal baseline.")


if __name__ == "__main__":
    main()
