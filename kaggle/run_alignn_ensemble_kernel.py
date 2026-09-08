#!/usr/bin/env python3
"""
Push the ALIGNN Kaggle kernel, watch it, and pull the results back.
=====================================================================

    python kaggle/run_alignn_kernel.py              # push, poll, download
    python kaggle/run_alignn_kernel.py --status     # just report where it is
    python kaggle/run_alignn_kernel.py --fetch      # just download a finished run
    python kaggle/run_alignn_kernel.py --log        # print the running log

Separate from kaggle/run_kernel.py the same way kaggle/build_alignn_kernel.py
is separate from build_kernel.py: a different BUILD_DIR (kaggle/build_alignn/)
and staging directory (kaggle/output_alignn/), so pushing/polling/fetching
this kernel never touches the CGCNN kernel's own state.

WHAT LANDS WHERE
----------------
Both results/alignn/alignn_bulk_modulus_kv/ and
results/alignn/alignn_shear_modulus_gv/ land directly under results/alignn/
on fetch - same layout colab/run_alignn_cli.py's --fetch already produces, so
whichever platform actually finishes the run, the rest of the project (docs,
the presentation, scripts/alignn/15/16) reads the same paths either way.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# One staging dir per seed, matching build_alignn_ensemble_kernel.py -
# sharing one would let a push clobber another seed's metadata.
SEED = int(os.environ.get("ALIGNN_SEED", "42"))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", f"build_alignn_s{SEED}")
STAGING_DIR = os.path.join(PROJECT_ROOT, "kaggle", "output_alignn")

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
                 "`python kaggle/build_alignn_kernel.py` first.")
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
    """Download the kernel's output into results/alignn/."""
    os.makedirs(STAGING_DIR, exist_ok=True)
    print(f"Downloading output of {kid} -> {STAGING_DIR}")
    print(kaggle("kernels", "output", kid, "-p", STAGING_DIR))

    produced = os.path.join(STAGING_DIR, "results")
    if not os.path.isdir(produced):
        print(f"No results/ in the output. Contents: {sorted(os.listdir(STAGING_DIR))}")
        return

    # results/alignn/, not results/ - this project's own scripts/results
    # split by architecture. The downloaded kernel's own "results/" output
    # structure above is unaffected, still flat.
    destination = os.path.join(PROJECT_ROOT, "results", "alignn")
    copied = 0
    for root, _, files in os.walk(produced):
        rel = os.path.relpath(root, produced)
        dest_root = os.path.join(destination, rel) if rel != "." else destination
        os.makedirs(dest_root, exist_ok=True)
        for name in files:
            shutil.copy2(os.path.join(root, name), os.path.join(dest_root, name))
            copied += 1
    print(f"Copied {copied} files into results/alignn/")


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
    print("\nresults/alignn/alignn_bulk_modulus_kv/ and "
         "results/alignn/alignn_shear_modulus_gv/ are ready - each has "
         "best_model.pt, config.json, and prediction_results_test_set.csv.")


if __name__ == "__main__":
    main()
