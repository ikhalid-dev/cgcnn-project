#!/usr/bin/env python3
"""
Push the matbench subset size test kernel to Kaggle, watch it, and pull it back.
================================================================================

    python kaggle/build_matbench_subset_kernel.py     # build first
    python kaggle/run_direct_kappa_kernel.py       # push, poll, download
    python kaggle/run_direct_kappa_kernel.py --status
    python kaggle/run_direct_kappa_kernel.py --log
    python kaggle/run_direct_kappa_kernel.py --fetch

WHAT COMES BACK, AND WHERE IT LANDS
-------------------------------------
    models/*.pth, models/*.json    -> direct_kappa_no_slack/models/
    predictions/*.csv              -> direct_kappa_no_slack/results/csv/
    direct_vs_control_summary.csv  -> direct_kappa_no_slack/results/csv/

Both arms come back: the three direct-kappa runs and the three in-session
control runs (37_train_gamma.py, the round-9 configuration). 02 then scores
them locally against AFLOW's own kappa.

TWO KAGGLE BEHAVIOURS THAT LOOK LIKE BUGS AND ARE NOT
--------------------------------------------------------
1. `--log` while a run is IN FLIGHT returns the PREVIOUS version's log.
   Kaggle only serves logs for completed versions. There is no live progress;
   a stale log is not evidence that this run is doing anything.
2. Wall-clock can be four to six times compute time because of queueing. A
   ~45-minute job has sat at RUNNING for six hours before now. The default
   --timeout below is 6 h and it WILL exit non-zero at that point even though
   the kernel is perfectly healthy - just re-run with --status and --fetch.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build_matbench_subset")
STAGING_DIR = os.path.join(PROJECT_ROOT, "kaggle", "output_matbench_subset")

# Where each of the kernel's output subdirectories is copied to locally.
DEST = {
    "models": os.path.join(PROJECT_ROOT, "direct_kappa_no_slack", "models"),
    "predictions": os.path.join(PROJECT_ROOT, "direct_kappa_no_slack", "results", "csv"),
}

TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def parse_state(text):
    """Extract the state word from whatever shape this CLI version prints.

    The enum prefix and the case both vary between kaggle CLI releases, so
    matching the raw string breaks silently on upgrade.
    """
    if '"' not in text:
        return text.strip().lower()
    return text.split('"')[1].split(".")[-1].strip().lower()


def kernel_id():
    path = os.path.join(BUILD_DIR, "kernel-metadata.json")
    if not os.path.exists(path):
        sys.exit("No kernel-metadata.json - run "
                 "`python kaggle/build_matbench_subset_kernel.py` first.")
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
    """Download the kernel output and file each piece where 02 expects it."""
    os.makedirs(STAGING_DIR, exist_ok=True)
    print(f"Downloading output of {kid} -> {STAGING_DIR}")
    print(kaggle("kernels", "output", kid, "-p", STAGING_DIR))

    copied = 0
    for sub, destination in DEST.items():
        produced = os.path.join(STAGING_DIR, sub)
        if not os.path.isdir(produced):
            continue
        os.makedirs(destination, exist_ok=True)
        for name in sorted(os.listdir(produced)):
            src = os.path.join(produced, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(destination, name))
                copied += 1

    # The one-table summary the kernel assembles sits at the top level.
    top = os.path.join(STAGING_DIR, "direct_vs_control_summary.csv")
    if os.path.exists(top):
        os.makedirs(DEST["predictions"], exist_ok=True)
        shutil.copy2(top, os.path.join(DEST["predictions"],
                                       "direct_vs_control_summary.csv"))
        copied += 1

    if not copied:
        print(f"Nothing filed. Staging contents: {sorted(os.listdir(STAGING_DIR))}")
    else:
        print(f"Copied {copied} files into direct_kappa_no_slack/")
        print("Next:  python direct_kappa_no_slack/55_matbench_subset_verdict.py")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="report status and exit")
    parser.add_argument("--fetch", action="store_true", help="download output and exit")
    parser.add_argument("--log", action="store_true",
                        help="print the log and exit (COMPLETED versions only)")
    parser.add_argument("--poll", type=int, default=120,
                        help="seconds between status checks (default 120)")
    parser.add_argument("--timeout", type=int, default=6 * 3600,
                        help="give up waiting after this many seconds (default 6h); "
                             "the kernel keeps running on Kaggle regardless")
    args = parser.parse_args()

    kid = kernel_id()

    if args.status:
        state, text = status(kid)
        print(f"{kid}: {state}\n{text}")
        return
    if args.log:
        print(kaggle("kernels", "output", kid, "-p", STAGING_DIR, check=False))
        log = os.path.join(STAGING_DIR, "pink_mb_subset.log")
        if os.path.exists(log):
            with open(log) as fh:
                print(fh.read())
        else:
            print("No log yet - Kaggle only serves logs for COMPLETED versions.")
        return
    if args.fetch:
        fetch(kid)
        return

    push(kid)
    print(f"Polling every {args.poll}s (timeout {args.timeout // 3600}h). "
          f"Wall-clock is queue-dominated and unpredictable.")
    start = time.time()
    while True:
        state, _ = status(kid)
        elapsed = int(time.time() - start)
        print(f"  [{elapsed:>6}s] {state}", flush=True)
        if state in TERMINAL:
            break
        if time.time() - start > args.timeout:
            sys.exit(f"Timed out locally after {args.timeout}s. The kernel is "
                     f"probably still running - check with --status, then --fetch.")
        time.sleep(args.poll)

    if state != "complete":
        sys.exit(f"Kernel finished in state '{state}'. Check the log with --log.")
    fetch(kid)


if __name__ == "__main__":
    main()
