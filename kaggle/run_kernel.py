#!/usr/bin/env python3
"""
Push the Kaggle kernel, watch it, and pull the results back.
============================================================

    python kaggle/run_kernel.py              # push, poll, download
    python kaggle/run_kernel.py --status     # just report where it is
    python kaggle/run_kernel.py --fetch      # just download a finished run
    python kaggle/run_kernel.py --log        # print the running log

This is the whole reason for using Kaggle over Colab: the entire loop is
scriptable, so nobody has to sit watching a browser tab.

WHAT LANDS WHERE
----------------
Kaggle captures whatever the kernel leaves in /kaggle/working. `--fetch` pulls
that into `results/`, next to the models trained locally, so the tags stay
distinct (`K_VRH_full`, `K_VRH_s1`, ...) and nothing is overwritten by accident.

After fetching, finish the pipeline on this machine:

    python scripts/cgcnn/04_predict_moduli.py \\
        --k-tag K_VRH_full,K_VRH_s1,K_VRH_s2 \\
        --g-tag G_VRH_full,G_VRH_s1,G_VRH_s2

That step stays local because it needs the 1,213 CIFs, which are never uploaded.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "kaggle", "build")

# Kaggle reports these; anything else means the run is still going.
TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def parse_state(text):
    """Pull the bare status word out of whatever the CLI printed.

    The CLI says: has status "KernelWorkerStatus.RUNNING". Both the enum prefix
    and the case vary between kaggle versions, so strip to the last dotted
    component and lowercase it - matching the raw string against TERMINAL would
    never fire and the poll loop would spin until it timed out.
    """
    if '"' not in text:
        return text.strip().lower()
    return text.split('"')[1].split(".")[-1].strip().lower()


def kernel_id():
    """Read the kernel id from the metadata the build step wrote."""
    path = os.path.join(BUILD_DIR, "kernel-metadata.json")
    if not os.path.exists(path):
        sys.exit("No kernel-metadata.json - run `python kaggle/build_kernel.py` first.")
    with open(path) as fh:
        return json.load(fh)["id"]


def kaggle(*args, check=True):
    """Call the kaggle CLI and return its stdout."""
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and result.returncode:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return (result.stdout + result.stderr).strip()


def status(kid):
    """Return the kernel's current status string, lowercased."""
    text = kaggle("kernels", "status", kid, check=False)
    return parse_state(text), text


def push(kid):
    print(f"Pushing {kid}...")
    print(kaggle("kernels", "push", "-p", BUILD_DIR))


def fetch(kid):
    """Download the kernel's output into results/cgcnn/."""
    staging = os.path.join(PROJECT_ROOT, "kaggle", "output")
    os.makedirs(staging, exist_ok=True)
    print(f"Downloading output of {kid} -> {staging}")
    print(kaggle("kernels", "output", kid, "-p", staging))

    # The kernel writes results/ inside its output; merge that into the
    # project's results/cgcnn/ so GPU-trained checkpoints sit beside local ones.
    produced = os.path.join(staging, "results")
    if not os.path.isdir(produced):
        print(f"No results/ in the output. Contents: {sorted(os.listdir(staging))}")
        return

    # results/cgcnn/, not results/ - this project's own scripts/results split
    # by architecture (this kernel only ever trains CGCNN).
    destination = os.path.join(PROJECT_ROOT, "results", "cgcnn")
    os.makedirs(destination, exist_ok=True)
    copied = 0
    for name in sorted(os.listdir(produced)):
        shutil.copy2(os.path.join(produced, name), os.path.join(destination, name))
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
    parser.add_argument("--timeout", type=int, default=10800,
                        help="give up waiting after this many seconds")
    args = parser.parse_args()

    kid = kernel_id()

    if args.status:
        state, raw = status(kid)
        print(raw)
        return
    if args.log:
        print(kaggle("kernels", "output", kid, "-p",
                     os.path.join(PROJECT_ROOT, "kaggle", "output")))
        return
    if args.fetch:
        fetch(kid)
        return

    push(kid)

    # --- Poll until it finishes ------------------------------------------
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
    print("\nNow finish the pipeline locally:")
    print("  python scripts/cgcnn/04_predict_moduli.py \\")
    print("      --k-tag K_VRH_full,K_VRH_s1,K_VRH_s2 \\")
    print("      --g-tag G_VRH_full,G_VRH_s1,G_VRH_s2")


if __name__ == "__main__":
    main()
