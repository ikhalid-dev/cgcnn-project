#!/usr/bin/env python3
"""
Remote-side driver for training ALIGNN on a Colab GPU.
========================================================

Shared by two launch paths so they can never drift apart: `PINK_ALIGNN_colab.py`
embeds this file's source as a notebook cell, and `run_alignn_cli.py` sends it
directly via `colab exec -f`. Either way, this same code runs the same steps.

WHY THE TRAINING LOOP RUNS DETACHED, NOT INLINE
-------------------------------------------------
`colab exec -f` defaults to a 30s client-side timeout (`colab exec --help`),
and even an explicit longer --timeout only bounds THIS client's wait - it does
not kill the remote kernel (colab-cli's own runtime.py notes a client timeout
"does not interrupt the kernel"). A two-target, 150-epoch ALIGNN run has no
existing timing benchmark (only ever smoke-tested for 2-3 epochs locally on
CPU) and could plausibly run for hours - far past any reasonable exec timeout.

So this script does its BOUNDED work synchronously (GPU check, then
scripts/11_prepare_alignn_data.py - minutes, not hours), then launches the
actual two-target training as a fully separate, detached OS process and
returns almost immediately.

That's a real subprocess.Popen, deliberately not os.fork(): when launched via
`colab exec -f`, this code runs INSIDE the remote Jupyter kernel's own
long-lived process (an async/zmq-based process). Forking a running
kernel is exactly the kind of thing that corrupts inherited event-loop and
socket state in the child - Popen with a fresh interpreter avoids that
entirely, at the cost of needing a real file on disk to launch (see below).

WHY THE WORKER RE-INVOKES A DISK PATH, NOT __file__
-------------------------------------------------------
`colab exec -f` transmits this file's CONTENT to be executed as a Jupyter
cell - it is not run as a script from disk, so `__file__` is unreliable
inside the primary invocation. The worker re-launch therefore uses a fixed,
cwd-relative path (`colab/alignn_gpu_driver.py`) rather than `__file__` -
which means this file must ALSO be included in the project bundle uploaded
to the VM (see BUNDLE in PINK_ALIGNN_colab.py), so a real copy exists on disk
at that path when the Popen call needs to re-run it with `--worker`.

WHY cwd, NOT A COMPUTED PROJECT_ROOT
----------------------------------------
For the same reason - no reliable `__file__` when exec'd as a cell - this
script trusts that an earlier, separate `colab exec` call in the same
session already unzipped the bundle and `os.chdir()`'d into it (a Jupyter
kernel's cwd persists across separate exec calls in one session, since it's
one continuously-running process, not a fresh interpreter each time). A
sanity check aborts loudly if that assumption is wrong, rather than failing
confusingly deep inside scripts/11 or 12.

STATUS FILE CONTRACT
----------------------
Written to STATUS_PATH (JSON) after every state change, so a short, cheap
`colab download` of one small file (from run_alignn_cli.py's --status/--wait)
can always answer "how far along is this" without touching the long-running
process itself or needing a live exec connection held open for hours.
"""

import json
import os
import subprocess
import sys
import time
import zipfile

# Fixed extraction path both consumers use (the notebook's own os.chdir() in
# its step 3; run_alignn_cli.py's unpack step) - chdir here immediately, as
# the first thing this module does, rather than trust incoming cwd.
#
# Necessary for the CLI path specifically: verified directly (two separate
# `colab exec` calls to the SAME named session, one chdir'ing and printing
# os.getcwd(), the next just printing it again) that cwd does NOT persist
# between separate exec calls - the second call still saw /content, not
# wherever the first one chdir'd to. Confirmed this is cwd-specific, not "a
# fresh kernel every time": the identical experiment with os.environ instead
# of os.chdir() showed the env var DID persist across the same two calls.
# So each call gets a real hard reset of cwd specifically, for reasons this
# project doesn't need to fully explain to work around.
#
# Harmless no-op for the notebook path: a real Jupyter notebook's cells all
# share one persistent kernel where cwd persists completely normally, so by
# the time this code runs there it's already exactly BUNDLE_ROOT.
BUNDLE_ROOT = "/content/pink_alignn"
if not os.path.isdir(BUNDLE_ROOT):
    sys.exit(f"{BUNDLE_ROOT} doesn't exist - was the bundle actually "
             f"uploaded and unzipped before this ran?")
os.chdir(BUNDLE_ROOT)
if not os.path.exists(os.path.join(BUNDLE_ROOT, "scripts", "12_train_alignn.py")):
    sys.exit(f"{BUNDLE_ROOT} exists but scripts/12_train_alignn.py is missing "
             f"from it - the bundle looks incomplete.")

RESULTS_DIR = os.path.join(os.getcwd(), "results")
STATUS_PATH = os.path.join(os.getcwd(), "colab", "training_status.json")
LOG_PATH = os.path.join(os.getcwd(), "colab", "training.log")
ZIP_PATH = os.path.join(os.getcwd(), "colab", "alignn_results.zip")
WORKER_SCRIPT = os.path.join(os.getcwd(), "colab", "alignn_gpu_driver.py")

TARGETS = ("bulk_modulus_kv", "shear_modulus_gv")

# ALIGNN_SMOKE_TEST=1 swaps in scripts/12_train_alignn.py's own existing
# small-scale flags (matching how it was smoke-tested locally on CPU before
# any of this existed) instead of the full production hyperparameters -
# there's no argv available to the primary (colab-exec'd) invocation to pass
# a --smoke-test flag through normally, so run_alignn_cli.py's --smoke-test
# sets this env var via a preliminary exec call instead; it's inherited by
# the worker subprocess automatically since os.environ carries through.
if os.environ.get("ALIGNN_SMOKE_TEST") == "1":
    COMMON_ARGS = ["--epochs", "2", "--batch-size", "8", "--alignn-layers", "1",
                  "--gcn-layers", "1", "--hidden-features", "32",
                  "--embedding-features", "16", "--n-train", "200", "--n-val", "40",
                  "--n-test", "40", "--device", "cuda"]
else:
    COMMON_ARGS = ["--epochs", "150", "--batch-size", "64", "--learning-rate", "0.001",
                  "--alignn-layers", "4", "--gcn-layers", "4", "--hidden-features", "256",
                  "--embedding-features", "64", "--n-early-stopping", "30", "--device", "cuda"]


def write_status(state):
    state["updated"] = time.time()
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w") as fh:
        json.dump(state, fh, indent=2)


def check_gpu():
    """Abort before touching scripts/11 or 12 if there's no GPU visible.

    Not hypothetical: kaggle/build_kernel.py's own comment documents hitting
    exactly this failure class already on a different GPU runner -
    "enable_gpu ALONE IS NOT ENOUGH... accepted and silently ignored, and
    the kernel lands on the CPU image." Checking again here is informed by
    that prior incident, not generic caution.
    """
    import torch
    if not torch.cuda.is_available():
        write_status({"stage": "failed",
                     "error": "no CUDA device visible - the session landed "
                              "on a CPU image despite requesting a GPU"})
        sys.exit("No GPU visible to torch. Aborting before touching scripts/11 or 12.")
    print(f"GPU OK: {torch.cuda.get_device_name(0)}", flush=True)


def run_data_prep():
    write_status({"stage": "data_prep", "targets": {t: "pending" for t in TARGETS}})
    result = subprocess.run([sys.executable, "-u",
                            os.path.join("scripts", "11_prepare_alignn_data.py")])
    if result.returncode:
        write_status({"stage": "failed",
                     "error": f"data prep failed (exit {result.returncode})"})
        sys.exit(f"scripts/11_prepare_alignn_data.py failed (exit {result.returncode})")


def zip_results(state):
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for target in TARGETS:
            out_dir = os.path.join(RESULTS_DIR, f"alignn_{target}")
            if not os.path.isdir(out_dir):
                continue
            for root, _, files in os.walk(out_dir):
                for name in files:
                    full = os.path.join(root, name)
                    archive.write(full, os.path.relpath(full, RESULTS_DIR))

    n_ok = sum(1 for t in TARGETS if state["targets"].get(t) == "complete")
    state["stage"] = "complete" if n_ok == len(TARGETS) else ("partial" if n_ok else "failed")
    state["zip_path"] = ZIP_PATH
    write_status(state)
    print(f"Wrote {ZIP_PATH} ({n_ok}/{len(TARGETS)} targets succeeded)", flush=True)


def train_all_targets():
    """The actual multi-hour work.

    No stdout/stderr redirection here - subprocess.run(args) with no
    stdout=/stderr= simply inherits THIS process's own streams, and that is
    exactly right for both callers: run synchronously from a notebook cell,
    "this process's stdout" is the cell itself, so output streams live;
    run inside the detached --worker process, main()'s own Popen call
    already redirected that whole process's stdout (and merged stderr into
    it) to LOG_PATH before this function is ever reached, so the inheritance
    cascades there instead. No caller has to remember to pass anything.

    Sequential, not parallel: a single T4's VRAM is shared between whatever
    runs on it, and the original notebook already trains one target at a
    time. One target failing does not block the other - unlike
    PINK_ALIGNN_colab.py's OLD run() helper (now replaced by this function),
    which SystemExits on the first failure. That was correct for a human
    watching cell-by-cell but silently hid the real traceback and meant one
    bad target took the whole run down - exactly the bug report that led
    here: the notebook printed only "SystemExit: FAILED (exit 1)" with none
    of the actual Python traceback that would explain why.
    """
    state = {"stage": "training", "targets": {t: "pending" for t in TARGETS}}
    write_status(state)

    for target in TARGETS:
        state["targets"][target] = "running"
        state[f"{target}_started"] = time.time()
        write_status(state)

        out_dir = os.path.join(RESULTS_DIR, f"alignn_{target}")
        args = ([sys.executable, "-u", os.path.join("scripts", "12_train_alignn.py"),
                "--target", target, "--out-dir", out_dir] + COMMON_ARGS)
        # If run_alignn_cli.py re-uploaded a checkpoint from a previous,
        # interrupted attempt at this target (see its sync_checkpoints()),
        # it's sitting in out_dir already at this point - resume from it
        # instead of burning epochs 0..N again. --resume is a harmless
        # no-op if there's nothing there yet (first attempt at this target).
        if os.path.exists(os.path.join(out_dir, "current_model.pt")):
            args.append("--resume")
            print(f"Found an existing checkpoint for {target} - resuming.", flush=True)
        print(f"$ {' '.join(args)}", flush=True)
        result = subprocess.run(args)

        state["targets"][target] = "complete" if result.returncode == 0 else "failed"
        state[f"{target}_finished"] = time.time()
        write_status(state)
        if result.returncode:
            print(f"!! {target} failed (exit {result.returncode}) - see the "
                 f"traceback above. Continuing to the next target.", flush=True)

    zip_results(state)


def main():
    if "--worker" in sys.argv:
        # Only reachable via our own Popen call below, never via `colab exec
        # -f` (which cannot forward argv) - so this branch is unambiguous.
        train_all_targets()
        return

    check_gpu()
    run_data_prep()

    log_fh = open(LOG_PATH, "w")
    subprocess.Popen(
        [sys.executable, "-u", WORKER_SCRIPT, "--worker"],
        stdout=log_fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log_fh.close()  # the child has its own duplicated fd; don't leak ours
    write_status({"stage": "launched", "targets": {t: "pending" for t in TARGETS}})
    print(f"Training launched in the background. Poll {STATUS_PATH} for progress.",
         flush=True)


if __name__ == "__main__":
    main()
