#!/usr/bin/env python3
"""
Drive the ALIGNN GPU training run from here, via google-colab-cli.
======================================================================

    python colab/run_alignn_cli.py --launch [--gpu T4]   # provision + start
    python colab/run_alignn_cli.py --status              # how far along?
    python colab/run_alignn_cli.py --wait                # poll until done
    python colab/run_alignn_cli.py --log                  # tail the training log
    python colab/run_alignn_cli.py --fetch                # download + merge results
    python colab/run_alignn_cli.py --stop                 # tear down the session

This replaces the manual "upload colab/PINK_ALIGNN.ipynb, click through
Colab's UI" workflow with something scriptable end to end, using
`google-colab-cli` (github.com/googlecolab/google-colab-cli - the real
upstream of the fork this was built from; installing the published PyPI
package rather than the fork, since it's a byte-identical copy).

REQUIRES A ONE-TIME HUMAN STEP FIRST, NOT SOMETHING THIS SCRIPT CAN DO
---------------------------------------------------------------------------
`colab --auth oauth2 whoami` needs to have been run once, by a person, in
their own terminal: it prints a Google sign-in URL, the human signs in and
pastes back a code. That's a hard technical wall, not a policy one - there is
no browser or Google identity available to automate it from here. After that
one login, the refresh token at ~/.config/colab-cli/token.json makes every
`colab` command (including every one this script runs) fully non-interactive.

WHY colab new (a persistent, named session), NOT colab run
----------------------------------------------------------
`colab run SCRIPT` provisions a FRESH VM per call and tears it down after -
fine for a single short script, wrong here: calling it once per target would
mean two VMs, each redoing the multi-minute alignn/jarvis-tools/pymatgen/
matminer install and the matbench download. One named session
(SESSION_NAME below), reused across install/upload/exec/download/stop calls,
backed by the CLI's own keep-alive daemon (survives Colab's ~90-min idle
pruning), is the right primitive - see colab/alignn_gpu_driver.py's own
docstring for why the actual multi-hour training then still has to run
detached rather than inline in any single `colab exec` call.

WHY colab exec -f isn't enough for scripts/12 directly
------------------------------------------------------------
`colab exec -f FILE` reads FILE from THIS machine and transmits its content
to run as a Jupyter cell - it does not forward extra CLI arguments the way
`colab run SCRIPT ARGS...` does (confirmed against the tool's own test suite
naming, e.g. test_run_passes_argv - exec has no equivalent). So
scripts/12_train_alignn.py's --target/--epochs/etc. are never passed this
way; colab/alignn_gpu_driver.py builds that argv itself via subprocess,
exactly like PINK_ALIGNN_colab.py's own (now-retired) run() helper did.

WHAT COUNTS AS "DONE" HERE VS. THE OLD MANUAL PATH
-------------------------------------------------------
Not deleting colab/PINK_ALIGNN_colab.py or colab/PINK_ALIGNN.ipynb - kept as
a fallback (this tool works by mimicking Colab's own undocumented internal
endpoints, per its own design docs; it could break on a Colab-side change,
same category of risk already hit once in this project with ALIGNN's own
DGL claim not matching its installed behaviour). Both paths now share the
same colab/alignn_gpu_driver.py, so they can't silently drift apart.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, "colab", "build")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "colab", "alignn_output")
SESSION_NAME = "pink-alignn"

# `colab download`/`upload` go through the Jupyter Contents API
# (docs/03_file_management.md: GET/PUT /api/contents/<path>), which is
# rooted at the Jupyter SERVER's own root (/content on Colab) - completely
# independent of any one kernel's os.chdir() state. A bare "colab/foo.json"
# would resolve to /content/colab/foo.json, not where the bundle actually
# unpacked to - caught by an empty --status result immediately after a
# driver launch that had, in fact, succeeded. Every remote path below is
# rooted here explicitly instead of assuming chdir carries over.
REMOTE_ROOT = "/content/pink_alignn"

sys.path.insert(0, os.path.join(PROJECT_ROOT, "colab"))
from PINK_ALIGNN_colab import BUNDLE  # noqa: E402  (reuse the file list, don't re-list it)

TERMINAL_STAGES = {"complete", "partial", "failed"}


def colab_cli(*args, input=None, check=True):
    """Call the colab-cli binary. --auth oauth2 pinned explicitly everywhere:
    the tool's own README and its design docs disagree about which provider
    defaults, so this never relies on either being right.
    """
    result = subprocess.run(
        ["colab", "--auth", "oauth2", *args],
        input=input, capture_output=True, text=True)
    if check and result.returncode:
        sys.exit(f"colab {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def session_exists():
    result = colab_cli("sessions", check=False)
    return SESSION_NAME in result.stdout


def build_bundle_zip():
    os.makedirs(BUILD_DIR, exist_ok=True)
    dest = os.path.join(BUILD_DIR, "bundle.zip")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in BUNDLE:
            path = os.path.join(PROJECT_ROOT, relative)
            if not os.path.exists(path):
                sys.exit(f"missing bundle file: {relative}")
            archive.write(path, relative)
    return dest


def show(result):
    """Print both streams, always - a driver-side sys.exit() inside an
    exec'd Jupyter cell is reported by IPython as a normal error, not
    necessarily a nonzero colab-cli process exit code, and it isn't always
    on stdout. Printing only .stdout (an earlier version of this function)
    silently swallowed exactly that failure once already - caught only by
    manually re-running the same exec code directly and inspecting the VM.
    """
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    return result


def launch(gpu, smoke_test=False):
    if session_exists():
        print(f"Session '{SESSION_NAME}' already exists - reusing it rather "
             f"than risking a second, concurrently-billed VM.")
        print("(If this is stale/broken, `--stop` it first, then relaunch.)")
    else:
        print(f"Provisioning a new session '{SESSION_NAME}' (--gpu {gpu})...")
        result = show(colab_cli("new", "-s", SESSION_NAME, "--gpu", gpu, check=False))
        if result.returncode:
            sys.exit(f"colab new failed (exit {result.returncode}).\n\n"
                     f"If this is GPU quota/availability, retry with a different "
                     f"--gpu rather than assuming escalating tiers are free.")

    print("\nInstalling dependencies (pymatgen, matminer, alignn)...")
    show(colab_cli("install", "-s", SESSION_NAME, "-r",
                   os.path.join(PROJECT_ROOT, "colab", "alignn_requirements.txt")))

    print("\nBuilding and uploading the project bundle...")
    bundle_path = build_bundle_zip()
    print(f"  {bundle_path} ({os.path.getsize(bundle_path) / 1024:.0f} KB)")
    show(colab_cli("upload", "-s", SESSION_NAME, bundle_path, f"{REMOTE_ROOT}_bundle.zip"))

    print("\nUnpacking on the VM...")
    unpack_code = (
        "import zipfile, os\n"
        f"zipfile.ZipFile('{REMOTE_ROOT}_bundle.zip').extractall('{REMOTE_ROOT}')\n"
        "print('unpacked')\n"
    )
    # No os.chdir() here - verified directly that colab exec's cwd does not
    # persist to the NEXT separate exec call anyway (see
    # alignn_gpu_driver.py's own docstring for the experiment), so the
    # driver chdirs to REMOTE_ROOT itself, unconditionally, as its own first
    # action. This call only needs to get the files onto disk, which does
    # persist.
    show(colab_cli("exec", "-s", SESSION_NAME, "--timeout", "60", input=unpack_code))

    # Always set this explicitly, in both directions - os.environ changes
    # DO persist across separate exec calls to the same session (unlike
    # cwd, see above), so a session reused from an earlier --smoke-test
    # launch would otherwise silently keep training with smoke-test-sized
    # data forever. Caught live: a "real" launch right after a smoke-test
    # one on the same session finished in 0.2 minutes instead of hours,
    # because this exact leak was still unset at the time.
    flag = "1" if smoke_test else "0"
    if smoke_test:
        print("\n--smoke-test: using scripts/12's small-scale flags (2 epochs, "
             "~200 train crystals) instead of the full production run.")
    show(colab_cli("exec", "-s", SESSION_NAME, "--timeout", "30",
                   input=f"import os\nos.environ['ALIGNN_SMOKE_TEST'] = '{flag}'\n"
                         f"print('ALIGNN_SMOKE_TEST set to {flag}')\n"))

    print("\nRunning the driver: GPU check, data prep, then launching training "
         "in the background (this step can take a few minutes - matbench "
         "download + conversion)...")
    driver_path = os.path.join(PROJECT_ROOT, "colab", "alignn_gpu_driver.py")
    result = show(colab_cli("exec", "-s", SESSION_NAME, "-f", driver_path,
                            "--timeout", "600", check=False))
    if result.returncode:
        sys.exit(f"Driver launch failed (exit {result.returncode}) - see output above.\n\n"
                 f"The session is still up - inspect with --status/--log, "
                 f"`colab exec -s {SESSION_NAME}`, or `colab console -s {SESSION_NAME}` "
                 f"before deciding whether to --stop it.")

    print(f"\nLaunched. Poll with --status or --wait; session '{SESSION_NAME}' "
         f"stays up regardless (never auto-stopped on failure - inspect first).")


def fetch_remote_json(remote_path):
    """Download one small file and parse it as JSON, or return None if it
    doesn't exist yet (e.g. queried before the driver has written anything).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_path = os.path.join(OUTPUT_DIR, os.path.basename(remote_path))
    result = colab_cli("download", "-s", SESSION_NAME, remote_path, local_path, check=False)
    if result.returncode or not os.path.exists(local_path):
        return None
    with open(local_path) as fh:
        return json.load(fh)


def print_status(state):
    if state is None:
        print("No status file yet - the session may still be installing "
             "dependencies or running data prep. Try again shortly.")
        return
    stage = state.get("stage", "unknown")
    print(f"Stage: {stage}")
    for target, target_state in state.get("targets", {}).items():
        started = state.get(f"{target}_started")
        finished = state.get(f"{target}_finished")
        elapsed = ""
        if started and finished:
            elapsed = f" ({(finished - started) / 60:.1f} min)"
        elif started:
            elapsed = f" ({(time.time() - started) / 60:.1f} min so far)"
        print(f"  {target}: {target_state}{elapsed}")
    if state.get("error"):
        print(f"  error: {state['error']}")


def wait(poll_seconds, timeout_seconds):
    start = time.time()
    last_stage = None
    while time.time() - start < timeout_seconds:
        state = fetch_remote_json(f"{REMOTE_ROOT}/colab/training_status.json")
        stage = (state or {}).get("stage")
        if stage != last_stage:
            print(f"[{(time.time() - start) / 60:5.1f} min]", end=" ")
            print_status(state)
            last_stage = stage
        if stage in TERMINAL_STAGES:
            return state
        time.sleep(poll_seconds)
    sys.exit(f"Still running after {timeout_seconds / 3600:.1f}h - check with --status.")


def fetch_results():
    state = fetch_remote_json(f"{REMOTE_ROOT}/colab/training_status.json")
    if state is None or state.get("stage") not in TERMINAL_STAGES:
        print("Not finished yet (or status unreadable) - check --status first. "
             "Fetching whatever exists anyway, in case it's partially useful.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_zip = os.path.join(OUTPUT_DIR, "alignn_results.zip")
    result = colab_cli("download", "-s", SESSION_NAME, f"{REMOTE_ROOT}/colab/alignn_results.zip",
                       local_zip, check=False)
    if result.returncode or not os.path.exists(local_zip):
        sys.exit(f"Could not download colab/alignn_results.zip - "
                 f"has training finished? {result.stderr}")

    extract_dir = os.path.join(OUTPUT_DIR, "extracted")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(local_zip) as archive:
        archive.extractall(extract_dir)

    # alignn_gpu_driver.py's zip_results() writes archive names relative to
    # RESULTS_DIR itself (os.path.relpath(full, RESULTS_DIR)), so the zip's
    # top level IS alignn_<target>/... directly - no results/ wrapper folder
    # the way kaggle/run_kernel.py's kernel output has. Confirmed by
    # actually running this against a real downloaded zip, not assumed by
    # analogy with the Kaggle runner's different convention.
    entries = os.listdir(extract_dir)
    if not any(name.startswith("alignn_") for name in entries):
        sys.exit(f"No alignn_<target> directories inside the downloaded zip. "
                 f"Contents: {sorted(entries)}")

    destination = os.path.join(PROJECT_ROOT, "results")
    copied = 0
    for root, _, files in os.walk(extract_dir):
        for name in files:
            src = os.path.join(root, name)
            rel = os.path.relpath(src, extract_dir)
            dst = os.path.join(destination, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} files into results/")
    for target in ("bulk_modulus_kv", "shear_modulus_gv"):
        marker = os.path.join(destination, f"alignn_{target}", "Test_results.json")
        print(f"  {'OK' if os.path.exists(marker) else 'MISSING'}: "
             f"results/alignn_{target}/Test_results.json")


def show_log():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    local_log = os.path.join(OUTPUT_DIR, "training.log")
    result = colab_cli("download", "-s", SESSION_NAME, f"{REMOTE_ROOT}/colab/training.log",
                       local_log, check=False)
    if result.returncode or not os.path.exists(local_log):
        print("No training.log yet (data prep may still be running, before "
             "the training subprocess has produced any output).")
        return
    with open(local_log) as fh:
        print(fh.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch", action="store_true", help="provision + start training")
    parser.add_argument("--gpu", default="T4", help="GPU tier for --launch (default T4)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="with --launch: 2 epochs / ~200 crystals, to verify the "
                             "whole pipeline before trusting a multi-hour real run")
    parser.add_argument("--status", action="store_true", help="report status and exit")
    parser.add_argument("--wait", action="store_true", help="poll until finished")
    parser.add_argument("--poll", type=int, default=120, help="seconds between --wait polls")
    parser.add_argument("--timeout", type=int, default=6 * 3600,
                        help="give up --wait-ing after this many seconds")
    parser.add_argument("--log", action="store_true", help="print the training log and exit")
    parser.add_argument("--fetch", action="store_true", help="download + merge results")
    parser.add_argument("--stop", action="store_true", help="tear down the session")
    args = parser.parse_args()

    if args.stop:
        print(colab_cli("stop", "-s", SESSION_NAME, check=False).stdout)
        return
    if args.status:
        print_status(fetch_remote_json(f"{REMOTE_ROOT}/colab/training_status.json"))
        return
    if args.log:
        show_log()
        return
    if args.wait:
        wait(args.poll, args.timeout)
        return
    if args.fetch:
        fetch_results()
        return
    if args.launch:
        launch(args.gpu, smoke_test=args.smoke_test)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
