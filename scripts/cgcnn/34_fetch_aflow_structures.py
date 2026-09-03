#!/usr/bin/env python3
"""
STEP 34 - Fetch AFLOW crystal structures for the SOFT-material augmentation.
=============================================================================

    python scripts/cgcnn/34_fetch_aflow_structures.py
    python scripts/cgcnn/34_fetch_aflow_structures.py --limit 20   # smoke test

WHY THIS EXISTS
---------------
33_fetch_aflow_gamma.py downloaded AFLOW's AGL table: 5,653 entries with
Grueneisen parameter, kappa, Debye temperature and both elastic moduli. What
that table does NOT contain is atomic positions - it has `compound`, `natoms`
and `density`, but no structure. CGCNN builds its graph from atomic positions,
so the CSV on its own cannot be trained on at all.

AFLOW does serve the relaxed structure, as a VASP POSCAR, at the entry's own
aurl. This script fetches them.

WHY ONLY THE SOFT ONES
----------------------
The whole point of the augmentation is the low-kappa tail, where the model is
worst:

    Q1 (lowest kappa)  MAE log10(G) 0.1340   median true G  10.5 GPa
    Q4                 MAE log10(G) 0.0565   median true G  58.5 GPa

and where matbench is thinnest:

                          G < 5   G < 10   G < 20 GPa
    matbench                205      771     2371
    AFLOW (available)        84      321      953

Fetching all 5,653 would take roughly six times as long and mostly add stiff
crystals the model already handles well. MAX_SHEAR_MODULUS restricts the pull
to the part that addresses the actual deficit.

WHAT COMES OUT
--------------
    data_full/aflow_structures/<auid>.poscar   one VASP POSCAR per entry
    data_full/aflow_soft.csv                   the metadata rows that succeeded

35_merge_aflow.py turns those into graphs and merges them into the training
set. Nothing here touches the existing matbench data.

A WARNING ABOUT MIXING DFT SOURCES
-----------------------------------
AFLOW's AEL moduli and matbench's VRH moduli are different DFT calculations
with different settings. They are the same physical quantity but not the same
number, and mixing them adds label noise. 35_merge_aflow.py measures the
disagreement on overlapping compounds before merging - which doubles as an
estimate of the irreducible label noise in this regime.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # The AGL metadata table written by 33_fetch_aflow_gamma.py.
    "agl_csv": "data_full/aflow_agl.csv",
    # One POSCAR per entry lands here.
    "struct_dir": "data_full/aflow_structures",
    # Metadata for the entries we successfully fetched.
    "out_csv": "data_full/aflow_soft.csv",

    # Only fetch entries with a shear modulus at or below this, in GPa. This is
    # the knob that defines "soft". 20 GPa captures 953 AFLOW entries against
    # matbench's 2,371 - a ~40% enlargement of the soft tail. Raise it to pull
    # more (and dilute the targeting); lower it to concentrate on the extreme
    # tail at the cost of sample count.
    "max_shear_modulus": 20.0,

    # Skip entries with more atoms than this. Graph construction is O(atoms x
    # neighbours) and the periodic neighbour search is the slowest step in the
    # whole pipeline; a handful of 200-atom cells would dominate the runtime.
    "max_atoms": 60,

    # Hard cap on how many to fetch. 0 means no cap. Use a small number to
    # smoke-test the whole path before committing to the full pull.
    "limit": 0,

    "timeout": 40,      # seconds per structure request
    "retries": 3,       # attempts before giving up on one entry
    "pause": 0.15,      # seconds between requests, to stay polite
}
# =============================================================================

import argparse                    # CLI flag parsing, auto-generated from CONFIG
import os                          # path joining, existence checks, directory creation
import ssl                         # TLS context used to relax certificate checking for AFLOW's server
import sys                         # (imported for parity with the rest of the pipeline; not directly called below)
import time                        # sleep()/time() for pacing requests and reporting elapsed time
import urllib.error                # HTTPError, to distinguish a 404 from other failures
import urllib.request              # urlopen(), used to fetch each structure over HTTP

# __file__ is this script's own path; three dirname() calls climb
# scripts/cgcnn/ -> scripts/ -> the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_overrides():
    """One --flag per CONFIG key, so CONFIG stays the single source of truth."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, default in CONFIG.items():             # register one CLI flag per CONFIG entry
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key,
                            type=type(default), default=None,
                            help=f"override CONFIG['{key}'] (default: {default!r})")
    args = parser.parse_args()                      # parse sys.argv against the registered flags
    cfg = dict(CONFIG)                              # start from the defaults
    for key, value in vars(args).items():           # walk each parsed argument
        if value is not None:                       # None means the flag was not passed
            cfg[key] = value                        # otherwise the CLI value wins
    return cfg                                      # effective configuration for this run


def aurl_to_http(aurl):
    """Turn AFLOW's own aurl form into a fetchable http URL.

    AFLOW writes locations as `host:PATH/TO/ENTRY` with a colon where the
    first slash belongs. Only the FIRST colon is a separator - the rest of the
    path can legitimately contain them - so split with maxsplit=1.
    """
    return "http://" + aurl.replace(":", "/", 1)    # replace only the first colon (maxsplit=1) with a slash, then prefix the scheme


def fetch_structure(aurl, cfg, context):
    """Download one relaxed structure as a VASP POSCAR string, or return None.

    AFLOW is inconsistent about which filename holds the relaxed structure
    depending on which library the entry came from, so try the known names in
    order. A missing file is a 404 and means "try the next name", not a
    failure worth retrying.
    """
    base = aurl_to_http(aurl)                       # this entry's base URL
    for filename in ("CONTCAR.relax", "CONTCAR.relax.vasp", "POSCAR.relax"):   # try each known relaxed-structure filename in order
        for attempt in range(1, cfg["retries"] + 1):   # up to cfg["retries"] attempts per filename
            try:
                text = urllib.request.urlopen(
                    base + "/" + filename,
                    timeout=cfg["timeout"], context=context).read().decode()
                # ^ open the URL, read the raw response bytes, and decode them to a string
                # A POSCAR needs at least the comment, scale, three lattice
                # vectors, counts and a coordinate mode before any positions.
                if len(text.splitlines()) >= 8:     # crude sanity check that this looks like a real POSCAR, not an error page
                    return text                     # success - hand the POSCAR text back to the caller
                break                                # too short to be real - stop retrying this filename, try the next one
            except urllib.error.HTTPError as exc:   # server responded, but with an error status
                if exc.code == 404:
                    break          # wrong filename for this entry, try the next
                if attempt == cfg["retries"]:       # exhausted retries on a non-404 HTTP error
                    return None
                time.sleep(2 * attempt)             # back off longer after each failed attempt before retrying
            except Exception:                       # any other failure (timeout, connection reset, SSL error, ...)
                if attempt == cfg["retries"]:       # exhausted retries
                    return None
                time.sleep(2 * attempt)             # same backoff pattern
    return None                                     # every filename/attempt combination failed


def main():
    cfg = parse_overrides()                         # resolve CONFIG + CLI overrides
    import pandas as pd                             # imported here so --help works without pandas installed

    agl_path = os.path.join(PROJECT_ROOT, cfg["agl_csv"])   # absolute path to the AGL metadata CSV
    if not os.path.exists(agl_path):                # fail fast with a clear message
        raise SystemExit(f"missing {agl_path}\n"
                         f"Run scripts/cgcnn/33_fetch_aflow_gamma.py first.")
    struct_dir = os.path.join(PROJECT_ROOT, cfg["struct_dir"])   # absolute path to the output structures directory
    os.makedirs(struct_dir, exist_ok=True)          # create it (and any parents) if it does not already exist

    df = pd.read_csv(agl_path)                      # load every AGL row
    for col in ("ael_shear_modulus_vrh", "ael_bulk_modulus_vrh", "natoms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")   # coerce to numeric, turning bad values into NaN

    # Both moduli must be present and positive - the training target is
    # log10 of each, and a non-positive modulus is an unphysical DFT entry.
    before = len(df)                                # row count before filtering, for the printed summary
    df = df[(df.ael_shear_modulus_vrh > 0) & (df.ael_bulk_modulus_vrh > 0)]   # keep only physically valid positive moduli
    df = df[df.ael_shear_modulus_vrh <= cfg["max_shear_modulus"]]   # keep only "soft" entries per the configured threshold
    df = df[df.natoms <= cfg["max_atoms"]]          # drop cells above the atom-count ceiling
    df = df.dropna(subset=["aurl", "auid"])         # need both fields to locate and name the download
    print(f"  {before} AGL entries -> {len(df)} soft candidates "
          f"(G <= {cfg['max_shear_modulus']} GPa, <= {cfg['max_atoms']} atoms)")
    if cfg["limit"]:                                # 0 is falsy - only triggers when a smoke-test limit was set
        df = df.head(cfg["limit"])                  # keep only the first N rows
        print(f"  limited to {len(df)}")

    context = ssl.create_default_context()          # start from Python's default TLS settings
    context.check_hostname = False                  # AFLOW's server certificate does not match its hostname cleanly
    context.verify_mode = ssl.CERT_NONE             # disable certificate verification entirely (AFLOW-specific workaround)

    kept, failed, cached = [], 0, 0                 # accumulators: rows kept, failure count, already-on-disk count
    start = time.time()                             # wall-clock start, for the elapsed-minutes progress readout
    for i, row in enumerate(df.itertuples(), 1):    # iterate rows, 1-based counter for progress printing
        path = os.path.join(struct_dir, f"{row.auid.replace(':', '_')}.poscar")   # filesystem-safe filename for this entry
        # Resume support: a rerun after an interruption should not re-download
        # everything it already has.
        if os.path.exists(path) and os.path.getsize(path) > 0:   # already downloaded and non-empty
            cached += 1                             # count it as satisfied without a network request
            kept.append(row)                        # still include it in the output metadata
        else:
            text = fetch_structure(row.aurl, cfg, context)   # attempt the download
            if text is None:                        # every filename/attempt failed
                failed += 1
            else:
                with open(path, "w") as fh:
                    fh.write(text)                   # persist the POSCAR text to disk
                kept.append(row)                     # include it in the output metadata
                time.sleep(cfg["pause"])             # brief pause between successful requests, to stay polite to AFLOW's server
        if i % 50 == 0 or i == len(df):              # print progress every 50 rows, and on the final row
            print(f"  {i:5d}/{len(df)}  kept {len(kept):5d}  cached {cached:5d}  "
                  f"failed {failed:4d}   [{(time.time() - start) / 60:.1f} min]",
                  flush=True)                        # flush so progress is visible immediately during a long run

    if not kept:                                    # nothing succeeded at all
        raise SystemExit("nothing fetched - check connectivity and the aurl format")

    out = pd.DataFrame(kept).drop(columns=["Index"], errors="ignore")
    # ^ itertuples() rows carry a stray "Index" field; drop it if present, ignore if not
    out["poscar_file"] = [f"{a.replace(':', '_')}.poscar" for a in out.auid]   # record each row's on-disk filename
    out_path = os.path.join(PROJECT_ROOT, cfg["out_csv"])   # absolute path to the metadata CSV
    out.to_csv(out_path, index=False)               # write it, no pandas row-index column

    print()
    print("=" * 74)
    print(f"  {len(out)} structures -> {os.path.relpath(struct_dir, PROJECT_ROOT)}/")
    print(f"  metadata            -> {os.path.relpath(out_path, PROJECT_ROOT)}")
    print(f"  failed              {failed}")
    print("=" * 74)
    g = out.ael_shear_modulus_vrh                   # shear-modulus column, for the summary stats below
    print(f"  shear modulus: median {g.median():.2f} GPa, "
          f"range {g.min():.2f} to {g.max():.2f}")
    for thr in (5, 10, 20):                         # report how many entries fall below a few reference thresholds
        print(f"    below {thr:>2} GPa: {(g < thr).sum():5d}")


if __name__ == "__main__":                          # only run main() when executed directly, not on import
    main()
