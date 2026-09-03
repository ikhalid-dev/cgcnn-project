#!/usr/bin/env python3
"""
STEP 33 - Download AFLOW's AGL thermal data (Grueneisen parameter and friends).
================================================================================

    python scripts/cgcnn/33_fetch_aflow_gamma.py
    python scripts/cgcnn/33_fetch_aflow_gamma.py --max-pages 4    # quick test

WHY WE NEED THIS
----------------
The pipeline currently DERIVES the Grueneisen parameter from the predicted
moduli, through the empirical Poisson-ratio relation:

    x^2 = K/G + 4/3   ->   nu   ->   gamma = 3(1+nu) / (2(2-3nu))

That single step is the largest error source in the whole pipeline, and we
have the receipt. On the paper's own Table 1 (45 materials with measured
kappa), this reproduction scores:

    gamma DERIVED from predicted K/G   MAE 0.427 log10
    gamma SUBSTITUTED from tabulation  MAE 0.226 log10   <- same model, same K,G

Nearly halved, by changing nothing but where gamma came from. So the next
architecture step is a THIRD head that predicts gamma directly rather than
deriving it - which needs gamma labels, which is what this script fetches.

It matters most exactly where the screen operates. Stratifying the test set by
true kappa, the lowest quintile is the worst-predicted by a wide margin:

    Q1 (lowest kappa)  MAE 0.3230       <- the only quintile the screen cares about
    Q4                 MAE 0.1400
    ratio                    2.3x

Low-kappa crystals are soft, with extreme K/G, which is where the Poisson
relation is least trustworthy and where gamma swings hardest.

WHAT IT DOWNLOADS
-----------------
AFLOW's AGL (Automatic Gibbs Library) module runs a quasi-harmonic Debye
model over the AFLOW entries and tabulates the results. We pull every entry
that HAS a Grueneisen parameter - the `agl_gruneisen(*)` filter means "this
field is present", without which most rows come back null.

Three of the project's reference papers use this same set: Klochko 2024
(5,578 entries, as low-quality pretraining data for transfer learning),
Zeng 2024 (5,578 entries, gamma listed as a primary feature), and PINK itself
(2,535 entries for its AFLOW validation).

API NOTES, LEARNED THE HARD WAY
-------------------------------
  - The endpoint is https://aflow.org/API/aflux/. The older
    aflowlib.duke.edu/search/API/ address 404s.
  - `$paging(N,K)` is page N with K results. `$paging(0)` does NOT return a
    count - it returns the first page like any other query, so the only way to
    know when to stop is to page until a short or empty response comes back.
  - Fields are requested by naming them in the query, comma separated. A field
    with `(*)` appended becomes a FILTER (must be present) rather than just a
    requested column.
"""

# =============================================================================
#  CONFIG - every tunable lives here
# =============================================================================
CONFIG = {
    # Where the CSV lands. Sits beside labels.csv and graphs.pt so a future
    # training script can find it without another path to configure.
    "out_path": "data_full/aflow_agl.csv",

    # AFLUX endpoint. Do not use aflowlib.duke.edu/search/API/ - it 404s.
    "api_base": "https://aflow.org/API/aflux/",

    # Results per request. 500 is comfortably inside what the server returns in
    # one response; raising it risks timeouts, lowering it just means more
    # round trips.
    "page_size": 500,

    # Stop after this many pages. 0 means "keep going until a page comes back
    # short or empty". Set to a small number for a smoke test.
    "max_pages": 0,

    # Seconds to wait for each request. AFLUX is not fast; 90 is not generous.
    "timeout": 90,

    # Retries per page before giving up on the whole download. Transient 5xx
    # and connection resets are common enough to matter over ~12 pages.
    "retries": 4,

    # Seconds to wait between requests. Politeness, and it keeps the server
    # from rate-limiting a long run.
    "pause": 1.0,
}

# Columns pulled from AGL/AEL. agl_gruneisen carries the (*) filter, so only
# entries that actually have a Grueneisen parameter come back.
#   agl_*  = quasi-harmonic Debye model outputs (the thermal half)
#   ael_*  = elastic tensor outputs (the moduli, for cross-checking against
#            our own CGCNN predictions on overlapping compounds)
FIELDS = [
    "agl_gruneisen(*)",                 # THE target for the future third head
    "agl_thermal_conductivity_300K",    # AGL's own kappa, useful as a weak label
    "agl_debye",                        # Debye temperature
    "agl_heat_capacity_Cv_300K",        # the Cv that Zeng 2024 could not predict
    "ael_bulk_modulus_vrh",             # K, same VRH averaging as matbench
    "ael_shear_modulus_vrh",            # G, likewise
    "compound",
    "auid",                             # AFLOW's unique id
    "spacegroup_relax",
    "natoms",
    "density",
]
# =============================================================================

import argparse                    # CLI flag parsing, auto-generated from CONFIG
import json                        # decoding AFLUX's JSON responses
import os                          # path joining, directory creation
import ssl                         # TLS context used to relax certificate checking for AFLOW's server
import sys                         # (imported for parity with the rest of the pipeline; not directly called below)
import time                        # sleep()/time() for pacing requests and reporting elapsed time
import urllib.error                # URLError/HTTPError, to detect transient network failures
import urllib.request              # urlopen(), used to issue each AFLUX request

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
        if value is not None:                       # None means the flag was not passed on the CLI
            cfg[key] = value                        # otherwise the CLI value overrides the default
    return cfg                                      # effective configuration for this run


def fetch_page(cfg, page, context):
    """Fetch one page, retrying on transient failures.

    Returns a list of records, or raises after `retries` attempts. An empty
    list is a legitimate result - it is how the end of the data announces
    itself - so it is returned rather than retried.
    """
    query = ",".join(FIELDS) + f",$paging({page},{cfg['page_size']})"   # AFLUX query string: field list plus the paging directive
    url = cfg["api_base"] + "?" + query             # full request URL

    for attempt in range(1, cfg["retries"] + 1):    # up to cfg["retries"] attempts for this one page
        try:
            raw = urllib.request.urlopen(
                url, timeout=cfg["timeout"], context=context).read().decode()
            # ^ open the URL, read the raw response bytes, decode to a string
            data = json.loads(raw)                  # parse the JSON response body
            # AFLUX returns a bare list on success. Anything else (a dict with
            # an error key, a string) means the query was rejected, and
            # retrying an identical bad query will not help.
            if not isinstance(data, list):          # a non-list response means the query itself was rejected
                raise SystemExit(f"AFLUX rejected the query on page {page}: "
                                 f"{str(data)[:300]}")
            return data                             # success: list of record dicts for this page
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, TimeoutError) as exc:   # transient network/parsing failures worth retrying
            if attempt == cfg["retries"]:           # out of retries - give up on this page entirely
                raise SystemExit(f"page {page} failed after {attempt} attempts: "
                                 f"{type(exc).__name__}: {exc}")
            # Back off linearly. The failures we see are load-related, so a
            # short wait is usually enough.
            wait = 3 * attempt                      # linearly increasing backoff: 3s, 6s, 9s, ...
            print(f"    page {page} attempt {attempt} failed "
                  f"({type(exc).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait)                        # wait before the next attempt


def main():
    cfg = parse_overrides()                         # resolve CONFIG + CLI overrides
    out_path = os.path.join(PROJECT_ROOT, cfg["out_path"])   # absolute path to the output CSV
    os.makedirs(os.path.dirname(out_path), exist_ok=True)    # ensure data_full/ exists before writing into it

    # AFLOW's certificate chain is not always complete on macOS's default
    # trust store. This is a public read-only data API and no credentials are
    # sent, so relaxing verification costs nothing here - do NOT copy this
    # pattern to anything authenticated.
    context = ssl.create_default_context()          # start from Python's default TLS settings
    context.check_hostname = False                  # AFLOW's certificate does not match its hostname cleanly
    context.verify_mode = ssl.CERT_NONE             # skip certificate verification entirely (AFLOW-specific workaround)

    print("=" * 74)
    print("  AFLOW AGL download - Grueneisen parameter and thermal properties")
    print("=" * 74)
    print(f"  endpoint  : {cfg['api_base']}")
    print(f"  page size : {cfg['page_size']}")
    print(f"  fields    : {len(FIELDS)}")
    print()

    records, page = [], 1                           # accumulator for every record fetched so far, and the current page number
    start = time.time()                              # wall-clock start, for the elapsed-seconds progress readout
    while True:                                      # keep paging until a short/empty page or max_pages stops it
        batch = fetch_page(cfg, page, context)       # fetch this page's records
        records.extend(batch)                        # append them to the running total
        print(f"  page {page:3d}  +{len(batch):4d}  total {len(records):6d}"
              f"   [{time.time() - start:.0f}s]", flush=True)

        # A short page means we have reached the end of the data. An empty one
        # likewise. Either way there is nothing after it.
        if len(batch) < cfg["page_size"]:           # fewer records than a full page means this was the last page
            break
        page += 1                                    # advance to the next page
        if cfg["max_pages"] and page > cfg["max_pages"]:   # 0 is falsy, so this only triggers when a page cap was set
            print(f"  stopping at max_pages={cfg['max_pages']}")
            break
        time.sleep(cfg["pause"])                    # brief pause between requests, to stay polite to AFLOW's server

    if not records:                                  # the query returned nothing at all across every page
        raise SystemExit("no records returned - check the query fields")

    # pandas is imported late and after nothing torch-related runs, so the
    # usual import-order hazard does not apply in this script.
    import pandas as pd
    df = pd.DataFrame(records)                       # turn the list of record dicts into a table

    # Deduplicate on auid. AFLOW can serve the same entry through more than one
    # library path, and a duplicate would be counted twice in any training set
    # built from this file.
    before = len(df)                                 # row count before deduplication, for the printed message
    df = df.drop_duplicates(subset="auid")           # keep only the first occurrence of each unique auid
    if len(df) < before:                             # only print if duplicates were actually found
        print(f"\n  dropped {before - len(df)} duplicate auids")

    df.to_csv(out_path, index=False)                # write the final table, no pandas row-index column

    print()
    print("=" * 74)
    print(f"  {len(df)} unique entries -> {os.path.relpath(out_path, PROJECT_ROOT)}")
    print("=" * 74)
    for col in ("agl_gruneisen", "agl_thermal_conductivity_300K", "agl_debye",
                "ael_bulk_modulus_vrh", "ael_shear_modulus_vrh"):
        if col not in df:                            # guard against a field AFLUX did not return this run
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()   # numeric, non-null values for this column only
        print(f"  {col:<32} n={len(s):6d}  median {s.median():10.3f}"
              f"  range {s.min():.3g} to {s.max():.3g}")


if __name__ == "__main__":                           # only run main() when executed directly, not on import
    main()
