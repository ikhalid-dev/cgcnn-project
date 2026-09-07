#!/usr/bin/env python3
"""
STEP 57 - Screen the FULL GNoME set with ALIGNN, the project's best moduli model.
================================================================================

    python scripts/cgcnn/57_alignn_gnome_screen.py
    python scripts/cgcnn/57_alignn_gnome_screen.py --limit 200   # smoke test

WHY THIS EXISTS
-----------------
Every GNoME candidate this project has ever produced came from the CGCNN
ensemble or, later, the composition-only tree. ALIGNN has only ever been run on
subsets - the 1,213-crystal PINK set (step 15), the DFT shortlist (step 27) and
the 1,215-candidate small-cell pool (step 28). It has never seen the full
33,118-candidate screen.

That is worth fixing, because ALIGNN's win is the ONE model improvement in this
project that survived the honest metric. Step 50 re-scored 30 runs against
AFLOW-AGL's independent kappa and ALIGNN came top on rank agreement (Spearman
0.874), beating the CGCNN baseline by +0.027 with a paired-bootstrap 95% CI of
[+0.009, +0.048] - excluding zero. Seven rounds of CGCNN architecture work
produced nothing that survived that test; ALIGNN did.

So the best moduli model available has been sitting unused while a weaker one
picks the DFT shortlist.

WHAT THIS DOES AND DOES NOT CHANGE
------------------------------------
Changed:   the moduli. ALIGNN's K and G replace the CGCNN ensemble's.
Unchanged: everything downstream. The same Slack chain, the same derived-gamma
           Poisson relation, and the same structural constants (volume, density,
           atom count) that are already columns in step 39's output - they come
           from the CIF, not from any model, so they are identical for every
           screen and cannot leak a difference.

Gamma is DERIVED here, not predicted. ALIGNN has no gamma head, and step 49
established that AFLOW's tabulated gamma - the only label source for one - is
unusable (|d log10 gamma| 0.302 against literature, r = 0.25, which is 5.16x the
gamma head's own error against 1.39x for the moduli). So a derived gamma is not
a compromise here; it is the honest choice.

THE FAILURE MODE THIS PROJECT ALREADY FOUND, AND GUARDS AGAINST
-----------------------------------------------------------------
Step 28 caught ALIGNN emitting its regression floor - about 1e-3 GPa - for two
candidates, which sailed through as spectacularly low kappa. That is a
degenerate output, not a discovery. Every row here carries
`alignn_prediction_reliable`, false when either modulus lands below
`min_modulus_gpa`, and the summary reports the count rather than hiding it.

RESUMABILITY
--------------
~0.16 s per structure on this laptop, so the full screen is about 1.5 hours.
Progress is checkpointed every `checkpoint_every` rows, and a re-run picks up
where it stopped. Losing 90 minutes to a closed lid is a self-inflicted wound.
"""

# =============================================================================
#  CONFIG - every path, threshold and tunable lives here
# =============================================================================
CONFIG = {
    # ---- inputs -------------------------------------------------------------
    # Step 39's screen supplies the candidate list AND the structural constants
    # (Volume, Density, Number of Atoms), so no CIF is parsed twice for those.
    "screen_csv": "results/cgcnn/39_gnome_screen_all_gamma.csv",
    # GNoME's summary maps material_id -> Composition, which is how the CIF
    # archive names its entries.
    "gnome_summary": "gnome_data/stable_materials_summary.csv",
    "gnome_zip": "gnome_data/by_composition.zip",
    "zip_prefix": "by_composition",
    "alignn_k_dir": "results/alignn/alignn_bulk_modulus_kv",
    "alignn_g_dir": "results/alignn/alignn_shear_modulus_gv",

    # ---- outputs ------------------------------------------------------------
    "out_csv": "results/alignn/57_gnome_screen_alignn.csv",
    "checkpoint_csv": "results/alignn/57_gnome_screen_alignn.partial.csv",
    "csv_dir": "results/alignn",
    "png_dir": "results/alignn",

    # ---- screening ----------------------------------------------------------
    "kappa_threshold": 1.0,      # W/m/K - the low-kappa call this project screens on
    "min_modulus_gpa": 0.01,     # below this ALIGNN has hit its regression floor, not found a soft crystal
    "checkpoint_every": 500,     # rows between checkpoint writes
    "device": "cpu",             # a 33k-structure loop is graph-building bound, not matmul bound
    "limit": 0,                  # 0 = the whole screen; set small for a smoke test
}
# =============================================================================

import argparse
import os
import sys
import time
import zipfile

# torch FIRST: MKL loads its own OpenMP runtime and a duplicate libiomp5 aborts
# the process if numpy/pandas/pymatgen get in first. This bit during development
# of this very script.
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "cgcnn"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "alignn"))

from importlib import import_module   # noqa: E402
_al = import_module("15_alignn_predict_moduli")   # load_alignn_target, predict_pair
_gamma = import_module("37_train_gamma")          # kappa_full, derived_gamma

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#3f9142"
INK, INK_SOFT, MUTED, GRID = "#1c1c1c", "#4a4a4a", "#8a8a8a", "#e3e3e3"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N candidates (smoke test)")
    p.add_argument("--restart", action="store_true",
                   help="ignore any checkpoint and start from scratch")
    return p.parse_args()


def load_candidates(cfg):
    """The screen list, joined to the Composition key the CIF archive uses."""
    screen = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["screen_csv"]))
    summary = pd.read_csv(os.path.join(PROJECT_ROOT, cfg["gnome_summary"]),
                          usecols=["MaterialId", "Composition"])
    merged = screen.merge(summary, left_on="material_id", right_on="MaterialId",
                          how="left")
    missing = int(merged["Composition"].isna().sum())
    if missing:
        print(f"  WARNING: {missing} candidates have no Composition in the GNoME "
              f"summary and cannot be looked up in the CIF archive; they are skipped.")
    return merged


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    if args.limit is not None:
        cfg["limit"] = args.limit

    os.makedirs(os.path.join(PROJECT_ROOT, cfg["csv_dir"]), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, cfg["out_csv"])
    ckpt_path = os.path.join(PROJECT_ROOT, cfg["checkpoint_csv"])

    print("=" * 78)
    print("  STEP 57 - ALIGNN over the full GNoME screen")
    print("=" * 78)
    print()

    candidates = load_candidates(cfg)
    if cfg["limit"]:
        candidates = candidates.head(cfg["limit"])
    print(f"  candidates: {len(candidates)}")

    # ---- resume -------------------------------------------------------------
    done = {}
    if os.path.exists(ckpt_path) and not args.restart:
        # material_id is read as str explicitly. Most GNoME ids contain letters
        # so the column is object dtype anyway, but an all-digit id would be
        # parsed as an int and lose any leading zero, silently failing to match
        # the `mid` built below and re-scoring that row.
        prev = pd.read_csv(ckpt_path, dtype={"material_id": str})
        # .to_dict() per row, NOT the Series that .iterrows() yields. `rows`
        # below mixes resumed entries with freshly scored dicts, and
        # pd.DataFrame() on a list whose first element is a Series takes an
        # all-Series code path that dies on the first dict it meets with
        # "AttributeError: 'dict' object has no attribute 'dtype'". That killed
        # the first resume attempt at the first checkpoint write.
        done = {str(r["material_id"]): r for r in prev.to_dict("records")}
        print(f"  resuming: {len(done)} already scored "
              f"(delete {cfg['checkpoint_csv']} or pass --restart to start over)")

    # ---- models -------------------------------------------------------------
    device = torch.device(cfg["device"])
    k_model, k_cfg = _al.load_alignn_target(
        os.path.join(PROJECT_ROOT, cfg["alignn_k_dir"]), device)
    g_model, _ = _al.load_alignn_target(
        os.path.join(PROJECT_ROOT, cfg["alignn_g_dir"]), device)
    # Graph settings come from the MODEL'S OWN config, not from a constant here:
    # this project's ALIGNN was trained at cutoff 5.0, and building 8.0-radius
    # graphs for inference would silently feed it a different neighbourhood
    # than it ever saw in training.
    graph_settings = k_cfg
    print(f"  ALIGNN loaded, graph settings from the model: {graph_settings}")

    from jarvis.core.atoms import pmg_to_atoms
    from pymatgen.core import Structure
    archive = zipfile.ZipFile(os.path.join(PROJECT_ROOT, cfg["gnome_zip"]))

    # ---- the loop -----------------------------------------------------------
    rows, failures = [], []
    n_scored = 0          # NEWLY scored this run - not the resumed ones
    t0 = time.time()
    for i, (_, cand) in enumerate(candidates.iterrows()):
        mid = str(cand["material_id"])
        if mid in done:
            rows.append(done[mid])
            continue
        comp = cand.get("Composition")
        if not isinstance(comp, str):
            failures.append((mid, "no Composition"))
            continue
        try:
            text = archive.read(f"{cfg['zip_prefix']}/{comp}.CIF").decode("utf-8")
            atoms = pmg_to_atoms(Structure.from_str(text, fmt="cif"))
            k_gpa, g_gpa = _al.predict_pair(k_model, g_model, atoms,
                                            graph_settings, device)
        except Exception as exc:                    # one bad CIF must not kill 33k rows
            failures.append((mid, f"{exc.__class__.__name__}: {str(exc)[:60]}"))
            continue

        # ALIGNN can emit its regression floor (~1e-3 GPa) instead of a real
        # prediction. log10 of that is a huge negative number that would look
        # like a spectacular discovery, so it is flagged and excluded from the
        # candidate call rather than silently ranked first.
        reliable = (k_gpa > cfg["min_modulus_gpa"]) and (g_gpa > cfg["min_modulus_gpa"])
        if reliable:
            lk, lg = np.log10(k_gpa), np.log10(g_gpa)
            kappa = float(_gamma.kappa_full(
                np.array([lk]), np.array([lg]),
                _gamma.derived_gamma(np.array([lk]), np.array([lg])),
                np.array([float(cand["Volume (A3)"]) * 1e-30]),
                np.array([float(cand["Number of Atoms"])]),
                np.array([float(cand["Density (g cm-3)"])]))[0])
        else:
            kappa = np.nan

        rows.append({
            "material_id": mid, "formula": cand.get("formula"),
            "n_atoms": cand.get("Number of Atoms"),
            "K_alignn": k_gpa, "G_alignn": g_gpa,
            "Kappa_alignn": kappa,
            "alignn_prediction_reliable": reliable,
            # carried through so every model's call sits in one row
            "Kappa_r9_gamma": cand.get("Kappa_r9_gamma"),
            "K_r9_pred": cand.get("K_r9_pred"), "G_r9_pred": cand.get("G_r9_pred"),
        })

        n_scored += 1

        if len(rows) % cfg["checkpoint_every"] == 0:
            # The checkpoint exists to protect a ~90 minute run, so it must
            # never be the thing that ends one. Report loudly and keep scoring;
            # the final write at the end of main() is the one that must succeed.
            try:
                pd.DataFrame(rows).to_csv(ckpt_path, index=False)
            except Exception as exc:
                print(f"    WARNING: checkpoint write failed ({exc.__class__.__name__}: "
                      f"{exc}) - continuing, progress is still in memory", flush=True)
            # Rate must count only rows scored THIS run. `i` also counts the
            # checkpointed rows skipped instantly on resume, which inflated the
            # rate ~14x and made the ETA read "~5 min left" on a 70-minute run.
            rate = n_scored / max(time.time() - t0, 1e-9)
            left = (len(candidates) - i - 1) / max(rate, 1e-9)
            print(f"    {len(rows):>6}/{len(candidates)}  "
                  f"{rate:.1f}/s  ~{left / 60:.0f} min left", flush=True)

    scored = pd.DataFrame(rows)
    scored.to_csv(out_path, index=False)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)                        # the run finished; the partial is clutter

    # ---- report -------------------------------------------------------------
    elapsed = time.time() - t0
    n_bad = int((~scored["alignn_prediction_reliable"]).sum())
    print()
    print(f"  scored {len(scored)} candidates in {elapsed / 60:.1f} min "
          f"({len(scored) / max(elapsed, 1e-9):.1f}/s)")
    print(f"  failures (unreadable CIF / no Composition): {len(failures)}")
    for mid, why in failures[:5]:
        print(f"      {mid}: {why}")
    print(f"  ALIGNN hit its regression floor on {n_bad} candidates "
          f"(flagged, excluded from the call)")

    ok = scored[scored["alignn_prediction_reliable"]]
    thr = cfg["kappa_threshold"]
    n_al = int((ok["Kappa_alignn"] <= thr).sum())
    print()
    print("  " + "-" * 74)
    print(f"  LOW-KAPPA CALL (<= {thr} W/m/K) ON {len(ok)} RELIABLE CANDIDATES")
    print("  " + "-" * 74)
    print(f"    ALIGNN     {n_al:>7} ({100 * n_al / len(ok):.1f}%)")
    if "Kappa_r9_gamma" in ok:
        r9 = pd.to_numeric(ok["Kappa_r9_gamma"], errors="coerce")
        n_r9 = int((r9 <= thr).sum())
        both = int(((ok["Kappa_alignn"] <= thr) & (r9 <= thr)).sum())
        print(f"    round 9    {n_r9:>7} ({100 * n_r9 / len(ok):.1f}%)")
        print(f"    BOTH agree {both:>7}  <- the defensible list: two independent")
        print(f"                          architectures, sharing only the physics")
        valid = r9.notna() & ok["Kappa_alignn"].notna()
        if valid.sum() > 10:
            from scipy.stats import spearmanr
            rho = spearmanr(np.log10(ok["Kappa_alignn"][valid]),
                            np.log10(r9[valid])).correlation
            print(f"    rank agreement between them: Spearman rho = {rho:.3f}")

    # ---- figure -------------------------------------------------------------
    if "Kappa_r9_gamma" in ok and ok["Kappa_r9_gamma"].notna().sum() > 10:
        r9 = pd.to_numeric(ok["Kappa_r9_gamma"], errors="coerce")
        m = r9.notna() & ok["Kappa_alignn"].notna() & (r9 > 0) & (ok["Kappa_alignn"] > 0)
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        ax.scatter(r9[m], ok["Kappa_alignn"][m], s=5, alpha=0.25,
                   color=BLUE, edgecolor="none")
        lim = [max(1e-3, min(r9[m].min(), ok["Kappa_alignn"][m].min())),
               max(r9[m].max(), ok["Kappa_alignn"][m].max())]
        ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--")
        ax.axhline(thr, color=ORANGE, lw=1.0, ls=":")
        ax.axvline(thr, color=ORANGE, lw=1.0, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("round-9 CGCNN $\\kappa_L$ (W/m/K)", color=INK_SOFT)
        ax.set_ylabel("ALIGNN $\\kappa_L$ (W/m/K)", color=INK_SOFT)
        ax.set_title(f"Two architectures over {int(m.sum())} GNoME candidates\n"
                     f"lower-left quadrant = both call it low-kappa",
                     color=INK, fontsize=11)
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(PROJECT_ROOT, cfg["png_dir"],
                                 "57_alignn_vs_r9_gnome.png"),
                    dpi=150, facecolor="white")
        plt.close(fig)

    print()
    print("  wrote:")
    print(f"    {cfg['out_csv']}")
    print(f"    {cfg['png_dir']}/57_alignn_vs_r9_gnome.png")


if __name__ == "__main__":
    main()
