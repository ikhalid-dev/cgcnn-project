#!/usr/bin/env python
"""
Step 70 - the DFT queue, rebuilt from the crystals that survived a REAL gamma.
================================================================================

    ml_env/bin/python scripts/cgcnn/70_survivor_dft_queue.py
    ml_env/bin/python scripts/cgcnn/70_survivor_dft_queue.py --skip-mp   # offline

WHY THIS REPLACES STEP 66's QUEUE
---------------------------------
Step 66 built a queue of 12 live crystals (8 arm-A candidates + 4 arm-C
controls) on a kappa whose Grueneisen parameter came from the empirical Poisson
relation. Step 69 measured that relation against real MLIP phonon gammas on
those same crystals and found it anti-correlated - Spearman -0.50 - because it
compresses every crystal into gamma ~1.5-2.1 regardless of what it is.

Re-scoring with the real gamma removed three of the six testable arm-A
candidates. This script assembles what is left, from three sources:

    step 61/66   the original arm-A candidates and arm-C controls
    step 68      the ten replacement nominees
    step 69      the real-gamma kappa for everything that has phonons

ADMISSION RULE, STATED BEFORE LOOKING AT THE ANSWER
---------------------------------------------------
A crystal enters the QUEUE arm only if ALL THREE hold:

    1. the phonon run succeeded                      (status == ok)
    2. it is dynamically stable, and 0 < gamma < 10  (the production filter)
    3. kappa_max <= 1.0 W/m/K under PINK Eq. (2) with the REAL gamma

Rule 3 uses the project's primary formula. Classic Slack is carried alongside
as `clears_slack` rather than used as a gate, because Slack is the declared
robustness check, not the criterion - but a crystal that clears Eq. (2) and
fails Slack is a threshold case, and the DFT request must say so rather than
assert it. That column exists so the request cannot quietly forget.

Arm C is admitted unchanged. Its job is to catch a pipeline that calls
everything soft, and step 69 showed the real gamma pushes three of the four
controls DOWN while pushing the candidates up - opposite directions, which is
what makes the correction credible rather than an inflation artifact.

Arm B is NOT here. All six were dropped at step 66 after step 64 found round
9's predicted K exceeded every real measured K for those chemistries. Nothing
in step 69 revisits that, and they never had phonons computed.

THE SECOND QUESTION: HAS SOMEONE ALREADY DONE THIS?
---------------------------------------------------
Step 67 asked Materials Project whether the original twelve had already been
computed. The three replacement crystals did not exist when it ran, so the same
query is repeated here over the whole new queue - not just the new rows, so the
output is self-contained and does not have to be read beside 67's.

Two different questions, with different answers, and conflating them would
overstate what is known:

    does the STRUCTURE exist in MP?   -> several do; MP has ingested
                                        GNoME-derived entries, which arrive
                                        with a formation energy and nothing else
    does an ELASTIC TENSOR exist?     -> that is the thing we would be
                                        duplicating, and step 67 found none

OUTPUTS
    results/cgcnn/70_survivor_dft_queue.csv    the queue, one row per crystal
    results/cgcnn/70_prior_calculations.csv    the MP check over that queue
"""
import argparse
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "cgcnn")

KAPPA_THRESHOLD = 1.0      # W/m/K - the project's low-kappa line, unchanged


def load_rescored():
    """The two step-69 outputs, stacked into one frame with a common schema.

    pd.concat glues frames end to end; ignore_index=True renumbers the rows so
    the result has 0..n-1 instead of two overlapping index ranges.
    """
    frames = []
    for fn, origin in [("69_dft_queue_kappa_real_gamma.csv", "step 61 queue"),
                       ("69_replacement_kappa_real_gamma.csv", "step 68 replacement")]:
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} - run step 69 for both sets first")
        f = pd.read_csv(path, dtype={"material_id": str})
        f["origin"] = origin
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mp", action="store_true",
                    help="build the queue but do not query Materials Project")
    args = ap.parse_args()

    d = load_rescored()

    # Arm C rows carry the literal label "C: negative control"; everything else
    # is a low-kappa candidate. .str.startswith needs na=False or a missing
    # label would raise instead of simply not matching.
    is_control = d["label"].astype(str).str.startswith("C:", na=False)

    # ---- the admission rule, applied ---------------------------------------
    d["clears_pink"] = d["kappa_max_new"] <= KAPPA_THRESHOLD
    d["clears_slack"] = d["kappa_max_new_slack"] <= KAPPA_THRESHOLD
    d["status"] = "DROPPED"
    d.loc[is_control, "status"] = "CONTROL"
    d.loc[~is_control & d["clears_pink"], "status"] = "QUEUE"

    d["reason"] = ""
    d.loc[d.status == "QUEUE", "reason"] = (
        "kappa_max <= 1.0 with the real MLIP gamma, PINK Eq. (2)")
    d.loc[d.status == "CONTROL", "reason"] = (
        "stiff by every model - guards against a pipeline that calls everything soft")
    d.loc[d.status == "DROPPED", "reason"] = (
        "kappa_max exceeds 1.0 once the real gamma replaces the Poisson shortcut")

    # ---- crystals that never reached the kappa test at all -----------------
    # These are absent from step 69's output entirely (it drops failed and
    # unstable rows), so they have to be recovered from the raw phonon CSVs or
    # they vanish silently from the audit trail - and "silently vanished" is
    # exactly how a queue ends up with a crystal nobody re-checked.
    excluded = []
    for tag, origin in [("dft_queue", "step 61 queue"),
                        ("replacements", "step 68 replacement")]:
        g = pd.read_csv(os.path.expanduser(
            f"~/Desktop/data_generation/results/01_gamma_phonon_{tag}.csv"),
            dtype={"mp_id": str})
        for r in g.itertuples():
            if r.status != "ok":
                why = f"phonon pipeline failed: {r.error}"
            elif not r.dynamically_stable:
                why = f"dynamically unstable: {int(r.n_imaginary)} imaginary modes"
            elif not (0 < r.gamma_mlip < 10):
                why = f"gamma out of physical range: {r.gamma_mlip:.2f}"
            else:
                continue                      # it made it into step 69, skip
            excluded.append({"material_id": r.mp_id,
                             "formula": r.formula if isinstance(r.formula, str) else "",
                             "origin": origin, "status": "EXCLUDED", "reason": why})
    exc = pd.DataFrame(excluded)

    # A row that failed before the structure was even parsed has no formula in
    # the phonon CSV, so recover it from the files that nominated the crystal.
    # Without this the audit trail carries a blank where Cs4Sb4PdPt should be.
    names = {}
    for fn in ["61_dft_validation_set.csv", "68_replacement_pool.csv"]:
        src = pd.read_csv(os.path.join(RESULTS, fn), dtype={"material_id": str})
        names.update(dict(zip(src.material_id, src.formula)))
    exc["formula"] = exc.apply(
        lambda r: r["formula"] or names.get(r["material_id"], ""), axis=1)

    cols = ["status", "origin", "formula", "material_id", "space_group",
            "kappa_max_old", "kappa_max_new", "kappa_max_new_slack",
            "gamma_mlip", "clears_pink", "clears_slack", "reason"]
    queue = pd.concat([d[[c for c in cols if c in d.columns]], exc],
                      ignore_index=True)
    # Sort QUEUE first, then CONTROL, then the rejects; within QUEUE, the most
    # confident candidate first.
    order = {"QUEUE": 0, "CONTROL": 1, "DROPPED": 2, "EXCLUDED": 3}
    queue = (queue.assign(_o=queue.status.map(order))
                  .sort_values(["_o", "kappa_max_new"])
                  .drop(columns="_o").reset_index(drop=True))

    out = os.path.join(RESULTS, "70_survivor_dft_queue.csv")
    queue.to_csv(out, index=False)

    # ---- report -------------------------------------------------------------
    print("=" * 100)
    print("STEP 70 - THE DFT QUEUE, REBUILT ON REAL GAMMA")
    print("=" * 100)
    for st in ["QUEUE", "CONTROL", "DROPPED", "EXCLUDED"]:
        sub = queue[queue.status == st]
        if sub.empty:
            continue
        print(f"\n{st}  ({len(sub)})")
        show = sub[["formula", "material_id", "origin", "kappa_max_old",
                    "kappa_max_new", "kappa_max_new_slack", "gamma_mlip",
                    "clears_slack", "reason"]]
        print(show.to_string(index=False, na_rep="-",
                             float_format=lambda v: f"{v:.3f}"))
    n_q = (queue.status == "QUEUE").sum()
    n_both = int(queue[queue.status == "QUEUE"].clears_slack.sum())
    print(f"\n{n_q} candidates + {(queue.status=='CONTROL').sum()} controls = "
          f"{n_q + (queue.status=='CONTROL').sum()} DFT calculations")
    print(f"of the {n_q} candidates, {n_both} also clear the threshold under classic "
          f"Slack; the other {n_q - n_both} are threshold cases and the request "
          f"must label them as such.")
    print(f"\nwrote {out}")

    if args.skip_mp:
        return

    # ---- has anyone already computed these? --------------------------------
    from mp_api.client import MPRester
    from pymatgen.core import Composition

    live = queue[queue.status.isin(["QUEUE", "CONTROL"])]
    rows = []
    print("\n" + "=" * 100)
    print("PRIOR CALCULATIONS - querying Materials Project")
    print("=" * 100)
    with MPRester() as mpr:
        for r in live.itertuples():
            comp = Composition(r.formula)
            chemsys = "-".join(sorted(e.symbol for e in comp.elements))
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                fields=["material_id", "formula_pretty", "symmetry",
                        "energy_above_hull"])
            same = [x for x in docs
                    if Composition(x.formula_pretty).reduced_formula == comp.reduced_formula]
            hit = same[0] if same else None
            # An elastic tensor is a separate collection in MP - a summary hit
            # says the STRUCTURE is known, not that the moduli are.
            n_elastic = 0
            if same:
                n_elastic = len(mpr.materials.elasticity.search(
                    material_ids=[x.material_id for x in same],
                    fields=["material_id"]))
            rows.append({
                "status": r.status, "formula": r.formula,
                "gnome_id": r.material_id, "chemsys": chemsys,
                "mp_entries_in_chemsys": len(docs),
                "mp_same_formula": len(same),
                "mp_id": hit.material_id if hit else "",
                "mp_e_above_hull": hit.energy_above_hull if hit else None,
                "mp_has_elastic": n_elastic,
            })
            print(f"  {r.formula:16s} {chemsys:18s} chemsys={len(docs):3d}  "
                  f"same formula={len(same)}  elastic tensors={n_elastic}")

    pc = pd.DataFrame(rows)
    out2 = os.path.join(RESULTS, "70_prior_calculations.csv")
    pc.to_csv(out2, index=False)
    print(f"\nstructures already in MP : {(pc.mp_same_formula > 0).sum()} of {len(pc)}")
    print(f"ELASTIC TENSORS in MP    : {int(pc.mp_has_elastic.sum())}  "
          f"<- the number that decides whether this is duplicated work")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
