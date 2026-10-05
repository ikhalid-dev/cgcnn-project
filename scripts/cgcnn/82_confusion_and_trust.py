#!/usr/bin/env python
"""
Step 82 - confusion matrices and accuracy for the three screen models, and
which kinds of prediction a DFT check is likely to CONFIRM.
================================================================================

    ~/miniconda3/envs/ml_env/bin/python scripts/cgcnn/82_confusion_and_trust.py

WHY
---
Before paying for DFT on the step-81 queue we want to know: when this screen
says "low kappa", how often does DFT agree? We cannot ask the GNoME crystals
(no DFT exists for them - that is the point of the request), so we ask the
next-best thing: 1,648 matbench crystals that HAVE DFT elastic moduli and that
none of the three models trained on.

    ALIGNN      3-seed ensemble, test split seed 42       (held out)
    CGCNN-ens   3-seed ensemble, test split seed 42       (held out)
    newbase     XGBoost, matbench 5-fold OUT-OF-FOLD      (held out - each
                crystal is predicted by the fold model that never saw it)
    max3        the screen's own rule: the HIGHEST of the three kappas, so a
                crystal only counts as low-kappa if all three models agree

WHAT "TRUTH" MEANS HERE
-----------------------
The DFT run we are about to request computes the ELASTIC TENSOR - K and G.
It does not recompute gamma; gamma in the screen comes from MLIP phonons. So
the question DFT can answer is "were the predicted K and G right?", and the
kappa it implies is

    kappa_DFT = Slack(K_DFT, G_DFT) * exp(-gamma)     with the SAME gamma.

To mimic that, the main analysis holds gamma FIXED: prediction and truth both
use the gamma of the true (DFT) moduli, so the only thing that can differ is
the moduli. This is labelled "moduli only". The usual full PINK chain, where
each model also derives its own gamma from its own K/G, is reported alongside
as "full chain" because that is how the models were benchmarked before.

THE FOUR BOXES OF A CONFUSION MATRIX (threshold kappa <= 1.0 W/m/K)
-------------------------------------------------------------------
    true positive   model says low, DFT says low     <- a confirmed candidate
    false positive  model says low, DFT says NOT low <- a wasted DFT run
    false negative  model says not low, DFT says low <- a missed material
    true negative   both say not low

    accuracy   = (TP + TN) / all          - misleading here, see below
    precision  = TP / (TP + FP)           - THE number for a DFT queue: "of the
                                            crystals I send, what fraction pass?"
    recall     = TP / (TP + FN)           - "of the real low-kappa crystals,
                                            what fraction did I catch?"

Only 15% of the test crystals are low-kappa, so a model that always answers
"no" scores 85% accuracy while finding nothing. The script prints that
always-no score next to every accuracy so it cannot be misread.

OUTPUTS (results/cgcnn/)
    82_test_predictions.csv     one row per test crystal, every model, both chains
    82_confusion_matrices.csv   TP/FP/FN/TN + accuracy/precision/recall, 3 thresholds
    82_accuracy_in_range.csv    % of predictions within +-25% and within 2x of DFT
    82_trust_table.csv          confirmation rate by predicted kappa x model agreement
    82_confusion_matrices.png   the four confusion matrices + the trust bars
"""
# ---- imports -----------------------------------------------------------------
# `import X as Y` loads a library and gives it a short nickname. No torch here:
# nothing in this script needs it, and torch 2.2.2 clashes with this env's numpy.
import os                      # building file paths that work on any machine
import numpy as np             # fast arithmetic on whole columns at once
import pandas as pd            # tables ("DataFrames") - like a spreadsheet in code
import matplotlib              # plotting library
matplotlib.use("Agg")          # "draw to a file, not a window" - must come before pyplot
import matplotlib.pyplot as plt

# ---- where things live ---------------------------------------------------------
# __file__ is this script's own path; three dirname() calls walk up
# scripts/cgcnn/82_x.py -> scripts/cgcnn -> scripts -> project root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "cgcnn")
PINK = os.path.expanduser("~/Desktop/pink_reproduction")   # ~ expands to /Users/mac

T = 300.0                      # temperature in kelvin, as everywhere in the screen
THRESHOLDS = [1.0, 0.5, 0.3]   # W/m/K. 1.0 is the screen's gate; the A1 tier sits near 0.3 and below
MODELS = ["ALIGNN", "CGCNN-ens", "newbase"]
AGREE = 1.5                    # "the three models agree" = highest kappa / lowest kappa <= 1.5


# =============================================================================
#  physics
# =============================================================================
def slack_parts(K, G, rho, V, n):
    """PINK Eq.(2) split into its two halves: prefactor and Poisson gamma.

    K, G in GPa, rho in g/cm^3, V in A^3 (primitive cell), n = atoms in that cell.
    kappa = prefactor * exp(-gamma). Keeping the halves separate is what lets us
    swap gamma for the "moduli only" comparison.
    These lines are the same chain as step 71's slack_physics(); check_physics()
    below proves it by reproducing a stored screen column.
    """
    v_long = np.sqrt((K + 4 * G / 3) / rho) * 1000          # longitudinal sound speed, m/s
    v_trans = np.sqrt(G / rho) * 1000                       # transverse sound speed, m/s
    v_sound = ((1 / v_long**3 + 2 / v_trans**3) / 3) ** (-1 / 3)   # Debye average
    ratio = v_long / v_trans
    poisson = (ratio**2 - 2) / (2 * ratio**2 - 2)
    gamma = 3 * (1 + poisson) / (2 * (2 - 3 * poisson))     # the Poisson-ratio shortcut
    prefactor = G * 1e9 * v_sound * (V * 1e-30) ** (1 / 3) / (n * T)
    return prefactor, gamma


def check_physics():
    """Hard stop unless slack_parts() reproduces step 13's stored Kappa_cal."""
    s = pd.read_csv(os.path.join(RES, "13_gnome_screen_all.csv"), nrows=2000)
    pref, gam = slack_parts(s.K_VRH_pred, s.G_VRH_pred, s["Density (g cm-3)"],
                            s["Volume (A3)"], s["Number of Atoms"])
    worst = np.nanmax(np.abs(pref * np.exp(-gam) / s["Kappa_cal (W m-1 K-1)"] - 1))
    if worst > 1e-9:
        raise SystemExit(f"physics check FAILED (worst relative error {worst:.1e})")
    print(f"physics check passed: reproduces step 13 kappa on 2,000 crystals to {worst:.0e}")


# =============================================================================
#  loading - every join is CHECKED, because a silent mismatch would put one
#  crystal's prediction next to another crystal's truth
# =============================================================================
def load_test_set():
    # truth: the DFT moduli every model was trained/tested against
    lab = pd.read_csv(os.path.join(ROOT, "data_full", "labels.csv"))

    # geometry: primitive-cell volume, atom count and density per matbench row.
    # mb_index 42 <-> "mb-00042"; f"{i:05d}" pads a number to five digits.
    geo = pd.read_parquet(os.path.join(PINK, "pink_geometry.parquet"))
    geo["mb_id"] = [f"mb-{i:05d}" for i in geo.mb_index]
    d = lab.merge(geo, on="mb_id", suffixes=("", "_geo"))
    if not (d.formula == d.formula_geo).all():
        raise SystemExit("geometry rows do not line up with labels.csv")

    # CGCNN ensemble - the file holds every split; keep only "test"
    for t in ["K", "G"]:
        p = pd.read_csv(os.path.join(RES, f"predictions_{t}_VRH_ens.csv"))
        p = p[p.split == "test"].set_index("material_id")
        d[f"{t}_CGCNN-ens"] = d.mb_id.map(p.pred_GPa)       # .map looks each id up in p
        d[f"{t}_check_cg"] = d.mb_id.map(p.true_GPa)

    # ALIGNN ensemble - one file, a "target" column says K or G
    a = pd.read_csv(os.path.join(ROOT, "results", "alignn", "18_alignn_ensemble_predictions.csv"))
    for t in ["K", "G"]:
        p = a[a.target == t].set_index("material_id")
        d[f"{t}_ALIGNN"] = d.mb_id.map(p.pred_GPa)
        d[f"{t}_check_al"] = d.mb_id.map(p.true_GPa)

    # newbase - out-of-fold log10 predictions, row_index = matbench row
    for t, task in [("K", "kvrh"), ("G", "gvrh")]:
        p = pd.read_csv(os.path.join(PINK, "xgboost_oof",
                        f"matbench_log_{task}_composition+structure+angular_oof.csv"))
        p["mb_id"] = [f"mb-{i:05d}" for i in p.row_index]
        p = p.set_index("mb_id")
        d[f"{t}_newbase"] = 10 ** d.mb_id.map(p.y_pred_oof)   # undo the log10
        d[f"{t}_check_nb"] = 10 ** d.mb_id.map(p.y_true)

    # keep only crystals that are in the held-out test split of BOTH networks
    d = d.dropna(subset=["K_CGCNN-ens", "K_ALIGNN", "G_CGCNN-ens", "G_ALIGNN"]).copy()

    # every file must agree on the TRUE value of every crystal - proves the ids line up
    for t, col in [("K", "K_VRH"), ("G", "G_VRH")]:
        for c in ["cg", "al", "nb"]:
            worst = np.nanmax(np.abs(d[f"{t}_check_{c}"] / d[col] - 1))
            if worst > 1e-4:
                raise SystemExit(f"{t} truth disagrees between files ({c}): {worst:.1e}")
    if d[[f"{t}_newbase" for t in "KG"]].isna().any().any():
        raise SystemExit("some test crystals have no newbase out-of-fold prediction")
    print(f"test crystals with all three models + DFT truth: {len(d)}  "
          f"(ids and true values agree across all four files)")
    return d


# =============================================================================
#  scoring helpers
# =============================================================================
def wilson(k, n, z=1.96):
    """95% range for a success rate k/n. Honest on small n, unlike +-2 sd."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def confusion(pred, true, thr):
    """The four boxes and the rates built from them, for 'kappa <= thr'."""
    p, t = pred <= thr, true <= thr          # True/False per crystal
    tp, fp = int((p & t).sum()), int((p & ~t).sum())     # & = and, ~ = not
    fn, tn = int((~p & t).sum()), int((~p & ~t).sum())
    n = tp + fp + fn + tn
    prec_lo, prec_hi = wilson(tp, tp + fp)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": (tp + tn) / n,
            "always_no_accuracy": (fp + tn) / n,     # the do-nothing score to beat
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "precision_lo95": prec_lo, "precision_hi95": prec_hi,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp))}


def in_range(ratio):
    """How close prediction is to DFT, given ratio = predicted / DFT."""
    return {"n": len(ratio),
            "within_25pct": float(np.mean(np.abs(ratio - 1) <= 0.25)),
            "within_2x": float(np.mean((ratio >= 0.5) & (ratio <= 2))),
            "median_abs_pct_error": float(np.median(np.abs(ratio - 1)) * 100)}


# =============================================================================
#  main
# =============================================================================
def main():
    check_physics()
    d = load_test_set()

    # ---- kappa for truth and for every model, both chains ---------------------
    pref_t, gam_t = slack_parts(d.K_VRH, d.G_VRH, d.rho_g_cm3, d.V_A3, d.n_atoms)
    d["kappa_DFT"] = pref_t * np.exp(-gam_t)
    for m in MODELS:
        pref, gam = slack_parts(d[f"K_{m}"], d[f"G_{m}"], d.rho_g_cm3, d.V_A3, d.n_atoms)
        d[f"{m}_moduli_only"] = pref * np.exp(-gam_t)    # truth's gamma: only moduli differ
        d[f"{m}_full_chain"] = pref * np.exp(-gam)       # the model's own Poisson gamma
    for chain in ["moduli_only", "full_chain"]:
        cols = [f"{m}_{chain}" for m in MODELS]
        d[f"max3_{chain}"] = d[cols].max(axis=1)         # the screen's pessimistic rule
        d[f"spread3_{chain}"] = d[cols].max(axis=1) / d[cols].min(axis=1)

    n_low = int((d.kappa_DFT <= 1.0).sum())
    print(f"DFT says low-kappa (<= 1.0 W/m/K): {n_low} of {len(d)} "
          f"({100 * n_low / len(d):.0f}%)   <= 0.3: {int((d.kappa_DFT <= 0.3).sum())}")

    # ---- 1. confusion matrices ---------------------------------------------------
    rows = []
    for chain in ["moduli_only", "full_chain"]:
        for m in MODELS + ["max3"]:
            for thr in THRESHOLDS:
                rows.append({"chain": chain, "model": m, "threshold": thr,
                             **confusion(d[f"{m}_{chain}"], d.kappa_DFT, thr)})
                # ** unpacks the dictionary returned by confusion() into this row
    cm = pd.DataFrame(rows)
    cm.to_csv(os.path.join(RES, "82_confusion_matrices.csv"), index=False)

    print("\n" + "=" * 92)
    print("CONFUSION MATRICES - 'low kappa' = kappa <= 1.0 W/m/K, moduli only (what DFT checks)")
    print("=" * 92)
    show = cm[(cm.chain == "moduli_only") & (cm.threshold == 1.0)]
    for _, r in show.iterrows():
        print(f"\n{r.model}")
        print(f"                     DFT low   DFT not-low")
        print(f"  model says low     {r.TP:>7}   {r.FP:>11}")
        print(f"  model says not-low {r.FN:>7}   {r.TN:>11}")
        print(f"  accuracy {r.accuracy:.1%} (always-no would score {r.always_no_accuracy:.1%})"
              f"   precision {r.precision:.1%} [{r.precision_lo95:.0%}-{r.precision_hi95:.0%}]"
              f"   recall {r.recall:.1%}   balanced acc {r.balanced_accuracy:.1%}")

    print("\nsame table at every threshold, both chains (precision = 'sent to DFT and confirmed'):")
    print(cm[["chain", "model", "threshold", "TP", "FP", "FN", "accuracy", "precision",
              "recall", "balanced_accuracy"]].round(3).to_string(index=False))

    # ---- 2. accuracy as "in range" -------------------------------------------------
    rows = []
    for m in MODELS + ["max3"]:
        for what, ratio_all in [("K", d.get(f"K_{m}", pd.Series(dtype=float)) / d.K_VRH),
                                ("G", d.get(f"G_{m}", pd.Series(dtype=float)) / d.G_VRH),
                                ("kappa moduli_only", d[f"{m}_moduli_only"] / d.kappa_DFT),
                                ("kappa full_chain", d[f"{m}_full_chain"] / d.kappa_DFT)]:
            if ratio_all.isna().all():
                continue                  # max3 has no single K or G of its own
            # "called low" = the crystals this model would have put in a DFT queue
            called = d[f"{m}_moduli_only"] <= 1.0
            for subset, mask in [("all test crystals", slice(None)),
                                 ("crystals the model calls low", called)]:
                rows.append({"model": m, "quantity": what, "subset": subset,
                             **in_range(ratio_all[mask].values)})
    rng = pd.DataFrame(rows)
    rng.to_csv(os.path.join(RES, "82_accuracy_in_range.csv"), index=False)
    print("\n" + "=" * 92)
    print("ACCURACY AS 'IN RANGE' - fraction of predictions within +-25% / within 2x of DFT")
    print("=" * 92)
    print(rng.round(3).to_string(index=False))

    # ---- 3. which predictions does DFT confirm? -------------------------------------
    # Only the crystals the screen would send (max3 <= 1.0), split by HOW low the
    # prediction is and by whether the three models agree.
    sent = d[d.max3_moduli_only <= 1.0].copy()
    sent["band"] = pd.cut(sent.max3_moduli_only, [0, 0.3, 0.5, 1.0],
                          labels=["<=0.3", "0.3-0.5", "0.5-1.0"])
    sent["agreement"] = np.where(sent.spread3_moduli_only <= AGREE,
                                 f"agree (spread <= {AGREE})", f"disagree (> {AGREE})")
    # a crystal "looks like" the GNoME candidates if it holds Cs or Rb.
    # str.contains with a regex: Cs or Rb followed by a non-lowercase letter or end
    sent["cs_rb"] = sent.formula.str.contains(r"(?:Cs|Rb)(?![a-z])", regex=True)
    sent["ratio"] = sent.kappa_DFT / sent.max3_moduli_only    # >1 = DFT is HIGHER than predicted

    rows = []
    groups = ([("all sent", sent)]
              + [(f"band {b}", g) for b, g in sent.groupby("band", observed=True)]
              + [(a, g) for a, g in sent.groupby("agreement")]
              + [(f"band {b}, {a}", g) for (b, a), g in sent.groupby(["band", "agreement"], observed=True)]
              + [("contains Cs or Rb", sent[sent.cs_rb])])
    for name, g in groups:
        k = int((g.kappa_DFT <= 1.0).sum())
        lo, hi = wilson(k, len(g))
        k2 = int(((g.ratio >= 0.5) & (g.ratio <= 2)).sum())
        rows.append({"group": name, "n": len(g),
                     "confirmed_low": k, "confirmed_pct": 100 * k / len(g),
                     "ci95_lo": 100 * lo, "ci95_hi": 100 * hi,
                     "DFT_within_2x_pct": 100 * k2 / len(g),
                     "median_DFT_over_pred": g.ratio.median(),
                     "p10_DFT_over_pred": g.ratio.quantile(0.10),
                     "p90_DFT_over_pred": g.ratio.quantile(0.90)})
    trust = pd.DataFrame(rows)
    trust.to_csv(os.path.join(RES, "82_trust_table.csv"), index=False)
    print("\n" + "=" * 92)
    print("WHICH 'LOW' CALLS DOES DFT CONFIRM?  (max3 <= 1.0, moduli only)")
    print("confirmed = DFT kappa also <= 1.0;  DFT/pred > 1 means DFT came out HIGHER")
    print("=" * 92)
    print(trust.round(2).to_string(index=False))

    # ---- per-crystal table for anyone who wants to check a single row -----------
    keep = (["mb_id", "formula", "n_atoms", "K_VRH", "G_VRH", "kappa_DFT"]
            + [f"{t}_{m}" for m in MODELS for t in "KG"]
            + [f"{m}_{c}" for c in ["moduli_only", "full_chain"] for m in MODELS + ["max3"]]
            + ["spread3_moduli_only"])
    d[keep].to_csv(os.path.join(RES, "82_test_predictions.csv"), index=False)

    # ---- figure ---------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 8.5))
    for i, m in enumerate(MODELS + ["max3"]):
        r = show[show.model == m].iloc[0]
        ax = fig.add_subplot(2, 4, i + 1)
        box = np.array([[r.TP, r.FP], [r.FN, r.TN]])
        ax.imshow(np.log1p(box), cmap="Blues")           # log colours so small boxes still show
        for (y, x), v in np.ndenumerate(box):
            ax.text(x, y, f"{v}\n{100 * v / box.sum():.1f}%", ha="center", va="center",
                    color="white" if v > box.max() / 3 else "black", fontsize=11)
        ax.set_xticks([0, 1], ["DFT low", "DFT not low"])
        ax.set_yticks([0, 1], ["says low", "says not low"])
        title = "max3 (screen rule)" if m == "max3" else m
        ax.set_title(f"{title}\nprecision {r.precision:.0%}  recall {r.recall:.0%}", fontsize=11)
    ax = fig.add_subplot(2, 1, 2)
    t = trust[trust.group.str.startswith("band ") & trust.group.str.contains(",")]
    x = np.arange(len(t))
    # test for "disagree", not "agree": the text "disagree" CONTAINS "agree", so
    # the obvious check `"agree" in g` is true for every bar
    ax.bar(x, t.confirmed_pct, color=["#c9803a" if "disagree" in g else "#3b6ea8" for g in t.group])
    ax.errorbar(x, t.confirmed_pct, yerr=[t.confirmed_pct - t.ci95_lo, t.ci95_hi - t.confirmed_pct],
                fmt="none", ecolor="black", capsize=4)
    for xi, (_, r) in zip(x, t.iterrows()):
        ax.text(xi, 3, f"n={r.n}", ha="center", color="white", fontsize=10)
    ax.set_xticks(x, [g.replace("band ", "pred ").replace(", ", "\n") for g in t.group], fontsize=9)
    ax.set_ylabel("% confirmed by DFT moduli\n(DFT kappa also <= 1.0)")
    ax.set_ylim(0, 105)
    ax.set_title("Crystals the screen would send to DFT (max3 <= 1.0): blue = three models agree, "
                 "orange = they disagree; bars = 95% range", fontsize=11)
    fig.suptitle("Step 82 - 1,648 held-out matbench crystals, kappa <= 1.0 W/m/K, moduli only",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(RES, "82_confusion_matrices.png")
    fig.savefig(out, dpi=130)
    print(f"\nwrote 82_test_predictions.csv, 82_confusion_matrices.csv, "
          f"82_accuracy_in_range.csv, 82_trust_table.csv, {os.path.basename(out)}")


if __name__ == "__main__":     # run main() only when the file is executed, not imported
    main()
