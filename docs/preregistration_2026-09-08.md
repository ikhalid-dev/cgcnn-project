# Pre-registration — ALIGNN ensemble, and a re-test of low-κ loss weighting

Written **2026-09-08, before either experiment's data existed.** The point of
writing it now is to fix what each outcome will mean, so that whatever comes
back cannot be rationalised after the fact.

Baselines are frozen: the 1,648-crystal matbench test set at `split_seed=42`,
the same split ALIGNN and CGCNN were both verified to use.

```
                          K MAE    G MAE   log10, test set
CGCNN, 1 model each       0.0696   0.0836
CGCNN, 3-model ensemble   0.0630   0.0781
ALIGNN, 1 model (current) 0.0539   0.0725
```

---

## Experiment A — an ALIGNN ensemble

**Question.** ALIGNN currently exists as one model per target. Does ensembling
it (a) improve the moduli, and (b) survive an ensemble-vs-ensemble comparison
against the CGCNN ensemble, which is the only comparison the project's own
rules permit?

**Design.** Three seeds per target (42, 1, 2 — the CGCNN convention), six
models total. Everything else identical to the existing ALIGNN run: 150 epochs,
batch 64, lr 0.001, `keep_data_order=True` so the split is the prepared one and
the seed changes weight initialisation only. Members averaged in log₁₀ space
(geometric mean), matching `05_ensemble.py`.

**In-session control.** Seed 42 alone is scored as a single model in the same
kernel. If it does not land near the existing single ALIGNN's 0.0539 / 0.0725,
something about the environment moved and the ensemble numbers are not
comparable to anything.

**Primary metric.** K and G MAE log₁₀ on the test set, ensemble vs ensemble.
**Secondary.** Spearman ρ against AFLOW-AGL's independent κ_L on the 441
matched test crystals — the metric step 50 showed transfers between targets.
Recall@10% will be *recorded but not used for any claim*: step 50 found only
1 of 29 paired-bootstrap CIs excluded zero, so it cannot resolve movement.

**Both error bars, per the project's rule.** Seed spread across the three
members, and a paired bootstrap over test crystals for any claim that one model
beats another.

### Pre-registered outcomes

| If | Then |
|---|---|
| ALIGNN-3ens beats ALIGNN-1 by more than the seed spread, **and** the paired bootstrap CI on K excludes zero | Ensembling helps ALIGNN as it helps CGCNN (~9–10%). ALIGNN-3ens becomes the project's best moduli model and the default for every downstream step. |
| ALIGNN-3ens beats ALIGNN-1 but the bootstrap CI crosses zero | Report the mechanism, not the total — exactly as round 5's joint result was reported. Still adopt it, because the uncertainty interval alone is worth it. |
| ALIGNN-3ens ≈ ALIGNN-1 (within seed spread) | Ensembling does not transfer to ALIGNN. Report that as the result; it would be genuinely surprising and worth stating, since it contradicts the 6.5× seed-variance reduction seen for CGCNN. |
| ALIGNN-3ens still loses to CGCNN on **measured** κ_L (Table 1, step 63) despite winning on matbench moduli | The matbench-moduli win does not transfer to reality. That is the more important finding, and it would mean "best model" must be stated per-benchmark rather than absolutely. |

**Regardless of outcome, ALIGNN gains a Monte-Carlo p05/p95 interval** it does
not currently have. Every ALIGNN κ_L in the project so far is a bare point
estimate, which is why ALIGNN cannot contribute to the DFT set's Arm A ranking.

### Downstream steps to re-run once A lands

15/16 (moduli and κ on the 1,213-crystal PINK set) · 63 (measured κ_L, 45
Table-1 materials) · 57 (the full 33,118-candidate GNoME screen) · 58 (the ρ
decomposition, which currently uses single-ALIGNN) · 61 (the DFT validation
set, whose Arm A ranking changes if ALIGNN gains an interval).

---

## Experiment B — low-κ loss weighting, judged on the right metric

**Why this is being reopened.** The project record says soft-material weighting
"fails, do not re-run". Two things about that verdict do not hold up:

1. **It was judged on metrics this project later invalidated.** Round 7
   rejected α = 0.4 on total κ MAE (0.1868 → 0.1965) and recall@10%
   (70.1% → 65.9%). The κ reference was the circular target step 48 exposed;
   recall@10% is the metric step 50 showed cannot resolve movement. Re-scored
   against AGL's independent κ, α = 0.4 sits at Spearman **0.845** against the
   baseline's **0.847** — a wash, not a failure.
2. **The trade was scored on the wrong population.** Weighting moved Q1 by
   +0.016 and cost Q2–Q5 0.01–0.02 each. Averaging all five quintiles penalises
   a low-κ optimisation for doing what it was asked to do. **Q1 is the only
   quintile a low-κ screen reads.**

**A real parameter defect, also found.** Weights are `(κ/median)^(−α)` clipped
to [0.25, 4.0] and then renormalised to mean 1, so emphasis saturates:

```
α = 0.4  Q1 mean weight 2.04   Q1/Q5  4.9×
α = 0.8  Q1 mean weight 2.48   Q1/Q5 12.4×
α = 1.2  Q1 mean weight 2.37   Q1/Q5 16×    ← decreasing
α = 2.0  Q1 mean weight 2.17   Q1/Q5 16×    ← saturated
```

Only α = 0.4 was ever tested. The usable range is ~0.4–0.8 and 0.8 was never
tried.

**Design.** CGCNN side, where `kappa_weight_alpha` already works and no
third-party training loop needs modifying. Three arms — α ∈ {0.0 control, 0.4,
0.8} — three seeds each, ensembled. Same recipe as round 7 otherwise.

**Primary metric.** **Q1 κ MAE against AFLOW-AGL's κ_L**, on the matched test
crystals — the low-κ quintile, scored against an external target. This is the
metric round 7 should have used and did not.
**Secondary.** Spearman ρ against AGL over all matched crystals, to confirm the
gain in Q1 is not bought with a collapse everywhere else.
**Not used for any claim:** total κ MAE against the in-house reference, and
recall@10%.

### Pre-registered outcomes

| If | Then |
|---|---|
| Q1 improves against AGL at α = 0.4 or 0.8, **and** overall Spearman is within noise of control | Weighting works for its actual purpose and was rejected on the wrong metric. Adopt it for the screening model and correct the record. |
| Q1 improves but overall Spearman drops clearly | A real trade. Report both numbers and let the use case decide — screening takes it, a general moduli model does not. |
| Q1 does not improve against AGL at either α | The original conclusion stands, now on defensible evidence. Say so, and record that the reason was the metric, not the lever. |
| α = 0.8 saturates (indistinguishable from 0.4) | The clip is the binding constraint, not α. Any future attempt must widen the clip, not raise α. |

**Explicitly not claimed either way until data lands:** that weighting helps.
The current evidence says only that it was *not properly tested*, which is a
different statement.

---

## Experiment C — ALIGNN on log₁₀ targets

Added **2026-09-08, while A's seeds 42 and 1 were still training and before any
of A's data existed.** Recorded here rather than in a later write-up because it
changes how A's result should be read.

**The observation.** The two architectures have been optimising different
objectives for this project's entire life, and it was never noticed:

```
CGCNN    trains on log10(GPa)   MSE on log     -> minimises RELATIVE error
ALIGNN   trains on raw GPa      MSE on raw GPa -> minimises ABSOLUTE error
```

`scripts/alignn/11_prepare_alignn_data.py:104` feeds `float(record.K_VRH)` —
raw GPa, no log transform. The "0.0539 log₁₀ MAE" quoted for ALIGNN everywhere
is computed **post hoc**, by log-transforming raw predictions in
`17_alignn_diagnostics.py:123`.

**Why this matters twice over.**

1. **ALIGNN's win is understated.** It beats CGCNN on log₁₀ MAE while not
   optimising log₁₀ MAE, and while CGCNN does. The architecture result is
   stronger than the record claims, not weaker.
2. **Raw-GPa MSE is the wrong loss for this project.** Absolute error is
   dominated by stiff crystals: a 10 GPa miss on diamond weighs the same as a
   10 GPa miss on a 10 GPa halide. It systematically under-weights exactly the
   soft, low-κ population the screen exists to find — the same failure mode as
   Experiment B's, in a different place.

**Independent confirmation already in hand.** ALIGNN predicted a **negative**
bulk modulus (−0.57 GPa) for a soft validation crystal whose true value is
2 GPa. A log-space model cannot produce that — 10^x is strictly positive. Only
a raw-space regressor can, and it did, on a soft crystal. That edge case was
recorded in `17_alignn_diagnostics.py` as a curiosity; it is a symptom of the
loss.

**Design.** One seed (42), identical architecture, identical split, single
change: `11_prepare_alignn_data.py` writes `log10(K_VRH)` / `log10(G_VRH)`
instead of the raw values. Predictions are then already in log space and need
no post-hoc transform. One seed is enough to detect a real move; three only if
the first is promising.

**Primary metric.** Q1 (lowest-κ quintile) MAE against AFLOW-AGL's κ_L — the
same low-κ metric Experiment B uses, so B and C are directly comparable.
**Secondary.** Overall K/G MAE log₁₀ on the matbench test set, against
Experiment A's ensemble, and the count of non-positive modulus predictions,
which should fall to exactly zero by construction.

### Pre-registered outcomes

| If | Then |
|---|---|
| Q1 improves and overall log₁₀ MAE also improves | The raw-GPa target was simply a mistake. Retrain the ensemble on log targets and treat every ALIGNN number produced before this as superseded. |
| Q1 improves, overall log₁₀ MAE roughly unchanged | The objective mismatch costs nothing in aggregate and helps where the screen reads. Adopt for screening; say plainly that the aggregate metric could not see it. |
| Q1 unchanged, overall log₁₀ MAE unchanged | The loss space does not matter for this architecture — a genuinely surprising null worth reporting, since it would mean ALIGNN is insensitive to a change CGCNN is built around. |
| Overall log₁₀ MAE gets **worse** | Raw-space training was accidentally helping, probably by letting the model use headroom a log target compresses away. Report it and keep the current ensemble. |
| Negative modulus predictions do not fall to zero | Something other than the loss space is producing them, and the diagnosis in this section is wrong. |

**Not claimed until data lands:** that log targets are better. The claim on
record today is only that the two architectures were never compared on a
matched objective, and that the mismatch runs against this project's own goal.
