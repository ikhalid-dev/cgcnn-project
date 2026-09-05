# `direct_kappa_no_slack/` — predicting κ_L without the Slack model

This directory holds one experiment, kept separate from `scripts/cgcnn/` because
it deliberately **does not use the Slack formula at any point**. Everything else
in this repository routes a predicted bulk and shear modulus through Slack's
equation to get a lattice thermal conductivity. This asks whether that detour is
earning its keep.

## Why

The main pipeline is:

```
structure ─► CGCNN ─► K, G ─► Poisson relation ─► γ ─► SLACK ─► κ_L
```

Two of those arrows are known to be bad, and both were measured in this repo:

- **Slack itself.** Recomputing PINK's Table 1 gives 1.69× typical error, 1.32×
  systematic underprediction, worst case 4.5×, no uncertainty at all. It assumes
  three-phonon Umklapp dominance and an isotropic Debye solid, so it fails
  hardest on exactly the ultra-low-κ, strongly anharmonic, rattler-containing,
  complex-cell materials the screen exists to find.
- **γ.** Deriving γ from K/G through the empirical Poisson relation is the single
  largest error source in the chain — larger than the ML. The obvious fix is to
  train a γ head on AFLOW's tabulated γ, but step 49 showed AFLOW's γ disagrees
  with experimental/literature values by a mean |Δ log₁₀ γ| of 0.302 with a
  +0.98 systematic bias and r = 0.25 (n = 43). The label is not good enough to
  learn from.

Meanwhile AFLOW hands us **5,563 crystals with a tabulated κ_L and a cached
crystal graph already built** (`data_full/gamma_graphs.pt` + `gamma_labels.csv`).
So the question is simply: predict κ directly from the graph, and see whether
skipping the two bad arrows beats going through them.

```
structure ─► CGCNN ─► κ_L                    (this directory)
```

## The one variable

Everything is held to `scripts/cgcnn/37_train_gamma.py`'s recipe: identical
graphs, identical 70/15/15 split at `split_seed=42`, identical trunk
(`atom_fea_len=64`, `n_conv=3`, `h_fea_len=128`, one shared FC, one head FC of
width 64, dropout 0.10), identical optimisation (huber δ=1.0, 300 epochs, batch
64, Adam at lr 0.01, weight decay 1e-5, cosine schedule, early stop 80).

**The only difference is the head target and the absence of Slack.** Round 9
predicts log₁₀ G, log₁₀(K/G) and log₁₀ γ and then applies the formula. This
predicts log₁₀ κ and stops.

The trunk is re-implemented in `01_train_direct_kappa.py` rather than imported,
because `JointCrystalGraphConvNet` hardcodes `self.head1 = self.heads[1]` and
cannot be built with one head. The re-implementation is layer-for-layer
identical to that trunk — no shared module was modified.

## Pre-registered outcomes

Written before the first run, so that whatever comes back cannot be
rationalised after the fact. The comparison is against round 9 (re-scored in the
same script, not quoted from an old log) and the composition-only tree baseline,
on the same 835 held-out AFLOW crystals, with AFLOW's own κ as truth.

| If the direct model… | then the conclusion is |
|---|---|
| beats round 9 on F1 of the κ ≤ 1 call, 95% CI excluding zero | The Slack detour is costing real screening performance. Retire it from the AFLOW branch and re-run the GNoME screen without it. |
| beats round 9 but the CI crosses zero | Suggestive, not decisive on 835 crystals. Report as a mechanism, not a win — the same discipline rounds 4–7 were held to. |
| ties round 9 | Slack is neutral: it neither helps nor hurts once its inputs are predicted. Keep it, because it is interpretable and it generalises off-dataset where a direct model cannot. |
| loses to round 9 | The physics is carrying real information the graph alone does not supply. That is a positive result for the pipeline and closes this line of enquiry. |
| loses to the **tree baseline** | Whatever is happening, it is not about Slack — a composition-only model with no structure is beating a graph network, which points at underfitting (as step 43/44 already found on AFLOW) rather than at the formula. |

The tree baseline row matters most. Step 51 established the tree beats round 9
at the low-κ call (F1 0.646 vs 0.400, 95% CI [−0.352, −0.146]), so the tree —
not round 9 — is the bar to clear.

## Honest limits, stated up front

- **AFLOW's κ is not experiment.** It is a quasi-harmonic Debye–Grüneisen model,
  and step 49 showed its γ is badly wrong against literature. Every number here
  is agreement with an independent first-principles model, never accuracy.
- **5,563 crystals is small** for a graph network — a seventh of matbench. The
  architecture is already the reduced one 37 chose for that reason.
- **A direct model cannot extrapolate the way a formula can.** Slack, whatever
  its faults, encodes physics that holds outside the training distribution. A
  direct κ head has only the training set. This matters for the GNoME screen,
  which is far off-distribution, and it is an argument for keeping Slack even if
  the direct model wins here.
- **Train/test ratio is reported.** Step 43/44 found round 9 was *underfit* on
  AFLOW (training error worse than the tree's test error). If the direct model
  lands in the same 1.0–1.3 band, the result is about capacity, not about Slack.

## Files

| file | what it does |
|---|---|
| `01_train_direct_kappa.py` | Trains the single-head κ model. `--seed` and `--tag` for multi-seed runs. Writes a checkpoint, a summary JSON and per-crystal test predictions. |
| `02_compare_direct_vs_slack.py` | Scores direct vs round 9 vs tree on the same 835 crystals, with the paired bootstrap and the pre-registered decision table. |
| `models/` | Checkpoints and summaries, one per seed. |
| `results/csv/`, `results/png/` | Scores and figures. |

## Running it

```bash
P=~/miniconda3/envs/ml_env/bin/python
cd "~/Desktop/Cgcnn project"
for s in 42 1 2; do $P direct_kappa_no_slack/01_train_direct_kappa.py --seed $s --tag s$s; done
$P direct_kappa_no_slack/02_compare_direct_vs_slack.py
```

Three seeds because this project's own rule is that any effect smaller than the
seed spread is not a result, and one seed cannot measure the spread.
