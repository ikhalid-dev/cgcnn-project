# Predicting elastic moduli with a CGCNN — a PINK reproduction

A from-scratch reimplementation of the machine-learning stage of **PINK**, trained on the full
matbench elastic benchmark and used to predict bulk and shear moduli for 1,213 Materials Project
crystals.

> Liu, Y.; Wang, X.; Hao, Y.; Li, X.; Sun, J.; Lookman, T.; Ding, X.; Gao, Z.
> *PINK: physics-informed machine learning for lattice thermal conductivity.*
> J. Mater. Inf. 2025, 5, 12. [doi:10.20517/jmi.2024.86](https://dx.doi.org/10.20517/jmi.2024.86)

PINK predicts **lattice thermal conductivity (κ_L)** from a CIF file by combining a crystal graph
network that predicts elastic moduli with a simplified Slack model:

```
CIF ──► CGCNN ──► bulk modulus K, shear modulus G
   └──► volume V, atom count n, density ρ  (pymatgen)
                    │
                    ▼
   sound velocities (v_l, v_t, v_s) ──► Poisson ratio ──► Grüneisen γ
                    │
                    ▼
                   κ_L
```

**This repository implements the first arrow** — CIF → (K, G) — including the network itself, and
produces the (K, G) table the κ_L stage consumes. The full method is written up in
**[`docs/method.pdf`](docs/method.pdf)**.

## Results

Measured on 1,648 held-out crystals. Every model shares the same train/val/test split, so all
numbers are directly comparable.

| Model | MAE log₁₀(GPa) | R² | MAE (GPa) | Relative error |
|---|---|---|---|---|
| **Bulk modulus, 3-model ensemble** | **0.0630** | **0.901** | 10.2 | **15.6%** |
| **Shear modulus, 3-model ensemble** | **0.0781** | **0.899** | 7.3 | **19.7%** |

For scale:

| Reference | MAE log₁₀(GPa) |
|---|---|
| **This work (bulk ensemble)** | **0.0630** |
| matbench CGCNN baseline (context only — see note below) | ≈0.07 |
| Best published on this benchmark ([coGN](https://matbench.materialsproject.org/)) | ≈0.054 |

**Note on that middle row.** The PINK paper never states its own modulus accuracy in log₁₀(GPa) —
it reports "MAE < 13" with no log unit, i.e. *raw* GPa. The ≈0.07 figure long attached to "the PINK
paper" here is the general matbench leaderboard's own published CGCNN baseline on this identical
benchmark — almost certainly close to what PINK's model scores, since both are CGCNN on the same
data, but never verified to be PINK's own stated number, because PINK doesn't report one in these
units. On the metric the paper *does* state, we also come out ahead: our ensemble's raw-GPa MAE is
10.2 (bulk) / 7.3 (shear), both under its stated "< 13." Full breakdown in
[`results/RESULTS.md`](results/RESULTS.md).

### Reading the error

The model is trained on log₁₀(modulus), so MAE converts to a **multiplicative** error of
`10^MAE − 1`: an MAE of 0.063 means predictions land within a factor of 1.156, i.e. ~16%.

`metrics_summary.csv` also reports `pct_of_range` (MAE in GPa over the full 1–575 GPa span, the
normalised-MAE convention), which is ~1.4–1.8%. That number is legitimate but flatters the model —
a wide data range shrinks it for free — so **quote it only alongside `rel_error_pct`, never
instead of it**.

**A ~2% relative error is not achievable on this task by any model.** The targets are DFT-computed
elastic moduli, and DFT elastic constants themselves disagree with experiment by roughly 5–15%. A
model cannot be more accurate than the labels it is fitted to, so 2% sits below the noise floor of
the training data.

## Quick start

```bash
conda env create -f environment.yml && conda activate pink-cgcnn
./run_pipeline.sh --predict
```

The six trained checkpoints are committed (~380 KB each), so this reproduces every figure, the
metrics table, and the 1,213-crystal prediction file **in about two minutes without training
anything**. It does download the matbench benchmark (~100 MB) on first run.

To retrain from scratch:

```bash
./run_pipeline.sh             # ~45 min on a GPU, ~9 hours on a CPU
./run_pipeline.sh --quick     # 5 epochs, ~10 min, checks the wiring only
```

**No GPU?** Use [`colab/PINK_CGCNN.ipynb`](colab/) — it runs these same scripts on a free Colab T4
in ~45 minutes and hands back the checkpoints. Upload that one file; the code travels inside it.

## The main output

**`results/pink_moduli_predictions.csv`** — bulk and shear modulus for all 1,213 crystals, the
input for the κ_L stage:

| column | meaning |
|---|---|
| `material_id`, `formula`, `n_sites` | the crystal |
| `K_VRH_pred`, `G_VRH_pred` | predicted moduli, GPa |
| `pugh_ratio` | G/K; below ~0.57 ductile, above brittle |
| `provenance` | `unseen` (935), or `train`/`val`/`test` if the crystal is in matbench |
| `K_VRH_dft`, `G_VRH_dft` | DFT reference where one exists, else blank |
| `*_spread_log10` | disagreement between ensemble members — an uncertainty estimate |

**`provenance` exists because 278 of the 1,213 crystals are in matbench and were therefore trained
on.** A prediction for one of those is recall, not generalisation, and reporting it as evidence
would be circular. Only `unseen` rows are the model doing its actual job; only `test` rows are
honest accuracy evidence.

That distinction is also a working correctness check: on the 42 `test`-provenance crystals the
error is 0.062 (K) / 0.085 (G), matching the benchmark test error. If the provenance mapping were
wrong we would instead see the much lower train-set error.

## What we did differently from the paper

Detailed in [`docs/method.pdf`](docs/method.pdf); in brief:

1. **Written from scratch.** `cgcnn_scratch/` is our own implementation of the graph
   construction, convolution, pooling and training loop — not a fork. No pre-trained weights are
   ever loaded.
2. **Ensembles instead of single models.** Three initialisations per target, averaged in log
   space. Worth ~9–10% of the error, and it supplies a per-crystal uncertainty the paper has no
   equivalent of.
3. **Adam + cosine annealing** rather than the original CGCNN's SGD + step decay.
4. **Provenance tracking.** Every prediction records whether the model was trained on that
   crystal. We could not find this distinction drawn in the paper's own screening results, and
   without it, accuracy claims on a screening set are not interpretable.
5. **A documented error floor.** We state explicitly why ~15% relative error is the realistic
   target and what sets it.
6. **A different train/validation/test split — caught, not chosen.** The paper's own released
   checkpoint (`args` saved inside its `.pth.tar`) records an 80/10/10 split; we use 70/15/15. This
   was found by inspecting the checkpoint directly, after an earlier version of this README
   incorrectly listed the split as unchanged — kept here as an honest difference rather than
   silently corrected to match, since retraining on 80/10/10 purely to match a number wouldn't
   itself improve anything.

Everything else architectural is verified identical **against the paper's own released
checkpoint**, not just its prose: `atom_fea_len=64`, `h_fea_len=128`, `n_conv=3`, `n_h=1`. Worth
noting — the paper's *text* says "two hidden layers," but the checkpoint's own saved `n_h` is 1,
matching ours. The paper's text and its own model disagree with each other.

## Repository layout

| Path | What it is |
|---|---|
| `cgcnn_scratch/` | The network. `model.py` (conv, pooling), `data.py` (CIF → graph) |
| `scripts/01b_prepare_full_dataset.py` | **The training set**: all 10,987 matbench crystals |
| `scripts/02_train.py` | Train one model |
| `scripts/03_evaluate.py` | Metrics, parity/residual/training figures |
| `scripts/04_predict_moduli.py` | **K and G for the 1,213 PINK crystals** |
| `scripts/05_ensemble.py` | Combine members, score the ensemble |
| `scripts/06_summarise.py` | Collapse all runs into one table + `RESULTS.md` |
| `scripts/07_predict_kappa.py` | **Stage 2**: our moduli → κ_L, with Monte Carlo uncertainty |
| `scripts/08_compare_kappa.py` | Validates stage 2 against the paper's own pipeline |
| `scripts/09_fetch_validation_set.py` | Fetches the paper's Table 1 (45 real materials) and runs our pipeline on them |
| `scripts/10_validate_table1.py` | Validates stage 2 against **real experimental κ_L** (Table 1) |
| `scripts/11_prepare_alignn_data.py` | matbench → ALIGNN's format, same split as the CGCNN ensemble |
| `scripts/12_train_alignn.py` | Trains ALIGNN (Phase 2, a second architecture) on one target |
| `colab/PINK_ALIGNN_colab.py` | Generates `colab/PINK_ALIGNN.ipynb`, the GPU run for the above |
| `scripts/13_screen_gnome.py` | **The actual GNoME screen** — 33,323 candidates, K/G/κ_L/uncertainty for each |
| `scripts/14_compare_gnome_screen.py` | Compares our screen against the paper's own 11,869 published candidates |
| `gnome_data/` | Downloaded GNoME summary CSV + structure zip (gitignored, ~620 MB) |
| `run_pipeline.sh` | Stage 1, everything above, in order |
| `results/` | Checkpoints, metrics, figures, predictions |
| `results/archive/` | One superseded run, kept for comparison (see below) |
| `complete-data/` | The 1,213 Materials Project CIFs (prediction set) |
| `data/table1_validation/` | The 45 fetched Table 1 CIFs used for real-data validation |
| `data_full/` | Labels + provenance mapping for the 10,987 training crystals |
| `colab/`, `kaggle/` | GPU runners (see `docs/method.pdf` §7) |
| `docs/` | The method write-up and its LaTeX source |
| `presentation/` | A from-scratch-explainer Beamer deck for a non-CGCNN audience (see below) |
| `pink_predict.py` | Separate: full κ_L inference using the *paper's* pre-trained weights |

## The presentation

**[`presentation/CGCNN_presentation.pdf`](presentation/CGCNN_presentation.pdf)** (34 slides, LaTeX/Beamer)
teaches the whole project to someone who has never seen a CGCNN: why crystals need a graph
representation at all, nodes/edges/cutoff radius worked through concretely in real rock-salt NaCl
(own figures, generated from the actual CIF via `pymatgen`/`networkx` — not a hand-drawn schematic),
the gated convolution and pooling explained before their equations, and then this project's own
results — the same performance table, literature comparison, parity plots, residuals and ensemble
figures as above.

Like the method PDF, nothing is hand-typed: `presentation/make_figures.py` regenerates every figure
and a `metrics.tex` macro file straight from `results/metrics_summary.csv`, so the talk can never go
stale relative to the pipeline's actual output.

```bash
cd presentation && ./build.sh    # regenerates figures + metrics.tex, then compiles (xelatex)
```

Requires `xelatex` (for the Georgia/Avenir Next system fonts via `fontspec` — `pdflatex` will not
work) and Python with `networkx`/`pymatgen`/`matplotlib` for the figure generator.

`results/archive/cpu-150epoch/` holds one superseded run: 150 epochs with a plateau LR schedule
instead of 200 with cosine annealing. It shares `--split-seed 42` with the current models, so it
remains directly comparable — it is the evidence for what the schedule change bought.

## Where the data comes from

**Structures.** 1,213 CIFs from the Materials Project, in `complete-data/`. These are the
*prediction* set.

**Labels.** From the matbench elastic benchmarks — `matbench_log_kvrh` and `matbench_log_gvrh`,
10,987 DFT-computed entries each, the same data the paper trained on. These are the *training*
set. Downloaded automatically by `matminer` on first run.

A CIF says where the atoms are; it does not say what the modulus is. The two sets are joined on
*structure*, not ID, because matbench strips Materials Project IDs: bucket by reduced formula
(cheap), then confirm with pymatgen's symmetry-aware `StructureMatcher`. **278 of the 1,213 match**
— that is the ceiling, not a bug, since matbench covers ~11k of MP's ~150k materials.

**This is the key methodological point.** It is tempting to label what you can and train on the
overlap — but the 1,213 CIFs were never the limiting factor, *labels* were, and intersecting the
two sets discards 97% of the labels that exist. We instead train on all 10,987 matbench crystals
and treat the 1,213 CIFs purely as a prediction set, which is also what the paper does.

## Environment

Python **3.11** (`pymatgen`/`matminer` do not yet install cleanly on 3.14). Exact versions in
`requirements.txt`; `environment.yml` builds the whole thing.

Two things that will bite otherwise:

- **Import `torch` before `numpy`/`pandas`/`pymatgen`.** In a conda environment MKL loads its own
  OpenMP runtime first, and the duplicate `libiomp5` segfaults torch. Every script here already
  orders its imports this way — do not "tidy" them alphabetically.
- **`torch.load` changed in torch 2.6**, where `weights_only` began defaulting to `True`. Every
  `torch.load` of our own artifacts passes `weights_only=False` explicitly, so the code works on
  both sides of that change.

## Reproducibility notes

- **Seeds.** `--seed` controls weight initialisation and batch order; `--split-seed` controls the
  data split. They are separate on purpose: ensemble members must differ in the first and agree on
  the second. `05_ensemble.py` refuses to combine members whose test splits disagree rather than
  trusting the caller to have got it right.
- **The graph cache** (`data_full/graphs.pt`, 240 MB) is not committed. It is deterministic and
  rebuilds in ~6 minutes.
- **Checkpoints always serialise on CPU**, so GPU-trained weights load on a machine without CUDA.
- **The Colab notebook is generated, never hand-edited** (`python colab/PINK_CGCNN_colab.py`). It
  embeds the source as a zip, so a hand-edited copy could silently train with a stale featuriser.
  Regenerate it after changing anything in `cgcnn_scratch/` or `scripts/`.
- **`PINK.pdf`** is not redistributed here; get it from the DOI above.

## Stage 2: from moduli to κ_L

`scripts/07_predict_kappa.py` runs the second half of PINK — a closed-form Slack-model physics
formula, not machine-learned — on our own predicted moduli:

```bash
python scripts/07_predict_kappa.py          # → results/pink_kappa_predictions.csv
python scripts/08_compare_kappa.py          # validates it against the paper's own pipeline
```

**The physics is not new — it's the same formula `pink_predict.py` already implements** (sound
velocities → Debye temperature → Grüneisen parameter → κ_L, Eq. 2 of the paper), reused rather than
reimplemented: `slack_physics()` is a numpy rewrite verified bit-for-bit identical to
`pink_predict.py`'s `add_physics()` (`--self-test` checks this directly, to machine precision).
What's new is *what drives it* — our own from-scratch CGCNN's predictions, not the paper's
pre-trained one — and a genuine addition the paper's single-model pipeline has no equivalent of:

**Uncertainty propagation.** The ensemble already gives every crystal a bulk/shear modulus spread
(std across 3 independently-seeded models, in log space) — a free confidence signal a single
pre-trained model can't produce. `07_predict_kappa.py` propagates it: 2,000 Monte Carlo draws of
(K, G) per crystal from that spread, each pushed through the identical physics, giving every κ_L
prediction a `p05`–`p95` interval rather than a bare point estimate.

**That propagation surfaced a real finding, not a decoration.** Interval width correlates with
shear-modulus disagreement (Spearman r = 0.98) far more than with bulk (r = 0.51) — because κ_L
depends on G *through* the sound-velocity term as well as directly, so a modest shear-ensemble
disagreement amplifies into a much wider κ_L range. Median interval width is a modest 1.9×, but the
tail is real: the widest crystal in the set spans nearly 900× between its 5th and 95th percentile,
and every such case traces back to an unusually large `G_VRH_spread_log10` — exactly the honest
signal an uncertainty estimate is supposed to produce.

### Validation against the paper's own pipeline

Real κ_L measurements are rare and expensive, which is why PINK predicts it in the first place —
so before turning to the small amount of real experimental data that does exist (next section), the
cheapest check available is the one this repo has pointed at since stage 1: run the paper's own
pre-trained CGCNN through the identical physics on the identical 1,213 crystals, and see whether two
independently-trained models agree.

| | Value |
|---|---|
| Pearson r (log₁₀ κ_L) | 0.932 |
| Spearman rank correlation | 0.926 |
| MAE | 0.169 log₁₀ (≈47% typical multiplicative error) |
| Overlap in the 20 lowest-κ_L candidates | 10/20 |

Rank correlation is reported alongside Pearson because PINK's actual use case is *screening* —
getting the ordering right matters as much as matching the paper's absolute number. 0.93 on both
counts, from two models that share nothing but architecture and the physics formula, is strong
agreement. The 47% MAE is larger than stage 1's own bulk/shear error because κ_L is a **product of
several quantities each carrying their own error** — it is not a new weakness so much as the
compounding of stage 1's already-documented one. `results/kappa_validation.png` is the parity plot;
`results/kappa_comparison.csv` the full merged table.

The uncertainty-calibration check is honestly weak (Spearman r = 0.20 between predicted interval
width and disagreement with the reference) — expected, since "disagreement with an independently
trained reference model" is a noisy proxy that also carries the reference model's own
idiosyncrasies, not a clean measure of our own error. Weak-but-positive is the correct, unexciting
answer here; reporting it any other way would be overselling a check that was never going to be a
strong one given what it can actually observe.

`pink_predict.py` and its dependency on `external/AI4Kappa` remain as the reference implementation
(needs the authors' code and checkpoints — `git clone https://github.com/ikhalid-dev/AI4Kappa.git`
into `external/`, not vendored here).

### Validation against real experimental κ_L (the paper's Table 1)

The check above compares two models against each other. The paper itself supplies something
stronger: **Table 1** of Liu *et al.* lists 46 real materials with room-temperature
*experimentally measured* κ_L, mp-ids included, alongside the paper's own predicted κ_PINK for
each. 45 of these 46 structures were fetched from the Materials Project (GaP's `mp-3490` has since
been deprecated/merged and couldn't be retrieved) and pushed through our own pipeline unchanged —
the same `04_predict_moduli.py` and `slack_physics()` used everywhere else, not a parallel
implementation that could quietly diverge.

```bash
python scripts/09_fetch_validation_set.py   # fetches the 45 CIFs, runs our pipeline on them
python scripts/10_validate_table1.py        # the comparison below
```

**Provenance, checked before anything else.** 43 of these 45 materials turned out to already be
inside the matbench training set our ensemble learned from (the same formula-bucket-then-
`StructureMatcher` check used throughout this project, against the live matbench structures — not
assumed from a material's absence in `complete-data/`). Only two — LiF and Bi₂Te₃ — are genuinely
unseen. Every number below is reported for both groups; the 43 are recall, not generalisation, and
are never blended into a headline figure that would overstate what they prove.

| | Ours | PINK (paper) |
|---|---|---|
| MAE, all 45 (log₁₀) | 0.427 | 0.227 |
| MAE, 43 in matbench training (log₁₀) | 0.437 | 0.229 |
| MAE, 2 genuinely unseen (log₁₀) | 0.196 | 0.172 |
| Head-to-head, closer to experiment | 8/45 | 37/45 |

Read at face value, the paper's own numbers win clearly. That didn't match what every earlier
section of this document found — our moduli ensemble matches or beats the paper's on the matbench
test set — so a sudden, large gap here demanded an explanation rather than a shrug.

**Localising the gap instead of stopping at one number.** Table 1 also lists the paper's own
predicted shear modulus and sound velocity for each material, so each pipeline stage can be checked
independently instead of only ever seeing one aggregate κ_L number:

- **Shear modulus**: our predictions agree closely with the paper's own (MAE 0.056 log₁₀, Pearson
  r = 0.988). Not the source of the gap.
- **Sound velocity** (folds in our bulk modulus K, which Table 1 has no separate column for): also
  close agreement (r = 0.991). K is fine too.
- **Grüneisen parameter**: our formula-derived γ (from predicted K/G via the standard Slack-model
  relation) differs from the paper's own tabulated γ by a mean of +0.42, systematically — our γ runs
  2–3.8× too high specifically for covalent semiconductors (AlAs, GaAs, InSb, ZnTe, CdTe, ...).
  **This is where the gap lives.**

The paper's own Results text explains why: Table 1's Grüneisen parameters "were obtained from the
AFLOW database and experimental data" — real, looked-up values, not ones derived from predicted
elastic constants. Neither our pipeline nor the paper's own released `pink_predict.py` (confirmed
directly by inspection — no AFLOW lookup anywhere in it) has access to this at genuine screening
scale, where no literature γ exists for a hypothetical GNoME candidate.

**The decisive check**: substituting the paper's own tabulated γ into our pipeline — K, G, and
structure otherwise unchanged — drops our MAE from 0.427 to **0.226 log₁₀**, matching the paper's
own 0.227 almost exactly.

**The honest reading**: once both pipelines are compared on equal footing (same γ source), our
from-scratch model is competitive with the paper's own pre-trained one against real experimental
κ_L. Table 1's headline accuracy benefits from a manual, non-scalable input that the paper's own
actual use case — a 377,221-candidate high-throughput screen — cannot have either. Both pipelines,
run the way they'd actually be run for screening, land in the same place. This is a materially
different conclusion from "the paper's model beats ours," and it was only found by checking each
intermediate quantity rather than accepting one aggregate number.

`results/table1_validation.png` is the parity plot against real κ_exp; `results/table1_gamma_diagnostic.png`
shows the two diagnostic panels (shear modulus agreement vs. Grüneisen disagreement) side by side;
`results/table1_comparison.csv` is the full merged table.

## Trying a second architecture: ALIGNN

CGCNN's convolution only ever sees bond *lengths* (the edges of the crystal graph). ALIGNN
(Choudhary & DeCost, 2021) adds a second, "line" graph over bond *angles* — the one piece of local
geometry a plain bond graph cannot represent at all — so it's a natural architecture to check PINK's
own recommended one (coGN, which was ruled out early: it isn't an installable package on its own and
realistically needs a working TensorFlow install alongside this project's existing torch pipeline,
the same class of framework-coexistence risk documented below).

**The comparison only means something if it's on the same data and the same split.** ALIGNN's own
`split_seed` shuffles with Python's stdlib `random`, not numpy's `RandomState` — a different RNG
algorithm entirely, so "the same seed number" would silently produce a *different* partition than the
one the CGCNN ensemble trained on. So the split is computed exactly once, with this project's own
`split_indices()` (imported directly, not re-derived), and ALIGNN is told `keep_data_order=True` —
"the order you were given already **is** train/val/test," rather than trying to make two different
RNGs agree.

```bash
python scripts/11_prepare_alignn_data.py     # matbench -> ALIGNN's format, same split as CGCNN
python scripts/12_train_alignn.py --target bulk_modulus_kv
python scripts/12_train_alignn.py --target shear_modulus_gv
```

**Status: infrastructure built and verified, full training run not yet done.** Both scripts ran
successfully end-to-end in a local smoke test (2–3 epochs on 60–90 crystals) — but ALIGNN builds a
line graph on top of the bond graph, real extra work per crystal per epoch that CGCNN never does, so
a full ~150-epoch run on all 7,691 training crystals needs a GPU. `colab/PINK_ALIGNN_colab.py`
generates `colab/PINK_ALIGNN.ipynb`, mirroring the CGCNN notebook's pattern (source embedded as a
zip so it can't drift from what's actually committed) — running it top to bottom is the next step,
not yet done in this session.

**Getting to a working smoke test surfaced three real bugs, worth recording rather than glossing
over:**

- **ALIGNN's own release notes say it dropped its DGL dependency; its actual installed code
  disagrees with its own notes.** `pip install alignn` pulls in no `dgl`, and the default training
  path (`model="alignn_atomwise"`, `neighbor_strategy="k-nearest"`) calls literal `dgl.graph(...)`
  regardless — confirmed by running it and reading the traceback, not by re-reading the release
  notes more carefully. The fix, found by reading `alignn/graphs.py` directly: pass
  `neighbor_strategy="pure_torch"` (routes through a genuinely dgl-free graph builder) together with
  `model="alignn_atomwise_pure"` (the model variant written to consume that builder's output) — both
  together, since either alone leaves a DGL-shaped object meeting code that doesn't expect one.
- **Installing `dgl` separately to sidestep the above was tried and reverted.** Its prebuilt wheel
  for this exact torch build fails at import with a symbol-not-found error from its compiled
  extension — an ABI mismatch, a different flavour of the same "two compiled extensions fighting
  over one runtime" problem already documented for MKL/`libiomp5` elsewhere in this project. The
  pure-torch path above avoids needing `dgl` at all, which is the better fix anyway.
- **The model's default `atom_input_features=1` doesn't match the data.** `get_train_val_loaders`'s
  default atom featuriser produces a 92-element one-hot vector (elements 1–92 — the same convention
  this project's own `atom_init.json` uses), so the model needs `atom_input_features=92` or its
  first linear layer's shapes don't line up. Caught by the shape-mismatch error it throws, not by
  reading either side's source closely enough to predict it in advance.

None of these are exotic — they're exactly the class of doc-vs-code mismatch this project has run
into before (the paper's own "two hidden layers" vs. its checkpoint's `n_h=1`), and exactly why every
claim in this project gets checked by running something, not by reading about it.

## The actual screen: GNoME, end to end

Everything above runs on 1,213 or 45 crystals. The paper's real headline result is a 377,221-material
high-throughput screen of Google DeepMind's [GNoME](https://github.com/google-deepmind/materials_discovery)
discovery database, filtered down to candidates with κ_L ≤ 1 W/m/K. This project had never run that
screen — only ever done inference on the same small, curated CIF sets. This section does it.

```bash
curl -o gnome_data/stable_materials_summary.csv https://storage.googleapis.com/gdm_materials_discovery/gnome_data/stable_materials_summary.csv
curl -o gnome_data/by_composition.zip        https://storage.googleapis.com/gdm_materials_discovery/gnome_data/by_composition.zip
python scripts/13_screen_gnome.py            # ~25 min on this laptop's CPU
python scripts/14_compare_gnome_screen.py    # compares against the paper's own published candidates
```

Both files are fetched anonymously over plain HTTPS from GNoME's public GCS bucket — no `gcloud`/
`gsutil` needed. `gnome_data/` is gitignored (620 MB combined, trivially re-downloadable).

**The snapshot has grown since the paper screened it — stated once here, true everywhere this result
is quoted.** This project's download (2026-08-09) has **554,054** stable materials; the paper's own
screen used **377,221**. There is no way to pull the exact historical snapshot the paper used, only
today's — so this is *a current GNoME screen using the paper's stated criteria*, not a bit-identical
replay. The funnel below is checked against the paper's own for plausibility (same order of
magnitude, scaled for the ~1.47× larger corpus), never claimed to match exactly.

| Stage | Ours (554,054-material snapshot) | Paper's (377,221-material snapshot) |
|---|---|---|
| Bandgap [0.1, 3.0] eV + decomposition energy ≤ 0 | 37,889 | 30,199 |
| + no radioactive elements | 33,323 | 26,305 |
| + κ_L ≤ 1 W/m/K (our/their model) | **16,299** | **11,869** |

Structures are matched from GNoME's `by_composition.zip` (554,054 CIFs, one per material) by the
summary CSV's `Composition` column — checked directly, not assumed: every one of the 554,054 zip
filenames matches a `Composition` value in the CSV, one-to-one. All 33,323 candidate CIFs are read
straight out of the zip and converted to graphs in memory — never written to disk individually, same
reasoning as `01b_prepare_full_dataset.py`'s graph caching.

### Comparison against the paper's own 11,869 published candidates

The paper's actual screening output ships in the cloned reference repo
(`external/AI4Kappa/JMI_Supporting_Information/Nature-filtered-low-kappa.csv`) — real, checked data,
not something taken on faith. It carries no material ID, only a reduced formula, so overlap is
necessarily checked by formula (verified to be the *same* pymatgen convention both sources use — zero
formatting mismatches in a 200-row spot check — so this isn't a stringification artifact deflating
the count).

**70.8% of the paper's own published candidates also appear in our independently-derived list** — an
independently-trained CGCNN ensemble, run on a different (larger, newer) GNoME snapshot, using the
exact same physics, agrees with more than two-thirds of what the paper's own model flagged. (51.6% in
the other direction — of *our* 16,299, just over half also appear in the paper's smaller/older
snapshot; the rest are, in large part, candidates that simply didn't exist yet in the paper's
snapshot, not disagreement.)

`results/gnome_kappa_distributions.png` overlays both candidate lists' κ_L distributions — similar
overall shape, both spanning the full 0–1 W/m/K range, ours somewhat more concentrated around
0.25–0.4. `results/gnome_overlap.csv` is the full set of 8,405 shared-formula rows.

**The part a point-estimate pipeline cannot do at all**: every one of our candidates carries a Monte
Carlo κ_L interval from the ensemble's own K/G spread (`results/gnome_overlap_confidence.png`). Of the
8,405 candidates both screens agree on:

| Confidence (p95/p05 ratio) | Count | Meaning |
|---|---|---|
| Tight (< 2×) | 1,770 | confidently low-κ |
| Moderate (2–5×) | 4,739 | reasonable confidence |
| Wide (> 5×) | 1,896 | worth a DFT check before trusting |

The paper's own pipeline has no equivalent of this list at all — it can name candidates, but not say
which ones it's actually sure about. `results/gnome_screen_all.csv` is the full 33,323-candidate
scored pool (pre-threshold); `results/gnome_screen_candidates.csv` is the 16,299 that pass κ_L ≤ 1.
