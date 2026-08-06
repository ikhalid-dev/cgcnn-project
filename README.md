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
| PINK paper | ≈0.07 |
| Best published on this benchmark ([coGN](https://matbench.materialsproject.org/)) | ≈0.054 |

The bulk ensemble slightly beats the paper it reproduces and sits between it and the state of the
art. Full breakdown in [`results/RESULTS.md`](results/RESULTS.md).

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
| `run_pipeline.sh` | Everything above, in order |
| `results/` | Checkpoints, metrics, figures, predictions |
| `results/archive/` | One superseded run, kept for comparison (see below) |
| `complete-data/` | The 1,213 Materials Project CIFs (prediction set) |
| `data_full/` | Labels + provenance mapping for the 10,987 training crystals |
| `colab/`, `kaggle/` | GPU runners (see `docs/method.pdf` §7) |
| `docs/` | The method write-up and its LaTeX source |
| `pink_predict.py` | Separate: full κ_L inference using the *paper's* pre-trained weights |

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

## Stage 2: κ_L (not yet reimplemented)

`pink_predict.py` runs the complete PINK pipeline using the *paper's* pre-trained weights, so it
serves as the reference implementation to check ours against:

```bash
python pink_predict.py <cif_dir> [-o out.csv]
```

It needs the authors' code and checkpoints, which are not vendored:

```bash
mkdir -p external && cd external
git clone https://github.com/ikhalid-dev/AI4Kappa.git
```

Substituting our `pink_moduli_predictions.csv` into the Slack-model half is the natural next step.
