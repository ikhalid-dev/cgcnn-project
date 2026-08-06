# CGCNN / PINK

Work based on **PINK** — Liu, Y.; Wang, X.; Hao, Y.; Li, X.; Sun, J.; Lookman, T.; Ding, X.; Gao, Z.
*PINK: physical-informed machine learning for lattice thermal conductivity.*
J. Mater. Inf. 2025, 5, 12. [doi:10.20517/jmi.2024.86](https://dx.doi.org/10.20517/jmi.2024.86)
(paper included as `PINK.pdf`)

PINK predicts **lattice thermal conductivity (κ_L)** directly from CIF files by combining a CGCNN
that predicts elastic moduli with a simplified Slack-model formula:

```
CIF ──► CGCNN ──► bulk modulus B, shear modulus G
             └──► volume V, atom count n, density ρ (pymatgen)
                        │
                        ▼
        sound velocities (v_l, v_t, v_s) ──► Poisson ratio ──► Grüneisen γ
                        │
                        ▼
                      κ_L
```

## Layout

| Path | What it is |
|---|---|
| `cgcnn_scratch/` | **Our own CGCNN implementation** — heavily commented `model.py` and `data.py` |
| `scripts/01_prepare_dataset.py` | Match raw CIFs to DFT elastic moduli → small labelled set (278) |
| `scripts/01b_prepare_full_dataset.py` | Build the **full** matbench training set (10,987) as cached graphs |
| `scripts/02_train.py` | Train a CGCNN from scratch for one modulus |
| `scripts/03_evaluate.py` | Metrics, prediction CSVs, and figures |
| `scripts/04_predict_moduli.py` | Predict K and G for all 1,213 PINK CIFs → stage-2 input |
| `data/` | `labels.csv` (278 labelled crystals) + the matched CIFs |
| `data_full/` | `labels.csv` (10,987), `graphs.pt` cache, `mp_to_mb.csv` provenance |
| `results/` | Trained checkpoints, per-epoch history, predictions, figures |
| `pink_predict.py` | Separate: PINK inference using the *paper's* pre-trained models |
| `complete-data/` | 1,213 Materials Project CIFs (raw input) |
| `Complete_model.ipynb` | Reference CGCNN classification tutorial (not the PINK pipeline) |
| `PINK.pdf` | The paper |
| `external/` | Cloned upstream repos — gitignored, see below |

## Stage 1: train a CGCNN from scratch (current work)

Predict bulk (`K_VRH`) and shear (`G_VRH`) modulus directly from CIFs, using only
the small local dataset. Run in order:

```bash
python scripts/01_prepare_dataset.py            # ~5 min, once
python scripts/02_train.py    --target K_VRH    # ~2 min
python scripts/03_evaluate.py --target K_VRH
python scripts/02_train.py    --target G_VRH
python scripts/03_evaluate.py --target G_VRH
```

### Where the labels come from

A CIF says where the atoms are; it does not say what the modulus is. Supervised
training needs both. `01_prepare_dataset.py` matches our CIFs against the
**matbench** elastic benchmarks (`matbench_log_kvrh` / `matbench_log_gvrh`,
10,987 DFT-computed entries — the same datasets the paper trained on).

Matbench strips the Materials Project IDs, so the join is done on *structure*:
bucket by reduced formula first (cheap), then confirm with pymatgen's
`StructureMatcher` (symmetry-aware). **278 of the 1,213 CIFs match.** The rest
simply never had their elastic tensor computed — matbench covers ~11k of the
~150k materials in MP.

### Results on 278 crystals (195 train / 42 val / 41 test)

| Target | Test MAE, log10(GPa) | Test R² | Test MAE, GPa |
|---|---|---|---|
| `K_VRH` (bulk) | 0.152 | 0.57 | 20.2 |
| `G_VRH` (shear) | 0.163 | 0.63 | 11.4 |

For scale, the paper's CGCNN reaches ≈0.07 log10(GPa) — but on 10,987 crystals,
40× more data. Roughly 2× worse error from 40× less data is the expected
trade, and the train/test gap (0.08 vs 0.15) shows the model is overfitting,
as a 26k-parameter network on 195 samples must.

Figures land in `results/`: `parity_*.png` (predicted vs DFT),
`training_*.png` (loss and MAE per epoch), `residuals_*.png` (error
distribution and error vs stiffness).

## Stage 1b: scale to the full matbench set (10,987 crystals)

The 278-crystal experiment above is limited by **labels, not structures**. Our
1,213 CIFs are fine; only 278 of them have ever had an elastic tensor computed.
Intersecting the two sets throws away 97% of the available labels.

So stop treating the 1,213 CIFs as the universe. The labels live in matbench, so
train on *all* of matbench — the same 10,987 crystals the paper used — and let
our 1,213 CIFs be what they should always have been: the **prediction** set that
feeds the κ_L stage.

```bash
python scripts/01b_prepare_full_dataset.py      # ~6 min, once
python scripts/02_train.py --target K_VRH --data-dir data_full --tag K_VRH_full \
    --epochs 150 --batch-size 128 --lr 0.01 --atom-fea-len 64 --h-fea-len 128
python scripts/03_evaluate.py --target K_VRH --data-dir data_full --tag K_VRH_full
# ...and the same two commands with G_VRH / G_VRH_full
python scripts/04_predict_moduli.py             # → results/pink_moduli_predictions.csv
```

Two things to know about how this is built:

**Graphs are cached, not re-parsed.** The slow step in the whole pipeline is
turning a crystal into a graph — a periodic neighbour search per atom. Rather
than round-trip 10,987 matbench `Structure` objects through `.cif` files on disk
and re-parse them on every run, `01b` converts them once and pickles the tensors
to `data_full/graphs.pt`. Both training runs then load it in seconds. The
conversion goes through the same `structure_to_graph` that `CIFData` uses, so a
cached graph is identical to one built from a CIF.

**`--tag` keeps the two experiments apart.** The big run writes
`model_K_VRH_full.pth`, `parity_K_VRH_full.png` and so on, so the 278-crystal
results stay on disk for comparison instead of being overwritten.

### Predictions carry their provenance

278 of the 1,213 CIFs *are* in matbench, so the model was trained on them. A
prediction for one of those is recall, not generalisation, and quoting it as
evidence would be circular. Every row of `pink_moduli_predictions.csv` therefore
carries a `provenance` column:

| Value | Meaning |
|---|---|
| `train` / `val` | the model was fitted on this crystal — not evidence |
| `test` | in matbench, held out during training — genuine evidence |
| `unseen` | not in matbench at all; no DFT reference exists. This is the model doing its actual job |

Rows also carry `K_VRH_dft` / `G_VRH_dft` where a reference exists, so the error
on the 278 can be inspected directly rather than taken on trust.

## Stage 1c: pushing the error down with an ensemble

Three models per target, averaged. Different random initialisations land in
different minima; they agree about the signal and disagree about their own
noise, so averaging keeps the first and partially cancels the second. Worth
10–15% off the error in practice.

```bash
# members 2 and 3 - note --split-seed is PINNED while --seed varies
python scripts/02_train.py --target K_VRH --tag K_VRH_s1 --data-dir data_full \
    --seed 1 --split-seed 42 --n-conv 4 --scheduler cosine --epochs 200 \
    --batch-size 128 --lr 0.01 --atom-fea-len 64 --h-fea-len 128
python scripts/05_ensemble.py --target K_VRH --data-dir data_full \
    --tags K_VRH_full,K_VRH_s1,K_VRH_s2
```

**`--seed` and `--split-seed` are deliberately separate.** `--seed` controls
weight initialisation and batch order — vary it, that is what makes members
differ. `--split-seed` controls the train/val/test split and must be *identical*
across members: if member A trained on a crystal member B held out, the
ensemble's "test" score is partly measured on training data. `05_ensemble.py`
refuses to combine members whose test splits disagree rather than trusting the
caller to have got this right.

Ensembling also averages **in log space, not in GPa**. The models are trained
against a log-space loss, so that is where their errors are symmetric. Two
members predicting 10 and 1000 GPa average to 100 GPa in log space and 505 GPa
in linear space — and 100 is the defensible answer when members disagree that
badly. The per-crystal spread between members is carried through to the output
as a free uncertainty estimate.

### How low can the error actually go?

The model predicts log₁₀(modulus), so MAE converts to a *multiplicative* error
of `10^MAE − 1`. Both numbers are reported in `metrics_*.csv` as
`rel_error_pct` and `pct_of_range`.

| | MAE log₁₀(GPa) | Relative error |
|---|---|---|
| 278-crystal model | 0.152 | 42% |
| 10,987-crystal model | ≈0.07 | ≈17% |
| PINK paper | ≈0.07 | ≈17% |
| Best published on this benchmark (coGN) | ≈0.054 | ≈13% |

**A ~2% relative error is not attainable here, by anyone.** The targets are
DFT-computed elastic moduli, and DFT elastic constants themselves disagree with
experiment by roughly 5–15%. A model cannot be more accurate than the labels it
is fitted to, so 2% is below the noise floor of the training data — a model
reporting it would be revealing a leak, not an achievement. The realistic floor
for this architecture is ~0.06 log₁₀, about 15%.

Note that `pct_of_range` (MAE in GPa over the full 1–575 GPa span) *does* come
in under 2%. It is a legitimate normalised-MAE convention, but it flatters the
model — a wide data range shrinks it for free — so it should only ever be quoted
next to `rel_error_pct`, never instead of it.

## Running it on a GPU (Colab)

This laptop is an **Intel i5-6360U** — a 2016 dual-core 15 W ultrabook chip with
no usable GPU (macOS 12 + torch 2.2 report `mps.is_available() == False`). At
~26–34 s/epoch it takes about **9 hours** to train six ensemble members. A Colab
T4 does the same work in roughly **30–60 minutes**.

```bash
python colab/PINK_CGCNN_colab.py     # regenerates colab/PINK_CGCNN.ipynb
```

Upload the notebook to [Colab](https://colab.research.google.com), set
**Runtime → Change runtime type → T4 GPU**, and run it top to bottom.

**The notebook is generated, not hand-maintained.** It embeds the current
`cgcnn_scratch/` and `scripts/` as a base64 zip, because Colab cannot reach this
private repo. Regenerate it after touching either — a stale embedded copy would
train with the wrong featuriser and produce a model whose predictions look
plausible and are wrong. Nothing is uploaded by hand, and the 240 MB graph cache
is not transferred at all: the notebook rebuilds it from matbench in ~2 minutes,
which is faster than pushing it over a network.

`04_predict_moduli.py` stays on the laptop — it needs the 1,213 local CIFs,
which never leave it. Checkpoints are always serialised on CPU, so GPU-trained
weights load here without CUDA.

## Setup

Upstream repos are not vendored. Clone them into `external/`:

```bash
mkdir -p external && cd external
git clone https://github.com/ikhalid-dev/AI4Kappa.git            # PINK authors' code + pre-trained models
git clone https://github.com/ikhalid-dev/materials_discovery.git # GNoME (code only; CIFs are in a GCS bucket)
```

`AI4Kappa` supplies everything the pipeline actually needs: the modified `cgcnn/` package, the two
pre-trained checkpoints under `model/`, and `root_dir/atom_init.json`.

### Python environment

Use the **`ml_env` conda environment (Python 3.11)** — not the system `python3` (3.14), which
cannot install AI4Kappa's pinned `torch==2.2.0` / `pymatgen==2023.11.12`.

```bash
/Users/mac/miniconda3/envs/ml_env/bin/python pink_predict.py complete-data -o results.csv
```

Only `streamlit` is missing from `ml_env`, and it is needed solely to run the original web app.

## Resuming work after a break

Everything except `external/` is committed, so a fresh clone is almost ready to go.

```bash
cd "/Users/mac/Desktop/Cgcnn project"
git pull
git log --oneline | head -5     # where did we stop?
cat results/metrics_*.csv       # current scores
```

**What you do NOT need to re-run:**

| | Why |
|---|---|
| `01_prepare_dataset.py` | `data/labels.csv` and the 278 matched CIFs are committed. The match takes ~5 min and is deterministic. |
| `02_train.py` | `results/model_*.pth` are committed (~120 KB each). |

So to regenerate every figure from scratch takes seconds, not minutes:

```bash
python scripts/03_evaluate.py --target K_VRH
python scripts/03_evaluate.py --target G_VRH
```

**What a fresh clone does need restoring:**

```bash
# 1. the upstream repos (gitignored)
mkdir -p external && cd external
git clone https://github.com/ikhalid-dev/AI4Kappa.git
cd ..

# 2. the notebook-output filter (git filters are per-clone, not stored in the repo)
python -m nbstripout --install --attributes .gitattributes
```

Only `AI4Kappa` is genuinely required, and only by `pink_predict.py`, which uses the
paper's pre-trained weights. The from-scratch pipeline in `cgcnn_scratch/` has no
dependency on `external/` at all.

### Where the work stands

Stage 1 is done — both moduli train and evaluate end to end. The open thread is that
**residuals fan out badly below ~30 GPa**: the model is least reliable on soft
materials, which is precisely the ultralow-κ regime this project cares about. That is
the natural next thing to attack, before moving to stage 2.

## Stage 2 (later): κ_L from the predicted moduli

Once the modulus models are good enough, their outputs feed the Slack-model
physics to give κ_L. `pink_predict.py` already does this end to end using the
*paper's* pre-trained models, so it doubles as the reference implementation to
check our own models against:

```bash
python pink_predict.py <cif_dir> [-o out.csv] [-n LIMIT] [--batch-size N]
```

Output columns: bulk/shear modulus, atom count, volume, density, atomic mass, longitudinal and
transverse sound velocities, speed of sound, acoustic Debye temperature, Poisson ratio, Grüneisen
parameter, and κ_L from both the classic Slack model and Equation (2) of the paper. Rows are sorted
ascending by κ_L, so ultralow-κ candidates come first.

## Gotchas

- **Import `torch` before `numpy`/`pandas`/`pymatgen`.** In this conda env MKL loads its own OpenMP
  runtime first, and the duplicate `libiomp5.dylib` segfaults torch. `pink_predict.py` already
  orders its imports for this.
- **AI4Kappa's `CIFData` is not stock CGCNN.** It skips a header row in `id_prop.csv` and treats
  `cif_id` as the *full filename*, so entries must keep the `.cif` suffix. A headerless file
  silently drops the first structure.
- **Notebook outputs are stripped on commit** via `nbstripout` (configured in `.gitattributes`).
  After cloning fresh, re-enable it with `python -m nbstripout --install --attributes .gitattributes`.

## Getting the GNoME dataset

The `materials_discovery` repo contains code only. The 377,221 CIFs the paper screens live in a
public bucket and are a separate, large download:

```bash
gcloud storage cp --recursive gs://gdm_materials_discovery/ data/
# or: python external/materials_discovery/scripts/download_data_wget.py
```
