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
| `scripts/01_prepare_dataset.py` | Match raw CIFs to DFT elastic moduli → labelled training set |
| `scripts/02_train.py` | Train a CGCNN from scratch for one modulus |
| `scripts/03_evaluate.py` | Metrics, prediction CSVs, and figures |
| `data/` | `labels.csv` (278 labelled crystals) + the matched CIFs |
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
