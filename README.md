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
| `pink_predict.py` | Batch CLI: a directory of CIFs → CSV of moduli, sound velocities, γ, and κ_L |
| `complete-data/` | 1,213 Materials Project CIFs |
| `Complete_model.ipynb` | Reference CGCNN classification tutorial (not the PINK pipeline) |
| `PINK.pdf` | The paper |
| `external/` | Cloned upstream repos — gitignored, see below |

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

## Usage

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
