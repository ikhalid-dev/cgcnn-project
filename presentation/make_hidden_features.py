#!/usr/bin/env python3
"""
What is INSIDE a trained CGCNN?  A map of its hidden features.

The network reads a crystal and, just before the very last layer, holds ONE
vector of 128 numbers that summarises that crystal. The last layer is a
plain weighted sum:

    prediction = w1*h1 + w2*h2 + ... + w128*h128 + b

so those 128 numbers ("hidden features") are everything the network knows
about the crystal when it answers. This script:

  1. loads the trained bulk-modulus model (results/cgcnn/model_K_VRH_full.pth,
     seed 42 - the "CGCNN, 1 model" row of the audit table)
  2. runs it over the 1,648 TEST crystals it never trained on, and copies the
     128-number vector out of the network on the way through (a "hook")
  3. GATE: checks the model's predictions here equal the saved ones in
     results/cgcnn/predictions_K_VRH_full.csv - proof that the right model,
     the right crystals and the right graphs were loaded
  4. squashes 128 numbers to 2 with PCA so they can be drawn on a page
  5. a "linear probe": can the SHEAR modulus G - which this model was never
     shown - be read off the 128 numbers with a plain weighted sum? And is
     that any better than reading G off the model's one predicted K? (It is
     barely better: G and K rise and fall together, so most of the "G" in
     the features is just K. Checked here, not assumed.)

Outputs
    figures/hidden_features.png        the three-panel figure for the deck
    figures/hidden_features_test.csv   one row per test crystal

Run with (infer_env is the env with a working torch):
    OMP_NUM_THREADS=1 ~/miniconda3/envs/infer_env/bin/python make_hidden_features.py
"""
import os
import re
import sys

# torch must be imported BEFORE numpy/pandas on this machine, or two copies
# of the OpenMP library get loaded and the process aborts (OMP Error #15).
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)          # so "import cgcnn_scratch" finds the project's code
from cgcnn_scratch.data import GraphCacheData, Normalizer, collate_pool  # noqa: E402
from cgcnn_scratch.model import CrystalGraphConvNet                     # noqa: E402

FIGS = os.path.join(HERE, "figures")
CKPT = os.path.join(ROOT, "results", "cgcnn", "model_K_VRH_full.pth")
SAVED = os.path.join(ROOT, "results", "cgcnn", "predictions_K_VRH_full.csv")
LABELS = os.path.join(ROOT, "data_full", "labels.csv")
GRAPHS = os.path.join(ROOT, "data_full", "graphs.pt")

NAVY, BLUE, GREEN, AMBER, RED = "#0F1B33", "#1450AA", "#1E8E5A", "#D97B12", "#B02A2A"
PURPLE, INK, MUTED, GRID, SUBTLE = "#7A3FA0", "#1F2937", "#5B6B82", "#E1E6EF", "#AAB6C8"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# =============================================================================
#  1. the model, rebuilt exactly as it was trained
# =============================================================================
# weights_only=False: this checkpoint stores plain Python settings next to
# the weights, and we wrote it ourselves, so it is safe to unpickle.
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
a = ckpt["args"]                  # the settings it was trained with (a dict)
model = CrystalGraphConvNet(ckpt["feature_lens"]["orig_atom_fea_len"],
                            ckpt["feature_lens"]["nbr_fea_len"],
                            atom_fea_len=a["atom_fea_len"], n_conv=a["n_conv"],
                            h_fea_len=a["h_fea_len"], n_h=a["n_h"])
model.load_state_dict(ckpt["state_dict"])
model.eval()                      # inference mode: batch-norm uses its saved averages
normalizer = Normalizer(torch.zeros(1))
normalizer.load_state_dict(ckpt["normalizer"])
print(f"model: {a['tag']}, {a['n_conv']} conv layers, atom vectors {a['atom_fea_len']} wide, "
      f"hidden layer {a['h_fea_len']} wide")

# =============================================================================
#  2. the hook - copy the 128 numbers going INTO the last layer
# =============================================================================
# A "forward hook" is a small function torch calls every time a layer runs.
# It receives (layer, inputs, output). inputs is a tuple, so inputs[0] is the
# (batch, 128) tensor of hidden features. We append a copy to a list.
captured = []
model.fc_out.register_forward_hook(lambda layer, inputs, output: captured.append(inputs[0].detach()))

# =============================================================================
#  3. the test crystals, from the same graph cache training used
# =============================================================================
saved = pd.read_csv(SAVED)
test = saved[saved.split == "test"]
labels = pd.read_csv(LABELS).set_index("mb_id")     # rows can now be looked up by id
# targets: id -> log10 K. Only used to fill the dataset's label slot.
targets = {i: float(np.log10(labels.loc[i, "K_VRH"])) for i in test.material_id}
print(f"loading {GRAPHS} ...")
data = GraphCacheData(GRAPHS, targets, ids=list(test.material_id))
loader = DataLoader(data, batch_size=128, shuffle=False, collate_fn=collate_pool)

ids, preds = [], []
with torch.no_grad():             # no gradients: we are only reading the network
    for inputs, _target, batch_ids in loader:
        out = model(*inputs)      # *inputs unpacks the 4-tuple into 4 arguments
        preds.append(normalizer.denorm(out).squeeze(1))
        ids += batch_ids
hidden = torch.cat(captured).numpy()      # (1648, 128)
pred_log = torch.cat(preds).numpy()       # (1648,) log10 K in GPa
print(f"hidden features: {hidden.shape[0]} crystals x {hidden.shape[1]} numbers")

# =============================================================================
#  4. GATE - these predictions must equal the saved ones
# =============================================================================
ref = test.set_index("material_id").loc[ids, "pred_log10"].to_numpy()
worst = float(np.max(np.abs(pred_log - ref)))
print(f"GATE: max |this run - saved prediction| = {worst:.2e} log10 over {len(ids)} crystals")
if worst > 1e-4:
    raise SystemExit("GATE FAILED - wrong model, crystals or graphs; nothing below is trustworthy")

# =============================================================================
#  5. one row per crystal: truth, chemistry family
# =============================================================================
def family(formula):
    """Crude chemistry family from the element symbols in the formula.

    re.findall(r"[A-Z][a-z]?", "Ca(AgGe)2") -> ["Ca", "Ag", "Ge"]:
    a capital letter, optionally followed by one lower-case letter.
    The first rule that matches wins, so an oxyfluoride counts as an oxide.
    """
    el = set(re.findall(r"[A-Z][a-z]?", formula))
    if "O" in el:
        return "oxide"
    if el & {"F", "Cl", "Br", "I"}:       # & = elements in BOTH sets
        return "halide"
    if el & {"S", "Se", "Te"}:
        return "chalcogenide"
    if el & {"N", "P", "As", "Sb"}:
        return "pnictide"
    return "intermetallic / other"


lab = labels.loc[ids]
df = pd.DataFrame({"material_id": ids,
                   "formula": lab.formula.to_numpy(),
                   "family": [family(f) for f in lab.formula],
                   "true_log10_K": np.log10(lab.K_VRH.to_numpy()),
                   "true_log10_G": np.log10(lab.G_VRH.to_numpy()),
                   "pred_log10_K": pred_log})
# a few crystals have G <= 0 (unphysical DFT entries); log10 gives -inf/NaN
ok_G = np.isfinite(df.true_log10_G.to_numpy())

# =============================================================================
#  6. PCA: 128 numbers -> the 2 directions along which crystals differ most
# =============================================================================
# StandardScaler is not used: every hidden feature is already on the same
# scale (softplus outputs), and scaling would inflate near-dead features.
pca = PCA(n_components=2).fit(hidden)
xy = pca.transform(hidden)
var = pca.explained_variance_ratio_ * 100
df["PC1"], df["PC2"] = xy[:, 0], xy[:, 1]
r_pc1 = np.corrcoef(xy[:, 0], df.true_log10_K)[0, 1]
print(f"PCA: PC1 carries {var[0]:.1f}% of the spread, PC2 {var[1]:.1f}%; "
      f"corr(PC1, true log K) = {r_pc1:+.3f}")

# The map is a V. Its tip is the crystal with the lowest PC2; crystals left
# of the tip form one arm, right of it the other. A straight line is fitted
# to each arm (np.polyfit(x, y, 1) = slope and intercept), and "off_arm" is
# how far a crystal sits above its arm's line.
tip = df.PC2.idxmin()                         # row label of the lowest point
df["arm"] = np.where(df.PC1 < df.PC1[tip], "left", "right")
df["off_arm"] = np.nan
for arm, g in df.groupby("arm"):              # g = the rows of one arm
    slope, intercept = np.polyfit(g.PC1, g.PC2, 1)
    df.loc[g.index, "off_arm"] = g.PC2 - (slope * g.PC1 + intercept)
df["abs_err_log10_K"] = (df.pred_log10_K - df.true_log10_K).abs()
OFF = 1.0     # chosen by eye from the plot, NOT tuned - say so wherever it is quoted
far = (df.off_arm > OFF).to_numpy()
for arm, g in df.groupby("arm"):
    print(f"  {arm} arm: {len(g)} crystals, median true K {10 ** g.true_log10_K.median():.0f} GPa")
print(f"  {far.sum()} crystals more than {OFF} off their arm: median |error| "
      f"{df.abs_err_log10_K[far].median():.3f} log10 vs {df.abs_err_log10_K[~far].median():.3f} on the arms; "
      f"{(df.family[far] == 'oxide').sum()} of them oxides")

# =============================================================================
#  7. linear probe: can a weighted sum of the 128 numbers give G?
# =============================================================================
# 5-fold cross-validation: the probe is fitted on 4/5 of the test crystals and
# scored on the remaining 1/5, five times, so every crystal is scored by a
# probe that never saw it. RidgeCV = least squares with a small penalty whose
# strength it picks itself.
def probe(y, mask):
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), hidden[mask], y[mask], cv=cv)
    resid = y[mask] - p
    r2 = 1 - np.sum(resid ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2)
    return p, r2, float(np.mean(np.abs(resid)))


yG = df.true_log10_G.to_numpy()
pG, r2G, maeG = probe(yG, ok_G)
_, r2K, maeK = probe(df.true_log10_K.to_numpy(), np.ones(len(df), bool))

# The BASELINE the probe must beat: G from the model's single predicted K.
# Same probe, same folds, but one input column instead of 128.
# [:, None] turns the (1648,) list into a (1648, 1) table, the shape sklearn wants.
cv = KFold(n_splits=5, shuffle=True, random_state=0)
pK = df.pred_log10_K.to_numpy()[:, None]
base = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), pK[ok_G], yG[ok_G], cv=cv)
r2_base = 1 - np.sum((yG[ok_G] - base) ** 2) / np.sum((yG[ok_G] - yG[ok_G].mean()) ** 2)

# The CEILING: a CGCNN actually trained on G, same test crystals - read from
# the step 95 table (CGCNN single, seed 42), never typed by hand.
m95 = pd.read_csv(os.path.join(ROOT, "results", "cgcnn", "95_final_all_metrics.csv"), low_memory=False)
row = m95[(m95.primary == True) & (m95.family == "CGCNN moduli") & (m95.split == "test")
          & (m95.model == "CGCNN single (seed 42)") & (m95.target == "G_VRH") & (m95.metric == "R2_log10")]
assert len(row) == 1, row
r2_Gmodel = float(row.value.iloc[0])
df["probe_log10_G"] = np.nan
df.loc[ok_G, "probe_log10_G"] = pG
print(f"probe for K (the model's own target): R2 {r2K:.3f}, MAE {maeK:.4f} log10")
print(f"probe for G (NEVER shown to this model): R2 {r2G:.3f}, MAE {maeG:.4f} log10 "
      f"on {ok_G.sum()} crystals")
print(f"  baseline, G from predicted K alone: R2 {r2_base:.3f}  -> the 128 features add {r2G - r2_base:+.3f}")
print(f"  ceiling, a CGCNN trained on G:      R2 {r2_Gmodel:.3f}")
print("family counts:", df.family.value_counts().to_dict())

# =============================================================================
#  8. the figure
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
sc = ax.scatter(df.PC1, df.PC2, c=df.true_log10_K, cmap="viridis", s=7, lw=0)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("true log$_{10}$ K (GPa)", color=MUTED, fontsize=8)
cb.ax.tick_params(labelsize=7, colors=MUTED)
ax.set_title("A   the map, coloured by the model's target (K)", loc="left", fontsize=10, color=INK)

ax = axes[1]
colours = {"oxide": RED, "halide": GREEN, "chalcogenide": AMBER, "pnictide": PURPLE,
           "intermetallic / other": BLUE}
for fam, col in colours.items():
    m = (df.family == fam).to_numpy()
    ax.scatter(df.PC1[m], df.PC2[m], s=7, lw=0, color=col, alpha=0.75, label=f"{fam} ({m.sum()})")
ax.legend(fontsize=7.5, frameon=False, markerscale=2.5, loc="best")
ax.set_title("B   the same map, coloured by chemistry", loc="left", fontsize=10, color=INK)

for ax in axes[:2]:
    ax.set_xlabel(f"PC1 ({var[0]:.0f}% of spread)", color=MUTED, fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]:.0f}% of spread)", color=MUTED, fontsize=9)
    style_axes(ax)

ax = axes[2]
# three horizontal bars: how well G can be predicted from each source
names = ["K model: its 1 predicted K", "K model: all 128 features", "a CGCNN trained on G"]
vals = [r2_base, r2G, r2_Gmodel]
ax.barh([2, 1, 0], vals, color=[SUBTLE, BLUE, GREEN], height=0.6)
for yv, v in zip([2, 1, 0], vals):
    ax.text(v + 0.01, yv, f"{v:.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels(names, fontsize=8.5, color=INK)
ax.set_xlim(0, 1.05)
ax.set_xlabel("$R^2$ for log$_{10}$ G on the 1,648 test crystals", color=MUTED, fontsize=9)
ax.set_title("C   does the K model secretly know G?", loc="left", fontsize=10, color=INK)
# transform=ax.transAxes: x, y measured as fractions of the panel (0..1),
# so y = -0.27 is just under the axis label whatever the data range is.
ax.text(0.0, -0.27, f"the 127 extra numbers add only {r2G - r2_base:+.3f}: G is mostly 'K in disguise'",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
style_axes(ax)
ax.grid(axis="y", visible=False)

fig.tight_layout()
os.makedirs(FIGS, exist_ok=True)
png = os.path.join(FIGS, "hidden_features.png")
fig.savefig(png, dpi=200, bbox_inches="tight")
csv = os.path.join(FIGS, "hidden_features_test.csv")
df.to_csv(csv, index=False)
print(f"wrote {png}\nwrote {csv}")
