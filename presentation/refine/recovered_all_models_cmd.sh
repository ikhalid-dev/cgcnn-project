cd "/Users/mac/Desktop/Cgcnn project/presentation" && cat >> make_pink_figures.py <<'PYEOF'


# =============================================================================
#  THREE-WAY versions of figures that were CGCNN-only
# =============================================================================
#  Several screen figures in this deck were produced from the CGCNN ensemble
#  alone and captioned as if they described "the screen". They describe ONE
#  MODEL'S screen. These regenerate them with every model that scored the same
#  33,118 candidates, so a reader can see where the choice of model changes the
#  answer - which, for the low-kappa call, it does a great deal.
# =============================================================================
MODEL_COLS = [("CGCNN ens", "Kappa_cal_derived_matbench", BLUE),
              ("ALIGNN",    "Kappa_alignn",               GREEN),
              ("tree (XGB/RF)", "Kappa_baseline",         AMBER),
              ("round 9",   "Kappa_r9_gamma",             MUTED)]


def _load_screen():
    """The 33k screen with every model's kappa on one row."""
    a = pd.read_csv(os.path.join(ROOT, "results/alignn/57_gnome_screen_alignn.csv"),
                    dtype={"material_id": str})
    s = pd.read_csv(os.path.join(ROOT, "results/cgcnn/39_gnome_screen_all_gamma.csv"),
                    dtype={"material_id": str})
    d = a.merge(s, on="material_id", suffixes=("", "_s"))
    d = d[d["alignn_prediction_reliable"]].copy()
    for _, col, _c in MODEL_COLS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return d


def fig_kappa_distributions_all(path):
    d = _load_screen()
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    for lbl, col, c in MODEL_COLS:
        v = d[col].dropna()
        v = v[v > 0]
        ax.hist(np.log10(v), bins=90, histtype="step", lw=1.8, color=c,
                label=f"{lbl}   {(v <= 1).sum():,} below 1")
    ax.axvline(0, color=AMBER, lw=1.2, ls=":")
    ax.set_xlabel("$\\log_{10}\\ \\kappa_L$  (W m$^{-1}$K$^{-1}$)")
    ax.set_ylabel("candidates")
    ax.set_title("The same 33,053 GNoME candidates, scored by four models\n"
                 "dotted line = the $\\kappa_L \\leq 1$ threshold the screen calls on",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    style(ax); ax.grid(color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def fig_elements_all(path):
    d = _load_screen()
    base = {lbl: (d[col] <= 1).mean() for lbl, col, _ in MODEL_COLS}
    freq, low = collections.Counter(), {lbl: collections.Counter() for lbl, _, _ in MODEL_COLS}
    for i, f in enumerate(d["formula"]):
        try:
            els = [e.symbol for e in Composition(str(f)).elements]
        except Exception:
            continue
        for s_ in els:
            freq[s_] += 1
            for lbl, col, _ in MODEL_COLS:
                if d[col].iloc[i] <= 1:
                    low[lbl][s_] += 1

    keep = [s_ for s_, c in freq.items() if c >= 400]
    rows = []
    for s_ in keep:
        try:
            x = Element(s_).X
        except Exception:
            continue
        if not np.isfinite(x):
            continue
        r = {"element": s_, "n_candidates": freq[s_], "electronegativity": x}
        for lbl, _, _ in MODEL_COLS:
            r[f"enrich_{lbl}"] = (low[lbl][s_] / freq[s_]) / base[lbl]
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("electronegativity")

    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    for lbl, _, c in MODEL_COLS:
        ax.plot(t["electronegativity"], t[f"enrich_{lbl}"], "o-", color=c,
                lw=1.3, ms=5, alpha=0.85, label=lbl)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=1)
    for _, r in t.iterrows():
        if r["enrich_CGCNN ens"] > 1.3 or r["enrich_CGCNN ens"] < 0.55:
            ax.annotate(r["element"], (r["electronegativity"], r["enrich_CGCNN ens"]),
                        fontsize=8, color=INK, xytext=(0, 8),
                        textcoords="offset points", ha="center")
    ax.set_xlabel("Pauling electronegativity")
    ax.set_ylabel("low-$\\kappa$ enrichment (1.0 = no preference)")
    ax.set_title("Element enrichment, all four models\n"
                 "every model agrees on the trend; they disagree on its size",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False)
    style(ax); ax.grid(color=GRID, lw=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)
    t.to_csv(path.replace(".png", ".csv"), index=False)
    print("wrote", os.path.relpath(path, ROOT), "+ .csv")


def fig_families_all(path):
    d = _load_screen()
    fam = d["stoichiometry_pattern"].fillna("non-oxide")
    order = fam.value_counts().index.tolist()
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    w = 0.2
    for i, (lbl, col, c) in enumerate(MODEL_COLS):
        rates, ns = [], []
        for f in order:
            m = fam == f
            ns.append(int(m.sum()))
            rates.append(100 * (d.loc[m, col] <= 1).mean())
        xs = [k + (i - 1.5) * w for k in range(len(order))]
        ax.bar(xs, rates, width=w, color=c, label=lbl, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{f}\n(n={int((fam==f).sum()):,})" for f in order],
                       fontsize=7.5, rotation=18, ha="right")
    ax.set_ylabel("% of family called low-$\\kappa$")
    ax.set_title("Hit rate by stoichiometry family, all four models\n"
                 "the ordering is stable; the absolute rate is not",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, ncol=4)
    style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=180, facecolor="white"); plt.close(fig)

    out = []
    for f in order:
        m = fam == f
        row = {"family": f, "n_candidates": int(m.sum())}
        for lbl, col, _ in MODEL_COLS:
            row[f"low_kappa_{lbl}"] = int((d.loc[m, col] <= 1).sum())
            row[f"pct_{lbl}"] = round(100 * (d.loc[m, col] <= 1).mean(), 2)
        out.append(row)
    pd.DataFrame(out).to_csv(path.replace(".png", ".csv"), index=False)
    print("wrote", os.path.relpath(path, ROOT), "+ .csv")
PYEOF
/Users/mac/miniconda3/envs/ml_env/bin/python -c "
import make_pink_figures as m, os
m.fig_kappa_distributions_all(os.path.join(m.FIGS,'kappa_dist_all_models.png'))
m.fig_elements_all(os.path.join(m.FIGS,'elements_all_models.png'))
m.fig_families_all(os.path.join(m.FIGS,'families_all_models.png'))
"