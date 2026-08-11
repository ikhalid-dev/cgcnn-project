#!/usr/bin/env python3
"""
Local web app over predict_engine.py - drag in CIF files, get K/G/kappa_L.
================================================================================

    streamlit run predict_app.py

The browser-based twin of predict.py: same predict_engine.py underneath (see
its module docstring for what's reused versus new), just a drag-and-drop
upload box and a results table instead of a terminal. Model loading is
wrapped in st.cache_resource so the 5 checkpoints load once when the server
starts, not once per upload - without that, every prediction would pay the
full weight-loading cost again.
"""

# torch first - see predict_engine.py's own note. Streamlit itself pulls in
# numpy/pandas as part of its own startup, so this has to come before even
# `import streamlit`, not just before this file's own numpy/pandas imports.
import torch  # noqa: F401

import os
import shutil
import tempfile

import pandas as pd
import streamlit as st

import predict_engine as engine

st.set_page_config(page_title="PINK predictor", page_icon=None, layout="wide")

CONF_BADGE = {
    "high": "\U0001F7E2 HIGH",
    "medium": "\U0001F7E1 MEDIUM",
    "low": "\U0001F534 LOW",
    "unverified": "⚪ UNVERIFIED",
}


@st.cache_resource(show_spinner="Loading CGCNN ensemble + ALIGNN models (once per server start)...")
def get_models():
    return engine.load_models()


def save_uploads(uploaded_files):
    """Streamlit's uploader gives in-memory file objects; predict_batch()
    (like every script it wraps) expects a real directory of .cif files on
    disk, via glob - write them out to a throwaway temp directory instead of
    teaching the engine a second, upload-specific input path.
    """
    tmp_dir = tempfile.mkdtemp(prefix="predict_app_")
    for uploaded in uploaded_files:
        with open(os.path.join(tmp_dir, uploaded.name), "wb") as fh:
            fh.write(uploaded.getvalue())
    return tmp_dir


def render_results(df):
    display = df[[
        "material_id", "formula", "provenance",
        "K_VRH_pred", "K_VRH_alignn", "G_VRH_pred", "G_VRH_alignn",
        "Kappa_cal_cgcnn", "Kappa_cal_alignn", "confidence", "model_disagreement_pct",
    ]].rename(columns={
        "material_id": "Material", "formula": "Formula", "provenance": "Provenance",
        "K_VRH_pred": "K, CGCNN (GPa)", "K_VRH_alignn": "K, ALIGNN (GPa)",
        "G_VRH_pred": "G, CGCNN (GPa)", "G_VRH_alignn": "G, ALIGNN (GPa)",
        "Kappa_cal_cgcnn": "kappa_L, CGCNN (W/m/K)", "Kappa_cal_alignn": "kappa_L, ALIGNN (W/m/K)",
        "confidence": "Confidence", "model_disagreement_pct": "Disagreement (%)",
    }).round(3)
    display["Confidence"] = display["Confidence"].map(CONF_BADGE)

    st.dataframe(display, use_container_width=True, hide_index=True)

    csv_bytes = df.round(4).to_csv(index=False).encode("utf-8")
    st.download_button("Download full CSV", data=csv_bytes,
                       file_name="predictions.csv", mime="text/csv")

    with st.expander("What does Confidence mean?"):
        st.markdown(
            "A heuristic, not a calibrated statistic - see `predict_engine.confidence_tier`'s "
            "docstring for the exact rule. In short: **HIGH** means CGCNN and ALIGNN agree on "
            "kappa_L within about 20%, which is what this pipeline looks like when it's working "
            "correctly (its own measured accuracy floor is about 15%, from DFT label noise - "
            "see `PREDICT_NEW_CIFS.txt`). **MEDIUM**/**LOW** mean wider disagreement or an "
            "implausible intermediate value. **UNVERIFIED** means one of the two models "
            "couldn't produce a number for that crystal at all, so there's nothing to "
            "cross-check. If Provenance isn't \"unseen\", a real DFT reference already exists "
            "for that crystal - open the downloaded CSV for the `K_VRH_dft`/`G_VRH_dft` columns.")


def main():
    st.title("PINK predictor")
    st.caption("K, G and lattice thermal conductivity for your own crystals - "
              "CGCNN ensemble + ALIGNN, cross-checked against each other. "
              "See PREDICT_NEW_CIFS.txt for the full explanation this app automates.")

    uploaded_files = st.file_uploader(
        "Drop .cif file(s) here", type="cif", accept_multiple_files=True)

    if not uploaded_files:
        st.info("Upload one or more .cif files to get predictions.")
        return

    models = get_models()

    tmp_dir = save_uploads(uploaded_files)
    try:
        with st.spinner(f"Predicting {len(uploaded_files)} crystal(s)..."):
            df, failures = engine.predict_batch(tmp_dir, models)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    for label, failed in failures.items():
        if failed:
            st.warning(f"{len(failed)} crystal(s) failed {label.upper()} "
                      f"featurisation: {[f[0] for f in failed]}")

    if df.empty:
        st.error("No crystal produced a prediction.")
        return

    render_results(df)


if __name__ == "__main__":
    main()
