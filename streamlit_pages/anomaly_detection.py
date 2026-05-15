# streamlit_pages/anomaly_detection.py
#
# PURPOSE:
# Page 4 of the dashboard.
# Shows Autoencoder anomaly detection results.
#
# DISPLAYS:
# - Threshold, flagged count, agreement with ML model
# - Reconstruction error distribution with threshold line
# - Row by row comparison: ML model vs Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from streamlit_utils.preprocessor import get_predictions


def render(X_scaled: np.ndarray, model, autoencoder):
    """
    Renders Page 4 — Anomaly Detection.

    Args:
        X_scaled    : scaled feature array
        model       : trained sklearn model (for comparison)
        autoencoder : trained AnomalyDetector
    """
    st.header("🤖 Autoencoder Anomaly Detection")
    st.markdown(
        "The Autoencoder was trained on **benign traffic only**. "
        "It flags anything it cannot reconstruct well as an anomaly. "
        "This is a second opinion alongside the ML model."
    )

    # ── Get results ──
    predictions, _ = get_predictions(model, X_scaled)
    anomaly_scores = autoencoder.get_reconstruction_errors(X_scaled)
    anomaly_preds = autoencoder.predict(X_scaled)

    # ── Metrics ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Anomaly Threshold", f"{autoencoder.threshold:.4f}")
    col2.metric("Flagged as Anomaly", int(anomaly_preds.sum()))
    col3.metric(
        "Agreement with ML Model",
        f"{(anomaly_preds == predictions).mean() * 100:.1f}%",
        help="How often Autoencoder and sklearn model agree on same row"
    )

    st.markdown("---")

    # ── Reconstruction Error Distribution ──
    # Shows where benign and attack errors cluster
    # Red line = threshold boundary
    st.subheader("Reconstruction Error Distribution")
    st.markdown(
        "Samples to the **right of the red line** are flagged as anomalies. "
        "Normal traffic clusters near 0."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(
        anomaly_scores,
        bins=50,
        color='steelblue',
        alpha=0.7,
        label='Reconstruction Error'
    )
    ax.axvline(
        autoencoder.threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold = {autoencoder.threshold:.4f}'
    )
    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Count')
    ax.set_title('Autoencoder Reconstruction Error Distribution')
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Comparison Table ──
    # Side by side: what ML model predicted vs what Autoencoder flagged
    st.subheader("ML Model vs Autoencoder — Row by Row")
    comparison_df = pd.DataFrame({
        'ML Model': ['🔴 ATTACK' if p == 1 else '🟢 BENIGN' for p in predictions],
        'Autoencoder': ['⚠️ Anomaly' if p == 1 else '✅ Normal' for p in anomaly_preds],
        'Anomaly Score': [f"{s:.4f}" for s in anomaly_scores],
        'Agreement': ['✅' if p == a else '❌' for p, a in zip(predictions, anomaly_preds)]
    })
    st.dataframe(comparison_df, use_container_width=True, height=400)