# streamlit_pages/predictions.py
#
# PURPOSE:
# Page 1 of the dashboard.
# Shows prediction results for every row in uploaded CSV.
#
# DISPLAYS:
# - Summary metrics (total, attacks, benign, avg confidence)
# - Results table with prediction + confidence + anomaly score
# - Download button for results CSV

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_utils.preprocessor import get_predictions


def render(df: pd.DataFrame, X_scaled: np.ndarray, model, autoencoder):
    """
    Renders Page 1 — Predictions.

    Args:
        df          : original uploaded DataFrame (for display)
        X_scaled    : scaled feature array (for prediction)
        model       : trained sklearn/XGBoost model
        autoencoder : trained AnomalyDetector
    """
    st.header("🔍 Intrusion Detection Predictions")

    # ── Get predictions ──
    # predictions  : 0 or 1 per row
    # confidence   : how confident the model is (%)
    predictions, confidence = get_predictions(model, X_scaled)

    # Anomaly score = reconstruction error from autoencoder
    # Higher score = more different from normal traffic
    anomaly_scores = autoencoder.get_reconstruction_errors(X_scaled)

    # ── Summary Metrics ──
    # st.columns(4) splits page into 4 equal columns
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Samples", len(predictions))
    col2.metric("⚠️ Attacks Detected", int(predictions.sum()))
    col3.metric("✅ Benign Traffic", int((predictions == 0).sum()))
    col4.metric("Avg Confidence", f"{confidence.mean():.1f}%")

    st.markdown("---")

    # ── Results Table ──
    results_df = pd.DataFrame({
        'Prediction': ['🔴 ATTACK' if p == 1 else '🟢 BENIGN' for p in predictions],
        'Confidence': [f"{c:.1f}%" for c in confidence],
        'Anomaly Score': [f"{s:.4f}" for s in anomaly_scores],
        'Anomaly Flag': [
            '⚠️ Anomaly' if s > autoencoder.threshold else '✅ Normal'
            for s in anomaly_scores
        ]
    })

    st.subheader("Prediction Results")
    st.dataframe(results_df, use_container_width=True, height=400)

    # ── Download Button ──
    # Combines original data with predictions for download
    full_results = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    csv = full_results.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )