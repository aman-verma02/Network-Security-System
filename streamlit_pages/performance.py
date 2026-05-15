# streamlit_pages/performance.py
#
# PURPOSE:
# Page 3 of the dashboard.
# Shows model performance metrics and plots
# generated during training in the Colab notebook.
#
# DISPLAYS:
# - Accuracy, F1, ROC-AUC metrics
# - Confusion matrix plot
# - Model comparison chart
# - Feature importance chart

import os
import streamlit as st


def render():
    """
    Renders Page 3 — Model Performance.
    No arguments needed — loads plots from plots/ directory.

    Update the metric numbers below with your actual
    results from the Colab notebook after training.
    """
    st.header("📊 Model Performance")
    st.markdown(
        "Metrics from training on CICIDS2017 dataset "
        "with SMOTE balancing and grid search tuning."
    )

    # ── Metrics ──
    # UPDATE THESE with your actual numbers from Colab output
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "98.7%")
    col2.metric("F1 Score", "98.4%")
    col3.metric("ROC-AUC", "99.2%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Confusion Matrix ──
    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists("plots/confusion_matrix.png"):
            st.image("plots/confusion_matrix.png", use_column_width=True)
        else:
            st.info("Run training pipeline to generate confusion matrix.")

    # ── Model Comparison ──
    with col2:
        st.subheader("Model Comparison")
        if os.path.exists("plots/model_comparison.png"):
            st.image("plots/model_comparison.png", use_column_width=True)
        else:
            st.info("Run training pipeline to generate model comparison.")

    st.markdown("---")

    # ── Feature Importance ──
    st.subheader("Feature Importance")
    if os.path.exists("plots/feature_importance.png"):
        st.image("plots/feature_importance.png", use_column_width=True)
    else:
        st.info("Run training pipeline to generate feature importance plot.")