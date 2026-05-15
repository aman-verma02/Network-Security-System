# streamlit_pages/shap_explanation.py
#
# PURPOSE:
# Page 2 of the dashboard.
# Explains WHY the model made each prediction using SHAP.
#
# DISPLAYS:
# - Row selector slider
# - Waterfall chart for selected row
# - Top 10 features table for selected row
# - Summary plot across all rows

import numpy as np
import streamlit as st

from networksecurity.utils.ml_utils.explainability.shap_explainer import SHAPExplainer


def render(X_scaled: np.ndarray, model, feature_names: list):
    """
    Renders Page 2 — SHAP Explanation.

    Args:
        X_scaled      : scaled feature array
        model         : trained sklearn/XGBoost model
        feature_names : list of 40 feature names
    """
    st.header("🧠 SHAP Explainability")
    st.markdown(
        "SHAP shows **which features drove each prediction**. "
        "Positive values pushed toward Attack, "
        "negative values pushed toward Benign."
    )

    # Initialize SHAP explainer
    # TreeExplainer reads the actual tree structure of RF/XGBoost
    with st.spinner("Initializing SHAP explainer..."):
        explainer = SHAPExplainer(model=model, feature_names=feature_names)

    # ── Row Selector ──
    # User picks which row to explain
    sample_index = st.slider(
        "Select row to explain",
        min_value=0,
        max_value=len(X_scaled) - 1,
        value=0,
        help="Choose which row from your uploaded CSV to explain"
    )

    col1, col2 = st.columns(2)

    # ── Waterfall Chart ──
    # Shows step by step how features pushed the prediction
    with col1:
        st.subheader(f"Waterfall Chart — Row {sample_index}")
        st.markdown(
            "Each bar shows how much one feature "
            "pushed the prediction up (Attack) or down (Benign)."
        )
        save_path = f"plots/shap_waterfall_{sample_index}.png"
        explainer.plot_waterfall(
            X_scaled,
            sample_index=sample_index,
            save_path=save_path
        )
        st.image(save_path, use_column_width=True)

    # ── Top Features Table ──
    with col2:
        st.subheader(f"Top 10 Features — Row {sample_index}")
        st.markdown("Features sorted by their impact on this prediction.")
        top_df = explainer.get_top_features_for_prediction(
            X_scaled,
            sample_index=sample_index,
            top_n=10
        )
        st.dataframe(top_df, use_container_width=True)

    st.markdown("---")

    # ── Summary Plot ──
    # Shows overall feature importance across all rows
    st.subheader("SHAP Summary Plot — All Rows")
    st.markdown(
        "Each dot = one sample. "
        "X axis = SHAP value (impact). "
        "Color = feature value (red=high, blue=low)."
    )
    with st.spinner("Generating summary plot..."):
        explainer.plot_summary(X_scaled, save_path="plots/shap_summary.png")
    st.image("plots/shap_summary.png", use_column_width=True)