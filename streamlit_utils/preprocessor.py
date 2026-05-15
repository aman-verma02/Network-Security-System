# streamlit_utils/preprocessor.py
#
# PURPOSE:
# Prepares user uploaded CSV for prediction.
# Kept separate so prediction logic stays clean
# and preprocessing can be reused across pages.
#
# FLOW:
# Raw uploaded CSV
#     → drop Label/target if present
#     → keep only the 40 selected features
#     → replace inf/nan values
#     → scale with StandardScaler
#     → return numpy array ready for model

import sys
import numpy as np
import pandas as pd
import streamlit as st

from networksecurity.exception.exception import NetworkSecurityException


def preprocess_input(
    df: pd.DataFrame,
    feature_names: list,
    preprocessor
) -> np.ndarray:
    """
    Prepares uploaded DataFrame for model prediction.

    Steps:
    1. Drop target/Label column if user included it
    2. Check all required features are present
    3. Keep only the 40 features model was trained on
    4. Replace inf and NaN with 0
    5. Apply StandardScaler (same one fitted during training)

    Args:
        df            : raw uploaded DataFrame from CSV
        feature_names : list of 40 feature names from feature_names.pkl
        preprocessor  : fitted StandardScaler from preprocessor.pkl

    Returns:
        np.ndarray: scaled feature matrix, shape (n_rows, 40)
    """
    try:
        # Step 1: Drop target columns if present
        # User might upload the full dataset including labels
        for col in ['target', 'Label']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Step 2: Check required features exist
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            st.error(
                f"Uploaded CSV is missing {len(missing)} required columns. "
                f"First few missing: {missing[:3]}"
            )
            st.stop()

        # Step 3: Keep only selected features in correct order
        df = df[feature_names]

        # Step 4: Handle inf and NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Step 5: Scale using the same scaler fitted during training
        X_scaled = preprocessor.transform(df)
        return X_scaled

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def get_predictions(model, X_scaled: np.ndarray) -> tuple:
    """
    Runs sklearn model on scaled input and returns predictions
    with confidence scores.

    Args:
        model    : trained sklearn/XGBoost model
        X_scaled : scaled feature array

    Returns:
        predictions : np.ndarray of 0/1 per row (0=Benign, 1=Attack)
        confidence  : np.ndarray of confidence % for predicted class
    """
    predictions = model.predict(X_scaled)

    # predict_proba returns [[prob_benign, prob_attack], ...]
    # np.max picks the higher probability (confidence in prediction)
    probabilities = model.predict_proba(X_scaled)
    confidence = np.max(probabilities, axis=1) * 100

    return predictions, confidence