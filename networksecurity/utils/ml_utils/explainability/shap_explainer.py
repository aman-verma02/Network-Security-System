# networksecurity/utils/ml_utils/explainability/shap_explainer.py
#
# PURPOSE:
# Explains WHY the model made a prediction using SHAP values.
# For every prediction, SHAP assigns each feature a score showing
# how much it pushed the prediction toward attack or benign.
#
# HOW SHAP WORKS INTERNALLY:
# SHAP uses TreeExplainer for tree-based models (RF, XGBoost).
# It looks at every decision tree in the model and traces which
# features caused the prediction to go up or down.
#
# HOW IT FITS IN YOUR PROJECT:
# called from Streamlit dashboard after model.predict()
# Input  → scaled feature array (output of preprocessor.pkl)
# Output → SHAP values array + matplotlib plots saved to plots/

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer with methods to:
    1. Compute SHAP values for any input sample
    2. Generate a bar chart — overall feature importance
    3. Generate a waterfall chart — single prediction explanation
    4. Generate a summary plot — across all predictions

    Works with any tree-based sklearn or XGBoost model.
    (Random Forest, XGBoost, GradientBoosting, DecisionTree)

    Args:
        model         : trained sklearn/XGBoost model loaded from model.pkl
        feature_names : list of feature names loaded from feature_names.pkl
    """

    def __init__(self, model, feature_names: list):
        try:
            self.model = model
            self.feature_names = feature_names

            # TreeExplainer is optimized for tree-based models
            # It is much faster than the generic KernelExplainer
            # It reads the actual tree structure to compute exact SHAP values
            logging.info("Initializing SHAP TreeExplainer")
            self.explainer = shap.TreeExplainer(model)
            logging.info("SHAP TreeExplainer initialized successfully")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Computes SHAP values for given input samples.

        SHAP values tell you: for each feature in each sample,
        how much did that feature push the prediction toward attack (positive)
        or toward benign (negative)?

        Args:
            X (np.ndarray): scaled input features
                            shape = (n_samples, n_features)

        Returns:
            np.ndarray: SHAP values
                        shape = (n_samples, n_features)
                        positive value = pushed toward attack
                        negative value = pushed toward benign
        """
        try:
            logging.info(f"Computing SHAP values for {X.shape[0]} samples")

            shap_values = self.explainer.shap_values(X)

            # For binary classification, shap_values returns a list of 2 arrays
            # Index 0 = SHAP values for class 0 (benign)
            # Index 1 = SHAP values for class 1 (attack)
            # We always want class 1 (attack) explanation
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            logging.info("SHAP values computed successfully")
            return shap_values

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: str = "plots/shap_summary.png"
    ):
        """
        Generates a SHAP summary plot across ALL input samples.

        Each dot = one sample
        X axis  = SHAP value (impact on prediction)
        Color   = feature value (red=high, blue=low)

        This plot shows:
        - Which features matter most overall (top = most important)
        - How feature values relate to their impact

        Use case: show this in README and Streamlit dashboard
                  to explain overall model behavior.

        Args:
            X         : scaled input features for all samples
            save_path : where to save the plot image
        """
        try:
            logging.info("Generating SHAP summary plot")
            shap_values = self.get_shap_values(X)

            plt.figure()
            shap.summary_plot(
                shap_values,
                X,
                feature_names=self.feature_names,
                show=False        # don't display — we save to file instead
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"SHAP summary plot saved to {save_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def plot_waterfall(
        self,
        X: np.ndarray,
        sample_index: int = 0,
        save_path: str = "plots/shap_waterfall.png"
    ):
        """
        Generates a SHAP waterfall plot for a SINGLE prediction.

        Shows exactly how each feature contributed to one specific prediction:
        - Starts from baseline (average model output)
        - Each bar shows how much one feature pushed the prediction up or down
        - Ends at the final prediction value

        Use case: user uploads one row of network traffic,
                  Streamlit shows this chart to explain the prediction.

        Args:
            X            : scaled input features
            sample_index : which sample to explain (default: first row)
            save_path    : where to save the plot image
        """
        try:
            logging.info(f"Generating SHAP waterfall plot for sample {sample_index}")
            shap_values = self.get_shap_values(X)

            # Get SHAP explanation object for one sample
            explanation = shap.Explanation(
                values=shap_values[sample_index],           # SHAP values for this sample
                base_values=self.explainer.expected_value,  # baseline prediction
                data=X[sample_index],                       # actual feature values
                feature_names=self.feature_names            # feature names for labels
            )

            # Handle case where expected_value is a list (binary classification)
            if isinstance(explanation.base_values, np.ndarray):
                explanation.base_values = explanation.base_values[1]

            plt.figure()
            shap.plots.waterfall(explanation, show=False)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"SHAP waterfall plot saved to {save_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def plot_bar(
        self,
        X: np.ndarray,
        save_path: str = "plots/shap_bar.png"
    ):
        """
        Generates a SHAP bar chart showing mean absolute SHAP values.

        Simple horizontal bar chart:
        - Each bar = one feature
        - Bar length = average impact across all samples
        - Sorted by importance (longest bar at top)

        Use case: easiest chart to understand — good for README
                  and for Streamlit dashboard overview tab.

        Args:
            X         : scaled input features
            save_path : where to save the plot image
        """
        try:
            logging.info("Generating SHAP bar plot")
            shap_values = self.get_shap_values(X)

            # Mean absolute SHAP value per feature
            # abs() because direction doesn't matter for importance ranking
            mean_shap = np.abs(shap_values).mean(axis=0)

            # Build a sorted dataframe for clean plotting
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=True)

            # Plot horizontal bar chart
            plt.figure(figsize=(10, 8))
            plt.barh(
                importance_df['feature'],
                importance_df['importance'],
                color='steelblue'
            )
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            logging.info(f"SHAP bar plot saved to {save_path}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_top_features_for_prediction(
        self,
        X: np.ndarray,
        sample_index: int = 0,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Returns top N most influential features for a single prediction
        as a clean DataFrame.

        Used by Streamlit to show a table alongside the waterfall chart:

        | Feature          | SHAP Value | Direction      |
        |------------------|------------|----------------|
        | Flow Bytes/s     | 0.32       | → Attack       |
        | SYN Flag Count   | 0.28       | → Attack       |
        | ACK Flag Count   | -0.08      | → Benign       |

        Args:
            X            : scaled input features
            sample_index : which sample to explain
            top_n        : how many top features to return

        Returns:
            pd.DataFrame with columns: Feature, SHAP Value, Direction
        """
        try:
            shap_values = self.get_shap_values(X)
            sample_shap = shap_values[sample_index]

            # Build dataframe of feature name + shap value
            df = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP Value': sample_shap
            })

            # Sort by absolute value — biggest impact first
            df['Abs'] = df['SHAP Value'].abs()
            df = df.sort_values('Abs', ascending=False).head(top_n)
            df = df.drop(columns=['Abs'])

            # Add direction column — makes it readable in Streamlit table
            df['Direction'] = df['SHAP Value'].apply(
                lambda x: '→ Attack' if x > 0 else '→ Benign'
            )

            return df.reset_index(drop=True)

        except Exception as e:
            raise NetworkSecurityException(e, sys)