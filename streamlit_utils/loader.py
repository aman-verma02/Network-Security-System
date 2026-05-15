# streamlit_utils/loader.py
#
# PURPOSE:
# Loads all trained model artifacts from final_model/ directory.
# Kept separate so any page can import and use it without
# duplicating the loading logic.
#
# WHY @st.cache_resource:
# Without caching, every user interaction (button click, slider move)
# would reload all models from disk — making the app very slow.
# @st.cache_resource runs load_models() ONCE and keeps it in memory.

import pickle
import streamlit as st


@st.cache_resource
def load_models():
    """
    Loads all 4 artifacts from final_model/ directory.

    Returns:
        model         : trained sklearn/XGBoost model (model.pkl)
        preprocessor  : fitted StandardScaler (preprocessor.pkl)
        autoencoder   : trained AnomalyDetector with threshold (autoencoder.pkl)
        feature_names : list of 40 selected feature names (feature_names.pkl)

    Raises:
        Streamlit error and stops app if any file is missing.
    """
    try:
        with open("final_model/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("final_model/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open("final_model/autoencoder.pkl", "rb") as f:
            autoencoder = pickle.load(f)

        with open("final_model/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        return model, preprocessor, autoencoder, feature_names

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()