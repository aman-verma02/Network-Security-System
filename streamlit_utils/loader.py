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
import torch
import streamlit as st
from networksecurity.components.anomaly_detector import AutoEncoder, AnomalyDetector


@st.cache_resource
def load_models():
    try:
        with open("final_model/model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("final_model/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)

        with open("final_model/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # Load meta dict (threshold + config) — pure Python, no CUDA
        with open("final_model/autoencoder_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        # Rebuild AnomalyDetector from scratch on CPU
        autoencoder = AnomalyDetector(
            input_dim=meta['input_dim'],
            epochs=meta['epochs'],
            batch_size=meta['batch_size'],
            lr=meta['lr'],
            threshold=meta['threshold']
        )

        # Load weights onto CPU
        autoencoder.model.load_state_dict(
            torch.load(
                'final_model/autoencoder_weights.pth',
                map_location=torch.device('cpu')
            )
        )
        autoencoder.model.eval()

        return model, preprocessor, autoencoder, feature_names

    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()