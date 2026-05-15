# streamlit_app.py
#
# PURPOSE:
# Entry point of the Streamlit app. Kept minimal intentionally.
# All logic lives in streamlit_pages/ and streamlit_utils/.
#
# HOW TO RUN:
#   streamlit run streamlit_app.py
#
# FLOW:
# 1. Load models (cached)
# 2. Render sidebar + file uploader
# 3. If file uploaded → preprocess → show 4 tabs
# 4. If no file → show landing page

import pandas as pd
import streamlit as st

from streamlit_utils.loader import load_models
from streamlit_utils.preprocessor import preprocess_input
from streamlit_pages import predictions, shap_explanation, performance, anomaly_detection

st.set_page_config(
    page_title="Network Security System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def render_sidebar():
    """
    Renders sidebar with project info and file uploader.
    Returns uploaded file object or None.
    """
    st.sidebar.title("🛡️ Network Security System")
    st.sidebar.markdown(
        "Upload network traffic CSV to detect intrusions, "
        "explain predictions, and analyze model performance."
    )
    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"],
        help="Upload CICIDS2017 format CSV"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built with:**")
    st.sidebar.markdown("- Scikit-learn + XGBoost")
    st.sidebar.markdown("- PyTorch Autoencoder")
    st.sidebar.markdown("- SHAP Explainability")
    st.sidebar.markdown("- MLflow + DagsHub")

    return uploaded_file


def render_landing_page():
    """
    Shown when no file is uploaded.
    Explains what the app does and how to use it.
    """
    st.title("🛡️ Network Security Intrusion Detection System")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.info("**🔍 Predictions**\n\nDetect attacks with confidence scores.")
    col2.info("**🧠 SHAP Explanations**\n\nUnderstand WHY each prediction was made.")
    col3.info("**🤖 Anomaly Detection**\n\nPyTorch Autoencoder flags unusual traffic.")

    st.markdown("---")
    st.subheader("How to Use")
    st.markdown(
        "1. Upload a CICIDS2017 format CSV using the sidebar\n"
        "2. Explore predictions, explanations, and anomaly scores\n"
        "3. Download results as CSV"
    )

    st.subheader("Tech Stack")
    st.markdown(
        "- **ML Models**: Random Forest, XGBoost, Gradient Boosting\n"
        "- **Anomaly Detection**: PyTorch Autoencoder\n"
        "- **Explainability**: SHAP TreeExplainer\n"
        "- **Experiment Tracking**: MLflow + DagsHub\n"
        "- **Deployment**: Streamlit Cloud\n"
        "- **CI/CD**: GitHub Actions + Docker"
    )


def main():
    """
    Main entry point.
    Loads models, renders sidebar, routes to pages.
    """
    model, preprocessor, autoencoder, feature_names = load_models()
    uploaded_file = render_sidebar()

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")

        # Preprocess: select features + scale
        X_scaled = preprocess_input(df, feature_names, preprocessor)

        # 4 tabs — one per page
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Predictions",
            "🧠 SHAP Explanation",
            "📊 Model Performance",
            "🤖 Anomaly Detection"
        ])

        with tab1:
            predictions.render(df, X_scaled, model, autoencoder)

        with tab2:
            shap_explanation.render(X_scaled, model, feature_names)

        with tab3:
            performance.render()

        with tab4:
            anomaly_detection.render(X_scaled, model, autoencoder)

    else:
        render_landing_page()


if __name__ == "__main__":
    main()