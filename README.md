# 🛡️ Network Security Intrusion Detection System

An end-to-end Machine Learning system that detects network intrusions in real-time using the CICIDS2017 dataset. Built with a production-grade modular pipeline, PyTorch Autoencoder for anomaly detection, SHAP explainability, and deployed as an interactive Streamlit dashboard.

---

## 🔴 Live Demo
👉 [Click here to open the live app](https://network-security-system.streamlit.app/)

---

## 📊 Results

| Model | Accuracy | F1 Score | ROC-AUC | Train Time |
|---|---|---|---|---|
| **XGBoost** ✅ | **99.93%** | **99.91%** | **100%** | 15s |
| Random Forest | 99.89% | 99.86% | 99.99% | 212s |
| Decision Tree | 99.89% | 99.86% | 99.90% | 27s |
| Logistic Regression | 96.94% | 96.26% | 99.52% | 6s |

**Best Model: XGBoost** — selected automatically by pipeline based on F1 Score.

### Autoencoder Anomaly Detection
| Metric | Value |
|---|---|
| Detection Rate | 23.15% |
| False Alarm Rate | 1.49% |
| Threshold | 0.8841 |

> Note: The Autoencoder complements the XGBoost classifier as an unsupervised
> second opinion. Lower detection rate is expected since many CICIDS2017 attack 
> patterns overlap with benign traffic in feature space — a known challenge 
> in unsupervised network intrusion detection.

---

## 🏗️ Architecture

```
Raw CICIDS2017 Data (958k rows, 100 features)
        │
        ▼
MongoDB Atlas (Data Store)
        │
        ▼
DataIngestion → DataValidation → DataTransformation → ModelTrainer → AnomalyDetector
                                        │                   │               │
                                 Feature Selection      XGBoost        PyTorch
                                 + StandardScaler      + SMOTE       Autoencoder
                                        │                   │               │
                                        └───────────────────┴───────────────┘
                                                            │
                                                     Streamlit Dashboard
                                                            │
                                          ┌─────────────────┼─────────────────┐
                                          │                 │                 │
                                   Predictions         SHAP Charts    Anomaly Scores
```

---

## ✨ Features

- **Multi-model Training** — XGBoost, Random Forest, Decision Tree, Logistic Regression trained with grid search, best model selected automatically
- **Class Imbalance Handling** — SMOTE oversampling on training data
- **Feature Engineering** — Correlation drop, variance threshold, RF importance selection (100 → 40 features)
- **PyTorch Autoencoder** — Unsupervised anomaly detection trained on benign traffic only
- **SHAP Explainability** — Per-prediction feature contribution charts (waterfall + summary)
- **Experiment Tracking** — MLflow + DagsHub for full experiment reproducibility
- **Modular Pipeline** — Loosely coupled components with artifact passing
- **CI/CD** — GitHub Actions + Docker
- **Interactive Dashboard** — Streamlit app with 4 pages

---

## 📁 Project Structure

```
Network-Security-System/
├── networksecurity/
│   ├── components/
│   │   ├── data_ingestion.py        # pulls data from MongoDB
│   │   ├── data_validation.py       # schema validation
│   │   ├── data_transformation.py   # feature selection + scaling
│   │   ├── model_trainer.py         # trains models + MLflow tracking
│   │   └── anomaly_detector.py      # PyTorch Autoencoder
│   ├── pipeline/
│   │   └── training_pipeline.py     # orchestrates all 5 stages
│   ├── utils/
│   │   └── ml_utils/
│   │       └── explainability/
│   │           └── shap_explainer.py # SHAP charts + feature tables
│   ├── entity/                       # config + artifact dataclasses
│   ├── constant/                     # pipeline constants
│   ├── logging/                      # custom logger
│   └── exception/                    # custom exception handler
├── streamlit_pages/
│   ├── predictions.py               # Page 1: attack/benign predictions
│   ├── shap_explanation.py          # Page 2: SHAP charts
│   ├── performance.py               # Page 3: model metrics + plots
│   └── anomaly_detection.py         # Page 4: autoencoder results
├── streamlit_utils/
│   ├── loader.py                    # cached model loading
│   └── preprocessor.py             # feature selection + scaling
├── notebooks/                       # EDA + training Colab notebook
├── plots/                           # saved charts for dashboard + README
├── final_model/                     # trained model artifacts
├── data_schema/                     # schema.yaml for validation
├── streamlit_app.py                 # dashboard entry point
├── push_data.py                     # pushes CSV to MongoDB
├── Dockerfile                       # container config
├── requirements.txt                 # dependencies
└── .github/workflows/               # CI/CD pipeline
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| ML Models | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Explainability | SHAP |
| Experiment Tracking | MLflow, DagsHub |
| Data Store | MongoDB Atlas |
| Dashboard | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS (S3, EC2, ECR) |
| Dataset | CICIDS2017 |

---

## 🚀 How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/aman-verma02/Network-Security-System.git
cd Network-Security-System
```

**2. Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Set environment variables**
```bash
cp .env.example .env
# Add your MONGODB_URL to .env
```

**4. Run Streamlit dashboard**
```bash
streamlit run streamlit_app.py
```

---

## 📓 Notebook

The full EDA, feature selection, model training, and evaluation notebook is available in `Colab-Google/`. It includes:
- Class distribution analysis
- Correlation heatmap
- Feature importance plots
- Model comparison charts
- Confusion matrices

---

## 🔬 MLflow Experiments

All experiments tracked on DagsHub:
👉 [View Experiments](https://dagshub.com/aman-verma02/Network-Security-System)

---

## 👤 Author

**Aman Verma**
NIT Bhopal — Data Science, 2025
- GitHub: [@aman-verma02](https://github.com/aman-verma02)
- LinkedIn: [Aman Verma](https://www.linkedin.com/in/aman-verma2002/)