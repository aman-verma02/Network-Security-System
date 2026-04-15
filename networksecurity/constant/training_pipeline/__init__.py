# Built-in libraries
import os   # Used for file and directory operations (paths, folders)
import sys  # Used for system-level operations (rarely used here, but helpful in pipelines)
import numpy as np


'''
Defining common constant variables for training pipeline
These values will be reused across different modules (ingestion, training, etc.)
'''
TARGET_COLUMN = "Result"  # Name of the target column in the dataset
PIPELINE_NAME: str = "NetworkSecurity" # Name of the pipeline (used for folder structure / logging / tracking)
ARTIFACT_DIR: str = "Artifacts"  # Root folder where all artifacts (outputs) will be stored
FILE_NAME: str = "phisingData.csv"  # Raw dataset file name
TRAIN_FILE_NAME: str = "train.csv" # Processed training dataset file
TEST_FILE_NAME: str = "test.csv"  # Processed testing dataset file

SAVED_MODEL_DIR = os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")


"""
Data ingestion related constants
Naming convention: DATA_INGESTION_<something>
This helps organize constants by pipeline stage
"""
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"  # MongoDB collection name (where data is stored)
DATA_INGESTION_DATABASE_NAME: str = "JACKAI"  # MongoDB database name
DATA_INGESTION_DIR_NAME: str = "data_ingestion" # Folder name for ingestion stage inside artifacts
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store" # Folder where raw data is stored after fetching from DB
DATA_INGESTION_INGESTED_DIR: str = "ingested" # Folder where processed/split data is stored
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2 # Train-test split ratio (20% test, 80% train)


"""
Data validation related constants
Naming convention: DATA_VALIDATION_<something>
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation" # Folder name for validation stage inside artifacts   
DATA_VALIDATION_VALID_DIR: str = "validated" # Folder where validated data is stored
DATA_VALIDATION_INVALID_DIR: str = "invalid" # Folder where invalid data is stored
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report" # Folder where drift report is stored
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml" # Name of the drift report file


"""
Data transformation related constants
Naming convention: DATA_TRANSFORMATION_<something>
""" 
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation" # Folder name for transformation stage inside artifacts
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed" # Folder where transformed data is stored 
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object" # Folder where transformed object is stored
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl" # Name of the preprocessing object file

## KKN imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform"

}

DATA_TRANFORMATION_TRAIN_FILE_PATH: str = "train.npy"
DATA_TRANFORMATION_TEST_FILE_PATH: str = "test.npy"

"""
Model trainer related constants
Naming convention: MODEL_TRAINER_<something>
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer" # Folder name for training stage inside artifacts
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model" # Folder where trained model is stored
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl" # Name of the trained model
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FIIITING_THRESHOLD: float = 0.05