# Built-in libraries
import os   # Used for file and directory operations (paths, folders)
import sys  # Used for system-level operations (rarely used here, but helpful in pipelines)




'''
Defining common constant variables for training pipeline
These values will be reused across different modules (ingestion, training, etc.)
'''

# Target column in dataset (what we want to predict)
TARGET_COLUMN = "Result"

# Name of the pipeline (used for folder structure / logging / tracking)
PIPELINE_NAME: str = "NetworkSecurity"

# Root folder where all artifacts (outputs) will be stored
ARTIFACT_DIR: str = "Artifacts"

# Raw dataset file name
FILE_NAME: str = "phisingData.csv"

# Processed training dataset file
TRAIN_FILE_NAME: str = "train.csv"

# Processed testing dataset file
TEST_FILE_NAME: str = "test.csv"


"""
Data ingestion related constants
Naming convention: DATA_INGESTION_<something>
This helps organize constants by pipeline stage
"""

# MongoDB collection name (where data is stored)
DATA_INGESTION_COLLECTION_NAME: str = "NetworkData"

# MongoDB database name
DATA_INGESTION_DATABASE_NAME: str = "JACKAI"

# Folder name for ingestion stage inside artifacts
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Folder where raw data is stored after fetching from DB
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# Folder where processed/split data is stored
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Train-test split ratio (20% test, 80% train)
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2