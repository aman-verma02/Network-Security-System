# networksecurity/pipeline/training_pipeline.py
#
# PURPOSE:
# Orchestrates the entire ML pipeline from data ingestion to model training.
# Each stage produces an "artifact" — an object containing output file paths
# that gets passed as input to the next stage.
#
# FLOW:
# DataIngestion → DataValidation → DataTransformation → ModelTrainer → AnomalyDetector

import os
import sys
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Import all pipeline components
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.anomaly_detector import AnomalyDetector  # NEW

# Import all config entities
# Config entities define WHERE to save outputs for each stage
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

# Import all artifact entities
# Artifact entities CARRY the output paths between stages
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.utils.main_utils.utils import load_numpy_array_data


class TrainingPipeline:
    """
    Master pipeline class that runs all stages in sequence.
    Each start_*() method:
        1. Creates a config object (where to save outputs)
        2. Creates the component object
        3. Calls the component's initiate_*() method
        4. Returns the artifact (output paths) for the next stage
    """

    def __init__(self):
        # TrainingPipelineConfig holds global settings like
        # artifact directory, timestamp, model save path
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Stage 1: Pull data from MongoDB and save as train/test CSV files.

        - Reads from MongoDB collection (CICIDSData)
        - Splits into train and test sets
        - Saves CSVs to artifact directory
        - Returns paths to those CSVs via DataIngestionArtifact
        """
        try:
            logging.info("Starting data ingestion")
            data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(
                data_ingestion_config=data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Stage 2: Validate the ingested data against schema.

        - Checks column names match schema.yaml
        - Checks data types
        - Saves validated train/test CSVs
        - Returns paths via DataValidationArtifact
        """
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_validation = DataValidation(
                data_validation_config=data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(
        self, data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        Stage 3: Feature selection + scaling.

        - Drops correlated features (>0.95)
        - Drops low variance features
        - Selects top 40 by Random Forest importance
        - Applies StandardScaler
        - Saves transformed numpy arrays + preprocessor.pkl + feature_names.pkl
        - Returns paths via DataTransformationArtifact
        """
        try:
            logging.info("Starting data transformation")
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """
        Stage 4: Train sklearn models with grid search.

        - Loads transformed numpy arrays from DataTransformationArtifact
        - Applies SMOTE for class imbalance
        - Trains multiple models (RF, XGBoost, etc.) with grid search
        - Picks best model by F1 score
        - Tracks metrics to MLflow + DagsHub
        - Saves model.pkl and preprocessor.pkl to final_model/
        - Returns metrics via ModelTrainerArtifact
        """
        try:
            logging.info("Starting model training")
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_anomaly_detector(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> dict:
        """
        Stage 5: Train PyTorch Autoencoder for anomaly detection.

        - Loads the SAME transformed numpy arrays used by ModelTrainer
          (no extra processing needed — data is already scaled)
        - Trains Autoencoder on benign samples only
        - Sets anomaly threshold = mean + 2*std of benign errors
        - Evaluates on full test set
        - Saves autoencoder.pkl to final_model/
        - Returns dict with threshold, accuracy, detection_rate, false_alarm_rate

        NOTE: input_dim=40 must match the number of features selected
        in data_transformation.py (top 40 by RF importance)
        """
        try:
            logging.info("Starting anomaly detector training")

            # Load the same numpy arrays ModelTrainer used
            # These are already scaled — ready for the Autoencoder
            train_arr = load_numpy_array_data(
                data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                data_transformation_artifact.transformed_test_file_path
            )

            # Split features and target
            # Last column is always the target (how save_numpy_array_data saves it)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # input_dim = number of features after selection (40)
            # epochs=20 is enough for this dataset
            # batch_size=256 processes 256 rows at a time
            anomaly_detector = AnomalyDetector(
                input_dim=X_train.shape[1],  # automatically picks correct feature count
                epochs=20,
                batch_size=256,
                lr=0.001
            )

            # Train + evaluate + save autoencoder.pkl
            results = anomaly_detector.initiate_anomaly_detector(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                save_path="final_model/autoencoder.pkl"
            )

            logging.info(f"Anomaly detector results: {results}")
            logging.info("Anomaly detector training completed")

            return results

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def run_pipeline(self):
        """
        Master method that runs all 5 stages in sequence.
        Each stage output feeds into the next stage as input.

        Call this from main.py to run the full pipeline:
            pipeline = TrainingPipeline()
            pipeline.run_pipeline()
        """
        try:
            # Stage 1: MongoDB → CSV files
            data_ingestion_artifact = self.start_data_ingestion()

            # Stage 2: CSV → validated CSV files
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # Stage 3: validated CSV → scaled numpy arrays + preprocessor.pkl
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )

            # Stage 4: numpy arrays → best sklearn model + model.pkl
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            # Stage 5: same numpy arrays → autoencoder.pkl
            # Runs AFTER model trainer so both models are trained in one pipeline run
            anomaly_detector_results = self.start_anomaly_detector(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("Full training pipeline completed successfully")
            logging.info(f"Anomaly Detector — Threshold: {anomaly_detector_results['threshold']:.4f}")
            logging.info(f"Anomaly Detector — Detection Rate: {anomaly_detector_results['detection_rate']:.4f}")

        except Exception as e:
            raise NetworkSecurityException(e, sys)