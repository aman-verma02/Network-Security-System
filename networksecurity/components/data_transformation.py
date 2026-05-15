import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_selected_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Step 1 — Drop highly correlated features (>0.95)
        Step 2 — Drop low variance features
        Step 3 — Keep top 40 by Random Forest importance
        Returns list of selected feature names.
        """
        try:
            logging.info("Starting feature selection")

            # Step 1: Drop highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
            X = X.drop(columns=drop_corr)
            logging.info(f"Dropped {len(drop_corr)} highly correlated features")

            # Step 2: Drop low variance features
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X)
            drop_variance = X.columns[~selector.get_support()].tolist()
            X = X[X.columns[selector.get_support()]]
            logging.info(f"Dropped {len(drop_variance)} low variance features")

            # Step 3: Top 40 features by Random Forest importance
            X_sample = X.sample(n=min(50000, len(X)), random_state=42)
            y_sample = y[X_sample.index]

            rf = RandomForestClassifier(
                n_estimators=50, random_state=42, n_jobs=-1
            )
            rf.fit(X_sample, y_sample)

            importances = pd.Series(rf.feature_importances_, index=X.columns)
            top_features = importances.nlargest(40).index.tolist()
            logging.info(f"Selected top {len(top_features)} features by importance")

            return top_features

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Returns a Pipeline with StandardScaler only.
        Feature selection is done separately before this step.
        """
        try:
            logging.info("Creating preprocessing pipeline with StandardScaler")
            processor = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation")
        try:
            # Read validated data
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"Train shape: {input_feature_train_df.shape}")
            logging.info(f"Test shape: {input_feature_test_df.shape}")

            # Feature selection (fit on train only)
            top_features = self.get_selected_features(
                input_feature_train_df,
                target_feature_train_df
            )

            # Keep only selected features
            input_feature_train_df = input_feature_train_df[top_features]
            input_feature_test_df = input_feature_test_df[top_features]

            # Scale
            preprocessor = self.get_data_transformer_object()
            preprocessor.fit(input_feature_train_df)

            transformed_train = preprocessor.transform(input_feature_train_df)
            transformed_test = preprocessor.transform(input_feature_test_df)

            # Combine features + target into arrays
            train_arr = np.c_[transformed_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_test, np.array(target_feature_test_df)]

            # Save arrays
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # Save preprocessor
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )
            save_object("final_model/preprocessor.pkl", preprocessor)

            # Save selected feature names — needed by Streamlit + SHAP later
            save_object("final_model/feature_names.pkl", top_features)

            logging.info("Data transformation completed")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)