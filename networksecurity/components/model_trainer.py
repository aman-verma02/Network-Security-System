import os
import sys
import numpy as np
import mlflow
import dagshub

dagshub.init(repo_owner='aman-verma02', repo_name='Network-Security-System', mlflow=True)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model_name, model, classification_metric, roc_auc, dataset_type):
        """Track metrics and model to MLflow + DagsHub."""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{dataset_type}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("dataset", dataset_type)
                mlflow.log_metric("f1_score", classification_metric.f1_score)
                mlflow.log_metric("precision_score", classification_metric.precision_score)
                mlflow.log_metric("recall_score", classification_metric.recall_score)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.sklearn.log_model(model, name="model")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            # Handle class imbalance with SMOTE
            logging.info("Applying SMOTE to handle class imbalance")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"After SMOTE — Train shape: {X_train.shape}")

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(verbose=1),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
                "XGBClassifier": XGBClassifier(
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                ),
            }

            params = {
                "LogisticRegression": {},
                "DecisionTreeClassifier": {
                    'criterion': ['gini', 'entropy'],
                },
                "RandomForestClassifier": {
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [64, 128]
                },
                "GradientBoostingClassifier": {
                    'learning_rate': [0.1, 0.05],
                    'subsample': [0.8, 0.9]
                },
                "AdaBoostClassifier": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [64, 128]
                },
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6]
                },
            }

            # Evaluate all models with grid search
            logging.info("Starting model evaluation with grid search")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # Get best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} with score: {best_model_score:.4f}")

            # Train metrics
            y_train_pred = best_model.predict(X_train)
            y_train_prob = best_model.predict_proba(X_train)[:, 1]
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )
            train_roc_auc = roc_auc_score(y_train, y_train_prob)

            # Test metrics
            y_test_pred = best_model.predict(X_test)
            y_test_prob = best_model.predict_proba(X_test)[:, 1]
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )
            test_roc_auc = roc_auc_score(y_test, y_test_prob)

            logging.info(f"Train — F1: {classification_train_metric.f1_score:.4f}, ROC-AUC: {train_roc_auc:.4f}")
            logging.info(f"Test  — F1: {classification_test_metric.f1_score:.4f}, ROC-AUC: {test_roc_auc:.4f}")

            # Track to MLflow + DagsHub
            self.track_mlflow(best_model_name, best_model, classification_train_metric, train_roc_auc, "train")
            self.track_mlflow(best_model_name, best_model, classification_test_metric, test_roc_auc, "test")

            # Save model
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            save_object("final_model/model.pkl", best_model)

            logging.info(f"Saved best model: {best_model_name}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Entered initiate_model_trainer")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)