from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from sklearn.metrics import f1_score, precision_score, recall_score
import os, sys




def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Calculate classification metrics (F1 score, precision, recall) and return as a ClassificationMetricArtifact.
    Args:
        y_true: array-like of shape (n_samples,) - True labels.
        y_pred: array-like of shape (n_samples,) - Predicted labels.
    Returns:
        ClassificationMetricArtifact: An object containing the calculated F1 score, precision, and recall.
    """
    try: 
        model_f1_score = f1_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e: 
        raise NetworkSecurityException(e,sys)