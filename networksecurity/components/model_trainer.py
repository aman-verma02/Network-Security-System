import os 
import sys 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import load_numpy_array_data, load_object, save_object, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
import mlflow
import dagshub
dagshub.init(repo_owner='aman-verma02', repo_name='Network-Security-System', mlflow=True)

# import mlflow
# with mlflow.start_run():
#   mlflow.log_param('parameter name', 'value')
#   mlflow.log_metric('metric name', 1)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class ModelTrainer: 
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact): 

        try: 
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e : 
            raise NetworkSecurityException(e,sys)
    

    def track_mlflow(self, best_model, classification_metric):
        try:
            with mlflow.start_run():
                f1_score = classification_metric.f1_score
                precision_score = classification_metric.precision_score
                recall_score = classification_metric.recall_score
                mlflow.log_params(best_model.get_params())
                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision_score", precision_score)
                mlflow.log_metric("recall_score", recall_score)
                mlflow.sklearn.log_model(best_model, name="model")
        except Exception as e:
            raise NetworkSecurityException(e, sys)




    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression(verbose=1),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "RandomForestClassifier": RandomForestClassifier(verbose=1),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(verbose=1),
            }

            params = {
                "DecisionTreeClassifier":{
                    'criterion':['gini','entropy','log_loss'],
                    # 'splitter':['best', 'random'],
                    # 'max_features':['sqrt', 'log2']    
                },
                "RandomForestClassifier":{
                    'criterion':['gini','entropy','log_loss'],
                    
                    
                    'n_estimators':[8,16,32,64,128,256]
                },
                "GradientBoostingClassifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[.6,.7,.75,.8,.85,.9]
                },
                'LogisticRegression':{},
                'AdaBoostClassifier':{
                    'learning_rate':[.1,.01,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "KNeighborsClassifier":{
                    'n_neighbors':[3,5,10]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            ## To get best model score from dict 
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            ## Track the experiments with ML FLow
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_true =y_test, y_pred=y_test_pred)
            
            ## Track the experiments with ML FLow
            self.track_mlflow(best_model, classification_train_metric)
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)



            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            save_object("final_model/model.pkl", best_model)
            ## Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )


            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
    
        except Exception as e: 
            raise NetworkSecurityException(e,sys)
        



    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try: 
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )


            mode_trainer_artifact = self.train_model(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test)
            return mode_trainer_artifact



        except Exception as e: 
            raise NetworkSecurityException(e,sys)