import sys
import os 
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import TARGET_COLUMN
from networksecurity.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_object, save_numpy_array_data
from networksecurity.entity.artifact_entity import DataTransformationArtifact






class DataTransformation: 
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):

        try: 
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config

        except Exception as e: 
            raise NetworkSecurityException(e,sys) 
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)



    ## Intiating KNN imputer to replace Nan value with something meaningful value
    def get_data_transformer_object(self) -> Pipeline: 
        '''
        It initialises a KNNImputer object with the parameter specified in the training_pipeline.py file and returns a Pipelline object with the KNNImputer object as the first step

        Args : 
            cls : DataTransformation

        Returns : 
            A Pipeline Object
        '''
        logging.info("Enterd get_data_transformer_object method of DataTransformation class")
        try: 
            knn_imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Intialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor: Pipeline = Pipeline(
                steps=[
                    ("knn_imputer", knn_imputer)
                ]
            )
            return processor
        except Exception as e: 
            raise NetworkSecurityException(e,sys)
        

    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        print(type(self.data_validation_artifact))
        logging.info(f"Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            ## trainig dataframe 
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            ## testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)


            preprocessor_object = self.get_data_transformer_object()
            preprocessor_object.fit(input_feature_train_df)

            tranformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[tranformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]


            ## Save numpy array data

            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=  train_arr)
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path, array=  test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, obj = preprocessor_object)

            save_object("final_model/preprocessor.pkl", preprocessor_object)

            logging.info("Completed data transformation")

            ## preparing artifacts 
            data_transdormation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            logging.info(f"Data transformation artifact: {data_transdormation_artifact}")

            return data_transdormation_artifact

        except Exception as e:
            raise NetworkSecurityException(e,sys)