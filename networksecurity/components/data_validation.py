from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from scipy.stats import ks_2samp   # ✅ CORRECT
import numpy as np
import os
import sys
import pandas as pd
from typing import Optional
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file
from networksecurity.utils.main_utils.utils import write_yaml_file




class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    ## functon to validate the numbers of columns
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    ## Function to check whether we have numerical columns
    def is_numerical_column_exist(self, dataframe: pd.DataFrame, column_name: str) -> bool:
        try:
            number_of_columns = self._schema_config["numerical_columns"]
            if column_name in number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    ## Function to Detect dataset drift
    def detect_dataset_drift(self, base_dataframe: pd.DataFrame, current_dataframe: pd.DataFrame, threshold: float = 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_dataframe.columns:
                d1 = base_dataframe[column]
                d2 = current_dataframe[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= float(is_same_dist.pvalue):
                    is_found = False                
                else: 
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            ## Creating the directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)




    def initiate_data_validation(self) -> DataValidationArtifact:
        
        try: 
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the. data from train and tst
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            ## Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status : 
                error_message = f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status : 
                error_message = f"Test dataframe does not contain all columns.\n"


            # ## Checking numerical columns exist or not
            # numerical_columns = self._schema_config["numerical_columns"]
            # for column in numerical_columns:
            #     status = self.is_numerical_column_exist(dataframe=train_dataframe, column_name=column)
            #     if not status:
            #         error_message = f"{error_message}Column: {column} is not numerical column.\n"
            #         break
            #     status = self.is_numerical_column_exist(dataframe=test_dataframe, column_name=column)       
            
            ## Lets check Datadrift 
            status = self.detect_dataset_drift(base_dataframe=train_dataframe, current_dataframe=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            self.data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return self.data_validation_artifact 
            

        except Exception as e:
            raise NetworkSecurityException(e, sys)

        