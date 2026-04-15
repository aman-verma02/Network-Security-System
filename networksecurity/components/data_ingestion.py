from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging



## Configuration of the Data Ingestion Config
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact



import os
import sys 
import numpy as np
import pymongo
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")



class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):     #This variable "data_ingestion" SHOULD be of type DataIngestionConfig”
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def export_collection_as_dataframe(self): 
        '''
        Read data from mongodb database
        export data as dataframe and return it 
        '''
        try:
            self.database_name = self.data_ingestion_config.database_name
            self.collection_name = self.data_ingestion_config.collection_name
            self.client = pymongo.MongoClient(MONGODB_URL)
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]

            df = pd.DataFrame(list(self.collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    ## functon the keep the raw data in the loacal
    def export_data_into_feature_store(self, dataframe : pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def split_data_as_train_test(self, dataframe : pd.DataFrame):
        try:
            logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
            train_set,test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42,
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )       
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Exporting train and test file path.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,index=False,header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,index=False,header=True
            )

            logging.info(f"Exported train and test file path.")
       
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            logging.info(f"Exporting collection data as pandas dataframe")
            df:pd.DataFrame = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            logging.info(f"Save data in feature store")
            self.split_data_as_train_test(df)
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path, 
            test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
        
    