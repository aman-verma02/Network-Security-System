from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import TrainingPipelineConfig


import sys

if __name__ == '__main__':
    try: 
        ## Intializing the classes
        training_pipeline_config = TrainingPipelineConfig()   
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion")
        data_ingestion.initiate_data_ingestion()
        logging.info("Completed the data ingestion")

    except Exception as e: 
        raise NetworkSecurityException(e,sys)