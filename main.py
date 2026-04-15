from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_valiation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys


if __name__ == '__main__':
    try: 
        ## Initializing configs
        training_pipeline_config = TrainingPipelineConfig()   

        # ------------------ Data Ingestion ------------------
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()  # ✅ store result
        logging.info("Completed the data ingestion")

        # ------------------ Data Validation ------------------
        logging.info("Initiate the data validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)

        data_validation = DataValidation(
            data_validation_config,
            data_ingestion_artifact   # ✅ pass artifact
        )

        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Completed the data validation")


        # ---------------------Data Transformation ------------------
        logging.info("Initiate the data transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact,
            data_transformation_config
        )

        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Completed the data transformation")


    except Exception as e: 
        raise NetworkSecurityException(e, sys)