import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.config import DataIngestionConfig, DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ =="__main__":
    try:
        obj = DataIngestion()
        train_path,test_path = obj.data_ingestion()
        data_transform = DataTransformation()
        train_arr,test_arr = data_transform.initiate_data_transformation(train_path,test_path)
        modelTrainer = ModelTrainer()
        print(modelTrainer.initiate_model_trainer(train_arr,test_arr))
    except Exception as e:
        raise CustomException(e,sys)