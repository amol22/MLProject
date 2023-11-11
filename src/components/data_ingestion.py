import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DataIngestionConfig, DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def data_ingestion(self):
        """Ingests the data into a dataframe and performs a train-test split
        """
        logging.info("Data ingestion initiated")
        try:
            df = pd.read_csv("data/train-data.csv")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok  = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 22)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data ingestion completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.data_ingestion()
    data_transform = DataTransformation()
    train_arr,test_arr = data_transform.initiate_data_transformation(train_path,test_path)
    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))