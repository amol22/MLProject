import os
import sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts','data.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


params = {
        "Random Forest":{
            # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            
            # 'max_features':['sqrt','log2',None],
            'n_estimators': [8,16,32,64,128,256]
        },
        "Gradient Boosting":{
            # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
            'learning_rate':[.1,.01,.05,.001],
            'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
            # 'criterion':['squared_error', 'friedman_mse'],
            # 'max_features':['auto','sqrt','log2'],
            'n_estimators': [8,16,32,64,128,256]
        },
        "XGBRegressor":{
            'learning_rate':[.1,.01,.05,.001],
            'n_estimators': [8,16,32,64,128,256]
        }
        }