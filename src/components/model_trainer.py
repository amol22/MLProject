import os
import sys
from src.exception import CustomException
from src.logger import logging
from utils import save_object,evaluate_models
from config import ModelTrainerConfig
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info("Train and test arrays split")
            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor()
            }
            
            models_report = evaluate_models(X_train,y_train,X_test,y_test,models)
            best_model_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")
            
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test,predicted)
            
            return r2score
        except Exception as e:
            raise CustomException(e,sys)
