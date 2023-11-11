import sys
import os
import dill
import numpy as np
import pandas as pd
from typing import Tuple
from src.exception import CustomException
from sklearn.metrics import r2_score

def get_num(value: str) -> float:
    try:
        strRep = str(value)
        floatRep = ""
        for char in strRep:
            if not char.isalpha() and not char.isspace() and char != '/':
                floatRep += char

        return float(floatRep)
    except:
        return np.NaN
    
def save_object(file_path:str,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train:np.array,y_train:np.array,X_test:np.array,y_test:np.array,models:dict):
    try:
        report={}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            model_train_score = r2_score(y_train,y_train_pred)
            model_test_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = model_test_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
            
        