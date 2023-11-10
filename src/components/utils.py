import sys
import os
import dill
import numpy as np
import pandas as pd
from typing import Tuple
from src.exception import CustomException

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
        