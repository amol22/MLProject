import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            processed_data = preprocessor.transform(features)
            pred = model.predict(processed_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
                 Year: int,
                 Kilometers_Driven: int,
                 Mileage: float,
                 Engine: int,
                 Power: int,
                 Seats: int,
                 Location: str,
                 Fuel_Type: str,
                 Transmission: str,
                 Brand: str,
                 Owner_Type: str):
        self.year = Year
        self.kilometers = Kilometers_Driven
        self.mileage = Mileage
        self.engine= Engine
        self.power = Power
        self.seats = Seats
        self.location = Location
        self.fueltype = Fuel_Type
        self.transmission = Transmission
        self.brand = Brand
        self.ownertype = Owner_Type
        
    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
                "Year": [self.year],
                "Kilometers_Driven": [self.kilometers],
                "Mileage": [self.mileage],
                "Engine": [self.engine],
                "Power": [self.power],
                "Seats": [self.seats],
                "Location": [self.location],
                "Fuel_Type": [self.fueltype],
                "Transmission": [self.transmission],
                "Brand": [self.brand],
                "Owner_Type": [self.ownertype]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)