import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from src.exception import CustomException
from src.logger import logging
from config import DataTransformationConfig
from utils import get_num, save_object


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def clean_data(self,df:pd.DataFrame) -> pd.DataFrame:
        """Cleans and formats the dataframe

        Args:
            df (pd.DataFrame): raw dataframe

        Raises:
            CustomException: exception

        Returns:
            pd.DataFrame: cleaned dataframe
        """
        try:
            df["Brand"]=df.Name.apply(lambda x: x.split(" ")[0])
            df=df.drop(["Unnamed: 0","New_Price","Name"],axis=1)
            df["Mileage"] = round(df["Mileage"].apply(get_num),2)
            df["Engine"] = round(df["Engine"].apply(get_num),0)
            df["Power"] = round(df["Power"].apply(get_num),0)
            df=df.dropna()
            
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def get_data_transformer(self):
        """Initiates and gets the data transformer object

        Raises:
            CustomException: exception
        """
        try:
            num_cols = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats']
            cat_cols_1 = ['Location', 'Fuel_Type', 'Transmission', 'Brand']
            cat_cols_2 = ["Owner_Type"]
            
            num_pipeline = Pipeline(
                steps = [("scaler", StandardScaler())]
            )
            
            cat_pipeline_1 = Pipeline(
                steps = [("One-hot encoder", OneHotEncoder())]
            )
            
            cat_pipeline_2 = Pipeline(
                steps = [("Ordinal encoder", OrdinalEncoder(categories=[["First","Second","Third","Fourth & Above"]]))]
            )
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_cols),
                ("cat_pipeline_1",cat_pipeline_1,cat_cols_1),
                ("cat_pipeline_2",cat_pipeline_2,cat_cols_2)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_df_clean = self.clean_data(train_df)
            test_df_clean = self.clean_data(test_df)
            list1 = list(train_df_clean.Brand.unique())
            list2 = list(test_df_clean.Brand.unique())
            extra_elements = list(set(list2) - set(list1))
            test_df_clean_filtered = test_df_clean[~test_df_clean['Brand'].isin(extra_elements)]
            logging.info("train and test datasets read")
            
            preprocessor = self.get_data_transformer()
            
            logging.info("Preprocessor object obtained")
            
            target_column_name = 'Price'
            
            input_feature_train_df = train_df_clean.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df_clean[target_column_name]
            input_feature_test_df = test_df_clean_filtered.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df_clean_filtered[target_column_name]
            
            logging.info("applying preprocessor")
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info(input_feature_train_arr.shape)
            logging.info(np.array(target_feature_train_df).shape)
            train_arr = np.c_[input_feature_train_arr.toarray(),np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr.toarray(),np.array(target_feature_test_df)]
            logging.info(train_arr.shape)
            
            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path, 
                obj = preprocessor
                )
            
            logging.info("Preprocessor object trained and stored")
            
            return (train_arr,test_arr,self.transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)