""" 
This transformation converts raw data into a format suitable for analysis:
- Handles missing values
- Encodes categorical variables
- Scales numerical variables
"""

import sys
import os
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """Configuration for Data Transformation"""
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_df):
        """Builds the preprocessing object dynamically based on available columns"""
        try:
            # Potential columns in the dataset
            numerical_columns = ['Total_Sales', 'Quantity_Purchased', 'Unit_Price']
            categorical_columns = [
                "Product_Category",
                "Product_Name",
                "Payment_Method",
                "Order_ID",
                "Customer_ID",
                "Product_ID",
                "Product_Source",
                "Month_Name",
                "Weekday",
            ]

            # Filter out missing columns automatically
            available_num_cols = [col for col in numerical_columns if col in train_df.columns]
            available_cat_cols = [col for col in categorical_columns if col in train_df.columns]

            missing_cols = set(numerical_columns + categorical_columns) - set(train_df.columns)
            if missing_cols:
                logging.warning(f"Missing columns (skipped): {missing_cols}")

            logging.info(f"Numerical columns used: {available_num_cols}")
            logging.info(f"Categorical columns used: {available_cat_cols}")

            # Define pipelines
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info("Pipelines for categorical and numerical features created successfully.")

            # Combine both
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, available_num_cols),
                    ("cat_pipeline", cat_pipeline, available_cat_cols)
                ],
                remainder="drop"  # drop other unused columns
        )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Train and test data read successfully.")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            target_column_name = "Total_Sales"
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in dataset", sys)

            preprocessing_obj = self.get_data_transformer_object(train_df)

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], errors='ignore')
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], errors='ignore')
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing to train and test data...")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine preprocessed features with targets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data Transformation completed successfully.")
            logging.info(f"Preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
