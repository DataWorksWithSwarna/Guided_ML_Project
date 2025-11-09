import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            #  Use absolute path for dataset
            data_path = os.path.join(os.getcwd(), "src", "notebook", "data", "Souq_Ecommerce_raw_data.csv")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at path: {data_path}")

            df = pd.read_csv(data_path)
            logging.info(f"Dataset loaded successfully with shape {df.shape}")

            #  Create artifacts directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #  Save the raw dataset to artifacts/data.csv
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved successfully")

            #  Train-test split
            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            #  Overwrite old files if exist
            for file_path in [self.ingestion_config.train_data_path, self.ingestion_config.test_data_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            #  Save the train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(
                f"Data Ingestion completed successfully! "
                f"Train shape: {train_set.shape}, Test shape: {test_set.shape}"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(f" Data Ingestion completed.\nTrain file: {train_path}\nTest file: {test_path}")
    
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_path, test_path)
