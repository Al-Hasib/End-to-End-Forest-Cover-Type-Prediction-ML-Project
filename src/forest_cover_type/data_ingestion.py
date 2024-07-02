import pandas as pd
import numpy as np
from src.exception.exception import customexception
from src.logger.logging import logging

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts", "raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("data ingestion started..")

        try:
            data= pd.read_csv(r"C:\Users\abdullah\projects\Machine_Learning_projects\Forest Cover Type Prediction\End-to-End-Forest-Cover-Type-Prediction-ML-Project\data\train.csv")
            logging.info(" reading a df")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(" raw data saved in the artifact folder")

            train_data, test_data = train_test_split(data, test_size=0.2)
            logging.info("train test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("data ingestion part completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("problem raise in the data ingestion part")
            raise customexception(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()