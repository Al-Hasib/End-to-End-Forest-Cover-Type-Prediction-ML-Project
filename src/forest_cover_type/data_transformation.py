import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')

            # pipeline added
            pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='medium')),
                    ('scaler',StandardScaler())
                ]
            )

            return pipeline
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e, sys)
        
    
    def initiatize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read train and test data completed")
            logging.info(f'Train Dataframe head : \n{train_df.head().to_string()}')
            logging.info(f"test Dataframe Head : \n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'Cover_Type'
            drop_columns = [target_column_name, "Id"]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.arrya(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("preprocessing pickle file saved")

            return (train_arr, test_arr)
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e, sys)
        