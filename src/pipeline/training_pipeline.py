import os
import sys
from src.logger.logging import logging
from src.exception.exception import customexception
import pandas as pd

from src.forest_cover_type.data_ingestion import DataIngestion
from src.forest_cover_type.data_evaluation import ModelEvaluation
from src.forest_cover_type.data_trainer import ModelTrainer
from src.forest_cover_type.data_transformation import DataTransformation

obj = DataIngestion()

train_data_path, test_data_path = obj.initiate_data_ingestion()

data_transformation= DataTransformation()

train_arr, test_arr = data_transformation.initiatize_data_transformation(train_data_path,test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr, test_arr)

model_eval_obj = ModelEvaluation()
model_eval_obj.initiated_model_evaluation(train_arr,test_arr)