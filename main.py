from src.forest_cover_type.data_ingestion import DataIngestion
from src.forest_cover_type.data_evaluation import ModelEvaluation
from src.forest_cover_type.data_trainer import ModelTrainer
from src.forest_cover_type.data_transformation import DataTransformation

obj = DataIngestion()

import pyfiglet
f = pyfiglet.figlet_format("Welcome to Abdullah", font="slant")
print(f)
f = pyfiglet.figlet_format("Forest Cover Type Prediction", font="digital")
print(f)

print("\nData Ingestion Started\n")
train_data_path, test_data_path = obj.initiate_data_ingestion()

print("\ndata Ingestion completed and data transformation started.\n")
data_transformation= DataTransformation()
train_arr, test_arr = data_transformation.initiatize_data_transformation(train_data_path,test_data_path)

print("\nData Transformation completed and Model Training started.\n")
model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(train_arr, test_arr)

print("\nModel training completed and Model Evaluation started\n")
model_eval_obj = ModelEvaluation()
model_eval_obj.initiated_model_evaluation(train_arr,test_arr)
print("\nALL the steps has completed.\n\n")

f = pyfiglet.figlet_format("Thank You", font="big")
print(f)