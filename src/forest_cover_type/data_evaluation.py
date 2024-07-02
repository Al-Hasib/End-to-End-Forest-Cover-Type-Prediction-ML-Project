import os
import sys
import numpy as np
import pickle
from src.utils.utils import load_object
from src.logger.logging import logging
from src.exception.exception import customexception
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        confusion = confusion_matrix(actual, pred)
        
        # precision = precision_score(actual, pred)
        # recall = recall_score(actual, pred)
        # f1 = f1_score(actual,pred)

        return accuracy, confusion
    
    def initiated_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:,:-1], test_array[:,-1])
            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)

            pred = model.predict(X_test)

            accuracy, confusion = self.eval_metrics(y_test, pred)

            information = (f"Accuracy {accuracy}\n confusion matrix \n{confusion}\n The Training pipeline has completed...")

            logging.info(information)
            print(information)

        except Exception as e:
            raise customexception(e,sys)