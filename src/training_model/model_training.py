import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils.logger import get_logger
from utils.custom_exception import CustomException
from utils.common_functions import read_yaml, load_data
from config.paths_config import *
from config.model_params import *

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        # Load configuration
        self.config = read_yaml(config_path)

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading data from {self.train_path}")
            # Load the training data
            train_df = load_data(self.train_path)

            logger.info(f"Loading data from {self.test_path}")
            # Load the testing data
            test_df = load_data(self.test_path)

            # Split the data into features and target variable
            target_column = self.config["data_processing"]["target_column"][0]

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]#.values.ravel()

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]#.values.ravel()
            
            logger.info("Data loaded and splitted for model training.")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error in loading and splitting data: {e}")
            raise CustomException('Failed to load data',e)

    def train_model(self, X_train, y_train):
        try:
            logger.info("Initiating model training")
            # Initialize the LightGBM model
            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])
            
            logger.info("Starting hyperparameter tuning")
            # Perform Randomized Search CV for hyperparameter tuning
            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                n_jobs=self.random_search_params['n_jobs'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            logger.info("Fitting the model")
            random_search.fit(X_train, y_train)
            
            logger.info("Hyperarameter tuning completed successfully")
            best_params = random_search.best_params_
            best_model = random_search.best_estimator_
            
            logger.info(f"Best parameters found: {best_params}")
            return best_model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise CustomException('Model training failed', e)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the model")
            # Make predictions
            y_pred = model.predict(X_test)
            logger.info("Predictions made successfully")

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            logger.info(f"Accuracy: {accuracy}")            
            logger.info(f"Precision: {precision}")            
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            logger.info(f"Model evaluation completed successfully")
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise CustomException('Model evaluation failed', e)

    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)

            logger.info("Saving the model")            
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException('Model saving failed', e)
        
    def run(self):
        try:
            with mlflow.start_run():

                logger.info("Starting model training pipeline")

                logger.info("Starting our  MLFLOW experimentation")

                logger.info("Logging the training and testing dataset into MLFLOW")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")


                # Load and split the data
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                
                # Train the model
                model = self.train_model(X_train, y_train)
                
                # Evaluate the model
                metrics = self.evaluate_model(model, X_test, y_test)
                
                # Save the model
                self.save_model(model)

                logger.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path, artifact_path="models")

                logger.info("Logging the model parameters into MLFLOW")
                mlflow.log_params(model.get_params())

                logger.info("Logging the model metrics into MLFLOW")
                mlflow.log_metrics(metrics)
                
                logger.info(f"Model training pipeline completed successfully")
            return metrics

        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException('Model training pipeline failed', e)

if __name__ == "__main__":
    
    # Create an instance of the ModelTraining class and run the pipeline
    model_trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_OUTPUT_DIR, CONFIG_PATH)
    metrics = model_trainer.run()