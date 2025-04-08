import os
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataPreprocessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        # self.config_path = config_path

        # Load configuration
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            # Create directories if they do not exist
            os.makedirs(processed_dir, exist_ok=True)

    def preprocess_data(self, df):
        try:
            # Load the configuration for preprocessing
            logger.info("Starting data preprocessing...")

            logger.info("Dropping some columns")
            df.drop(columns=["Unnamed: 0", "Booking_ID"], inplace=True)
            df.drop_duplicates(inplace=True)

            # Get preprocessing configuration from config file (config YAML)
            preprocessing_config = self.config["data_processing"]
            # Extract categorical feature names
            categorical_columns = preprocessing_config["categorical_columns"]
            # Extract numerical feature names
            numerical_columns = preprocessing_config["numerical_columns"]
            # Get target variable column name
            target_column = preprocessing_config["target_column"]

            logger.info("Applying Label Encoding")
            label_encoder = LabelEncoder()
            mappings = {}
            for column in categorical_columns:
                df[column] = label_encoder.fit_transform(df[column])
                mappings[column] = {label: code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}


            logger.info("Label Mappings are:")
            for column, mapping in mappings.items():
                logger.info(f"{column}: {mapping}")


            logger.info("Doing Skewness Handling")
            skew_threshold = preprocessing_config["skewness_threshhold"]
            skewness = df[numerical_columns].apply(lambda x: x.skew())
            for column in skewness[skewness > skew_threshold].index:
                df[column] = np.log1p(df[column])
            
            return df

        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException("Data preprocessing failed", e)
    
    def balance_data(self, df):
        try:
            logger.info("Handling inbalanced using SMOTE")
            target_column = self.config["data_processing"]["target_column"]
            X = df.drop(columns=target_column)
            y = df[target_column]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # converting to DataFrame
            balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_data[target_column] = y_resampled
            balanced_data.reset_index(drop=True, inplace=True)

            logger.info("Data balancing completed successfully")
            return balanced_data
        except Exception as e:
            logger.error(f"Error during data balancing: {e}")
            raise CustomException("Data balancing failed", e)

    def select_features(self, df):
        try:
            logger.info("Feature Selection")
            target_column = self.config["data_processing"]["target_column"]
            X = df.drop(columns=target_column)
            y = df[target_column].values.ravel()

            rf_model = RandomForestClassifier(random_state=42)
            rf_model.fit(X, y)

            # Get feature importances
            feature_importance = rf_model.feature_importances_
            # Create a DataFrame to hold feature importances
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
            top_features_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            # Select features based on importance threshold
            no_of_features_to_select = self.config["data_processing"]["no_of_features_to_select"]
            
            # retrieving columns names from top 10 features
            top_N_features = top_features_importance_df["Feature"].head(no_of_features_to_select).values

            logger.info(f"Top {no_of_features_to_select} features selected: {top_N_features}")
            # print("*****************")
            # print(target_column)
            # print("*****************")
            # retrieva data from top 10 features including target variable
            top_N_df = df[top_N_features.tolist()+[target_column[0]]]

            logger.info("Feature Selection completed successfully")

            return top_N_df

        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException("Feature selection failed", e)
        
    def save_processed_data(self, df, file_path):
        try:
            logger.info("Saving processed data...")
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException("Saving processed data failed", e)

    def process(self):
        try:
            # Load the training and testing data
            logger.info("Loading training and testing data...")
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)

            # Preprocess the data
            train_data = self.preprocess_data(train_data)
            test_data = self.preprocess_data(test_data)

            # Balance the training data
            balanced_train_data = self.balance_data(train_data)

            # Feature selection
            selected_train_data = self.select_features(balanced_train_data)
            selected_test_data = test_data[selected_train_data.columns]

            self.save_processed_data(selected_train_data, PROCESSED_TRAIN_FILE_PATH)
            self.save_processed_data(selected_test_data, PROCESSED_TEST_FILE_PATH)

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            raise CustomException("Data pipeline processing failed", e)
        
if __name__ == "__main__":
    processor = DataPreprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESSED_DIR, CONFIG_PATH)
    processor.process()