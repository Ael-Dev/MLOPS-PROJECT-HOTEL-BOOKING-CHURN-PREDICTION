import os

########################## DATA INGESTION ###########################
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = "config/config.yaml"


######################### DATA PROCESSING ###########################
PREPROCESSED_DIR = "artifacts/preprocessing"
PROCESSED_TRAIN_FILE_PATH = os.path.join(PREPROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_FILE_PATH = os.path.join(PREPROCESSED_DIR, "processed_test.csv")


######################### MODEL TRAINING ############################
MODEL_OUTPUT_DIR = "artifacts/models/lgbm_model.pkl"





