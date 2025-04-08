
from src.data_preprocessing.data_ingestion import DataIngestion
from src.data_preprocessing.data_preprocessing import DataPreprocessor
from src.training_model.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *


if __name__ == "__main__":
    # 001. Data ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))

    # 002. Data preprocessing
    processor = DataPreprocessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PREPROCESSED_DIR, CONFIG_PATH)
    processor.process()

    # 003. Model training
    model_trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH, PROCESSED_TEST_FILE_PATH, MODEL_OUTPUT_DIR, CONFIG_PATH)
    metrics = model_trainer.run()