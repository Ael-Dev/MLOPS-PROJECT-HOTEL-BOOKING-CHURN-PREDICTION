import os
import pandas
from utils.logger import get_logger
from utils.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Succesfully read the YAML file")
            return config
        
    except Exception as e:
        logger.error("Error while reading YAML file")
        raise CustomException("Fail to read YAML file", e)
    


def load_data(path):
    try:
        if not os.path.exists(path):
            logger.error("File is not in the given path")
            raise FileNotFoundError(f"File is not in the given path")
        
        data = pandas.read_csv(path)
        logger.info("Data is loaded successfully")
        return data
    
    except Exception as e:
        logger.error(f"Error while loading the data {e}")
        raise CustomException("Failed to load data", e)



