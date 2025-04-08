import logging
import os
from datetime import datetime

# create folder
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
# create log with the following namenusing time
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# config log
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger




