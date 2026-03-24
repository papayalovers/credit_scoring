import logging
import os 
from utils.utils import BASE_DIR
import copy
import json 

# Apps logger
def setup_logger():

    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    # format log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File for all logs
    file_handler = logging.FileHandler(f'{log_dir}/logs.log', mode='w')
    file_handler.setLevel(logging.INFO)

    class NoErrorFilter(logging.Filter):
        def filter(self, record):
            return record.levelno < logging.ERROR

    file_handler.addFilter(NoErrorFilter())
    file_handler.setFormatter(formatter)
    
    # File for error logs
    error_file_handler = logging.FileHandler(f'{log_dir}/error.log', mode='w')
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(error_file_handler)  

    return logger

# Training model logger
def _update_training_log(current_log, path_log):
    current_log = copy.deepcopy(current_log)

    # If training log exist
    try:
        with open(path_log, 'r') as file:
            last_log = json.load(file)
    # if training log not exist
    except FileNotFoundError as err:
        with open(path_log, 'w') as file:
            file.write('[]')
        file.close()

        # Reload training log that we knows as last log
        with open(path_log, 'r') as file:
            last_log = json.load(file)

    # Add the current log to last log
    last_log.append(current_log)

    # Rewrite the training log with the updated one
    with open(path_log, 'w') as file:
        json.dump(last_log, file, indent=4)
        file.close()

    return last_log

    