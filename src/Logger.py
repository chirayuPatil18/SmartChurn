import logging ##Python’s built-in logging library.
import os ##for handling file paths and directories.
from datetime import datetime ##to generate unique timestamps for log files.

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ##f"....log" - creates a filename ending in .log. Ex-09_09_2025_14_45_30.log
logs_path = os.path.join(os.getcwd(),"logs")
#os.getcwd() - current working directory.
#os.path.join() - joins parts into a valid path.
#"logs" - puts all logs in a logs/ folder.
os.makedirs(logs_path,exist_ok = True) 
##os.makedirs is trying to create a folder with the full path. exist_ok=True - if folder already exists, don’t throw an error.

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE) ## add's LOG_FILE into logs_path

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO

)
#filename=LOG_FILE_PATH - writes logs into that file.
#format= - defines log structure:
#%(asctime)s - time log created
#%(lineno)d - line number in code
#%(name)s - logger name
#%(levelname)s - type of log (INFO, ERROR, etc.)
#%(message)s - actual log message
#level=logging.INFO - only logs INFO and above (not DEBUG).

