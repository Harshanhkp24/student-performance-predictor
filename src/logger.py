import logging 
import os 
from datetime import datetime 

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path=os.path.join(os.getcwd(),"logs")

os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

handlers = [logging.StreamHandler()]

try:
    handlers = [logging.FileHandler(LOG_FILE_PATH)]
except OSError:
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    handlers=handlers,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)

