import yaml
def config_loader(path = "config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

import logging
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)