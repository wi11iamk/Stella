"""Logging utility for Stella."""
import logging
from utils.config import get_config

def setup_logger():
    """Configures logging settings, including file output."""
    config = get_config()
    log_file = config.get("log_file", "stella.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler(log_file, mode='a')  # Logs to file
        ]
    )
    return logging.getLogger("stella")
