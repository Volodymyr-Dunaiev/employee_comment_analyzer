# Logger setup utilities.
# Provides functions to configure logging based on config.yaml settings.

import os
import yaml
import logging

def setup_logger(name):
    # Set up a logger with configuration from config.yaml.
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_config = config['logging']

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_config['level'])

        # File handler
        log_dir = os.path.dirname(log_config['file']) or "./logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_config['file'])
        file_handler.setFormatter(logging.Formatter(log_config['format']))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_config['format']))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Alias for backward compatibility
def get_logger(name):
    # Get or create a logger with the specified name.
    return setup_logger(name)
