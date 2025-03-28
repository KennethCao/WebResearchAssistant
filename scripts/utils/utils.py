import logging
import os
from pathlib import Path

def setup_script_logger(name):
    """Set up a simple logger for scripts"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger