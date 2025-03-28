import os
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def create_directories():
    """Create necessary directory structure"""
    directories = [
        os.path.join(ROOT_DIR, 'app/static/pdfs'),           # PDF storage
        os.path.join(ROOT_DIR, 'app/static/knowledge_base'),  # Knowledge base
        os.path.join(ROOT_DIR, 'app/static/outputs'),         # Output directories
        os.path.join(ROOT_DIR, 'cache/models'),              # Model cache
        os.path.join(ROOT_DIR, 'cache/huggingface'),         # Hugging Face cache
        os.path.join(ROOT_DIR, 'logs')                       # Logs directory
    ]
    
    try:
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Creating directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        raise

if __name__ == "__main__":
    create_directories()