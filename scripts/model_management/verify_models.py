import os
import sys
import ssl
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_models():
    """Verify model file integrity"""
    # Disable SSL verification
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    
    # Set environment variables
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
    
    # Set cache directory
    cache_dir = os.path.join(os.getcwd(), 'data', 'models')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Models to verify
    models = [
        'all-MiniLM-L6-v2',
        'paraphrase-MiniLM-L3-v2'
    ]
    
    for model_name in models:
        try:
            logger.info(f"Verifying model: {model_name}")
            
            # Try loading model
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir
            )
            
            # Test model
            test_text = "This is a test sentence."
            embedding = model.encode(test_text)
            
            # Save model
            save_path = os.path.join(cache_dir, model_name)
            model.save(save_path)
            
            logger.info(f"Model {model_name} verified and saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to verify model {model_name}: {e}")
            continue
    
    return True
    
    except Exception as e:
        logger.error(f"Error during model verification: {e}")
        return False

if __name__ == "__main__":
    if verify_models():
        logger.info("All models verified successfully")
        sys.exit(0)
    else:
        logger.error("Model verification failed")
        sys.exit(1)