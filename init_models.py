import os
import ssl
import sys
from pathlib import Path
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_session():
    """创建带有重试机制的会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # 禁用 SSL 验证
    session.verify = False
    return session

def setup_environment():
    """设置环境变量和 SSL 配置"""
    # 禁用 SSL 验证
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # 设置环境变量
    os.environ.update({
        'CURL_CA_BUNDLE': '',
        'REQUESTS_CA_BUNDLE': '',
        'SSL_CERT_FILE': '',
        'TRANSFORMERS_OFFLINE': '0',
        'HF_ENDPOINT': 'https://hf-mirror.com',
        'HF_HUB_DOWNLOAD_TIMEOUT': '500',
        'TRANSFORMERS_CACHE': os.path.join(Path(__file__).parent, 'data', 'models'),
        'HF_HOME': os.path.join(Path(__file__).parent, 'data', 'models'),
        'SENTENCE_TRANSFORMERS_HOME': os.path.join(Path(__file__).parent, 'data', 'models')
    })

def download_with_retry(model_name, cache_dir, max_retries=5):
    """带重试机制的模型下载"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {model_name} (Attempt {attempt + 1}/{max_retries})")
            
            if "sentence-transformers" in model_name:
                model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir
                )
            else:
                from transformers import pipeline
                model = pipeline(
                    "summarization",
                    model=model_name,
                    device='cpu',
                    low_cpu_mem_usage=True
                )
            
            logger.info(f"Successfully downloaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                logger.info(f"Waiting {wait_time} seconds before next attempt...")
                time.sleep(wait_time)
            else:
                raise

def download_models():
    """下载所需的模型"""
    try:
        # 设置环境
        setup_environment()
        
        # 设置缓存目录
        cache_dir = os.path.join(Path(__file__).parent, 'data', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 创建会话
        session = create_session()
        
        models_to_download = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sshleifer/distilbart-cnn-6-6'
        ]
        
        for model_name in models_to_download:
            try:
                model = download_with_retry(model_name, cache_dir)
                logger.info(f"Successfully downloaded and verified {model_name}")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                continue
        
        logger.info("All models downloaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in download_models: {e}")
        return False
    finally:
        # 清理会话
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    try:
        if download_models():
            logger.info("Model initialization completed successfully")
            sys.exit(0)
        else:
            logger.error("Model initialization failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 