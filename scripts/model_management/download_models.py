import os
import sys
import ssl
from pathlib import Path
import requests
import torch
from tqdm import tqdm
import time

# Add project root to Python path
ROOT_DIR = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT_DIR)

# Set environment variables
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Create necessary directories
CACHE_DIR = os.path.join(ROOT_DIR, 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')
HF_CACHE_DIR = os.path.join(CACHE_DIR, 'huggingface')

# Ensure directory exists
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Set environment variables
os.environ['HF_HOME'] = HF_CACHE_DIR
if 'TRANSFORMERS_CACHE' in os.environ:
    del os.environ['TRANSFORMERS_CACHE']

# Configure SSL context
ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()

import nltk
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def setup_nltk():
    """设置NLTK数据"""
    try:
        # 设置NLTK数据目录
        nltk_data_dir = os.path.join(CACHE_DIR, 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # 下载必要的NLTK数据
        for package in ['punkt', 'averaged_perceptron_tagger', 'universal_tagset']:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Downloaded NLTK package: {package}")
            except Exception as e:
                logger.error(f"Failed to download NLTK package {package}: {e}")
                raise
                
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to setup NLTK: {e}")
        raise

def download_with_progress(model_name, operation, download_fn):
    """带进度条的下载函数"""
    logger.info(f"Starting {operation} {model_name}")
    start_time = time.time()
    
    try:
        with tqdm(desc=f"{operation} {model_name}", 
                 unit='B', unit_scale=True, 
                 miniters=1) as pbar:
            result = download_fn()
            pbar.update(1)
    except Exception as e:
        logger.error(f"{operation} failed: {str(e)}")
        raise
    
    duration = time.time() - start_time
    logger.info(f"{operation} completed in {duration:.2f} seconds")
    return result

def download_models():
    """下载所需的模型"""
    try:
        # 设置缓存目录
        cache_dir = os.path.join(ROOT_DIR, 'cache', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set environment variables
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = cache_dir
        
        # 下载 Sentence Transformer 模型
        logger.info("Downloading Sentence Transformer model...")
        model_names = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/distiluse-base-multilingual-cased-v1'
        ]
        
        for model_name in model_names:
            try:
                logger.info(f"下载模型: {model_name}")
                model = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir
                )
                logger.info(f"Successfully downloaded model: {model_name}")
            except Exception as e:
                logger.error(f"下载模型 {model_name} 失败: {e}")
                continue
        
        # 下载 BART 模型
        logger.info("下载 BART 模型...")
        try:
            model_name = "facebook/bart-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
            logger.info(f"Successfully downloaded model: {model_name}")
        except Exception as e:
            logger.error(f"下载 BART 模型失败: {e}")
        
        logger.info("所有模型下载完成")
        return True
        
    except Exception as e:
        logger.error(f"下载模型时出错: {e}")
        return False

if __name__ == "__main__":
    try:
        # 下载模型
        if download_models():
            logger.info("模型下载成功")
            sys.exit(0)
        else:
            logger.error("模型下载失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断下载")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {e}")
        sys.exit(1)