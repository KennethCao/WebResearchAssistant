import os
import sys
import ssl
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_tiny_model():
    """Download the smallest BERT model"""
    # Disable SSL verification
    # Set environment variables
    # Set cache directory
    # Download the smallest BERT model
    # Download tokenizer
    # Download model
    # Save to local
    try:
        # 禁用 SSL 验证
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        
        # 设置环境变量
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'
        
        # 设置缓存目录
        cache_dir = os.path.join(os.getcwd(), 'data', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 下载最小的BERT模型
        model_name = 'prajjwal1/bert-tiny'
        logger.info(f"Downloading model: {model_name}")
        
        # 下载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        # 下载模型
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True
        )
        
        # 保存到本地
        save_dir = os.path.join(cache_dir, 'bert-tiny')
        os.makedirs(save_dir, exist_ok=True)
        
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        
        logger.info(f"Model saved to {save_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

if __name__ == "__main__":
    if download_tiny_model():
        logger.info("Model downloaded successfully")
        sys.exit(0)
    else:
        logger.error("Failed to download model")
        sys.exit(1)