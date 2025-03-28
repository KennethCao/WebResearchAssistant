# summarizer.py
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration
)
import gc
import os
from app.utils.logger import setup_logger
from app.config.environment import MODEL_CACHE_DIR, DEVICE
import huggingface_hub
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import re
from typing import List, Optional

# 设置环境变量以限制内存使用
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['TORCH_DEVICE'] = 'cpu'

# 设置日志记录器
logger = setup_logger(__name__)

class TextSummarizer:
    """
    文本摘要生成器类。
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextSummarizer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            # 设置环境变量
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''
            os.environ['TRANSFORMERS_OFFLINE'] = '0'
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 设置设备
            self.device = torch.device('cpu')
            logger.info(f"Using device: {self.device}")
            
            # 使用更小的模型
            try:
                from transformers import pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-6-6",  # 使用更小的模型
                    device=self.device,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
            except Exception as e:
                logger.error(f"Failed to load summarizer model: {e}")
                self.summarizer = None
            
            self._initialized = True
            logger.info("Text summarizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {e}")
            self.summarizer = None
            self._initialized = True

    def generate_summary(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """生成文本摘要"""
        try:
            if self.summarizer is None:
                return self._simple_summary(text, max_length)
                
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return self._simple_summary(text, max_length)
    
    def _simple_summary(self, text: str, max_length: int = 150) -> str:
        """简单的摘要方法（作为后备方案）"""
        sentences = text.split('.')
        summary = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary.append(sentence)
            current_length += len(sentence)
            
        return '. '.join(summary) + '.'
    
    def _split_text(self, text: str, max_chunk_size: int = 500) -> list:
        """将长文本分成小块"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def generate_summaries(self, texts: list[str], **kwargs) -> list[str]:
        """批量生成文本摘要
        
        Args:
            texts (list[str]): 需要生成摘要的文本列表
            **kwargs: 传递给 generate_summary 的其他参数
            
        Returns:
            list[str]: 生成的摘要文本列表
        """
        try:
            summaries = []
            for text in texts:
                summary = self.generate_summary(text)
                summaries.append(summary)
            
            logger.info(f"Generated {len(summaries)} summaries successfully")
            return summaries
            
        except Exception as e:
            logger.error(f"Failed to generate summaries: {e}")
            raise

# 示例用法
if __name__ == "__main__":
    # 创建摘要生成器实例
    summarizer = TextSummarizer()
    
    # 测试文本
    text = """
    区块链是一个去中心化的分布式账本技术，它可以安全地记录交易和其他类型的数据。
    区块链通过密码学和共识机制来确保数据的不可篡改性和透明性。每个区块都包含多个交易记录，
    并通过哈希链接到前一个区块，形成一个不可篡改的链条。这种技术最初用于比特币，但现在已经
    扩展到许多其他领域，包括智能合约、供应链管理和数字身份认证等。
    """
    
    # 生成摘要
    summary = summarizer.generate_summary(text)
    print(f"原文: {text}\n")
    print(f"摘要: {summary}")

def generate_summary(text: str) -> str:
    """
    生成文本摘要
    """
    try:
        summarizer = TextSummarizer()
        summary = summarizer.generate_summary(text)
        return summary
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Failed to generate summary"  # 返回一个默认值而不是 None

def get_summarizer() -> Optional[TextSummarizer]:
    """获取摘要生成器实例"""
    try:
        return TextSummarizer()
    except Exception as e:
        logger.error(f"Failed to get summarizer: {e}")
        return None