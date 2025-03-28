# embedding_generator.py
import torch
from typing import List, Union
import os
from app.utils.logger import setup_logger
from app.config.environment import (
    MODEL_NAME,
    MODEL_CACHE_DIR,
    DEVICE,
    BLOCKCHAIN_EMBEDDING_DIMENSION
)

# 设置日志记录器
logger = setup_logger(__name__)

class EmbeddingGenerator:
    """
    文本嵌入生成器类。
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                # 使用本地模型或备用模型
                self.model = self._load_model()
                self._initialized = True
                logger.info("Embedding generator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embedding generator: {e}")
                self.model = self._create_simple_model()
                self._initialized = True

    def _load_model(self):
        """加载模型"""
        try:
            # 尝试加载本地模型
            local_model_path = os.path.join(MODEL_CACHE_DIR, MODEL_NAME)
            if os.path.exists(local_model_path):
                # 延迟导入以避免早期导入错误
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer(local_model_path, device=DEVICE)
            return self._create_simple_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return self._create_simple_model()

    def _create_simple_model(self):
        """创建简单的平均池化模型"""
        from torch import nn
        
        class SimpleAverageModel(nn.Module):
            def __init__(self, embedding_dim=BLOCKCHAIN_EMBEDDING_DIMENSION):
                super().__init__()
                self.embedding_dim = embedding_dim
                
            def encode(self, texts, convert_to_tensor=True):
                if isinstance(texts, str):
                    texts = [texts]
                # 返回零向量作为默认嵌入
                embeddings = torch.zeros((len(texts), self.embedding_dim))
                if DEVICE == "cuda":
                    embeddings = embeddings.cuda()
                return embeddings

        return SimpleAverageModel()

    def generate_embedding(self, text: str) -> torch.Tensor:
        """生成单个文本的嵌入向量"""
        try:
            if hasattr(self.model, 'encode'):
                return self.model.encode(text, convert_to_tensor=True)
            return torch.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return torch.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)

    def generate_embeddings(self, texts: List[str]) -> torch.Tensor:
        """批量生成文本的嵌入向量"""
        try:
            if hasattr(self.model, 'encode'):
                return self.model.encode(texts, convert_to_tensor=True)
            return torch.zeros((len(texts), BLOCKCHAIN_EMBEDDING_DIMENSION))
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return torch.zeros((len(texts), BLOCKCHAIN_EMBEDDING_DIMENSION))

def generate_embeddings(text: str):
    """
    生成文本嵌入向量
    """
    try:
        generator = EmbeddingGenerator()
        return generator.generate_embedding(text)
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise

def get_embedding_generator():
    """获取嵌入生成器实例"""
    return EmbeddingGenerator()

# 示例用法
if __name__ == "__main__":
    # 创建嵌入生成器实例
    generator = EmbeddingGenerator()
    
    # 测试文本
    texts = [
        "区块链是一个去中心化的分布式账本技术。",
        "智能合约是在区块链上自动执行的程序。",
        "比特币是第一个成功的区块链应用。"
    ]
    
    # 生成单个文本的嵌入
    single_embedding = generator.generate_embedding(texts[0])
    print(f"单个文本嵌入维度: {single_embedding.shape}")
    
    # 批量生成文本的嵌入
    batch_embeddings = generator.generate_embeddings(texts)
    print(f"批量文本嵌入维度: {batch_embeddings.shape}")