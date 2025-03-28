from app.config.base import BaseConfig
from pathlib import Path

class ConstantsConfig(BaseConfig):
    # 嵌入向量配置
    EMBEDDING_DIMENSION: int = 384
    FAISS_INDEX_PATH: Path = BaseConfig.CACHE_DIR / 'faiss_index.bin'
    
    # 文本处理配置
    SUMMARY_MAX_LENGTH: int = 200
    MAX_SEQUENCE_LENGTH: int = 512
    
    # 模型默认参数
    DEFAULT_BATCH_SIZE: int = 8
    DEFAULT_LEARNING_RATE: float = 2e-5

constants_config = ConstantsConfig()

# 保持原有导入兼容
__all__ = ['constants_config']

def __getattr__(name):
    return getattr(constants_config, name)