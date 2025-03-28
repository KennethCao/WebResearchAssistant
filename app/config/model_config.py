from app.config.base import BaseConfig
from pydantic import Field
from datetime import datetime
from pathlib import Path

from pydantic import field_validator

class ModelTrainingConfig(BaseConfig):
    BASE_MODEL: str = Field('sentence-transformers/all-MiniLM-L6-v2')
    BATCH_SIZE: int = Field(8)
    NUM_EPOCHS: int = Field(3)
    LEARNING_RATE: float = Field(2e-5)
    MODEL_SAVE_DIR: Path = BaseConfig.MODEL_DIR / 'fine_tuned'
    CHECKPOINT_DIR: Path = BaseConfig.MODEL_DIR / 'checkpoints'
    
    @field_validator('MODEL_SAVE_DIR', 'CHECKPOINT_DIR', mode='before')
    def create_model_dirs(cls, value):
        value.mkdir(parents=True, exist_ok=True)
        return value

# 修改类名为 DataConfigNew 以避免重名
class DataConfigNew(BaseConfig):
    CLEANING_PIPELINE: list = Field(['remove_special_chars', 'normalize_whitespace'])
    MIN_SAMPLES_FOR_UPDATE: int = Field(100)

training_config = ModelTrainingConfig()
# 修改为使用 DataConfigNew 类，因为 DataConfig 未定义
data_config = DataConfigNew()
    
    # 版本管理
    @staticmethod
    def get_model_version():
        return f"v{datetime.now().strftime('%Y%m%d%H%M')}"

# 数据预处理配置
class DataConfig:
    CLEANING_PIPELINE = [
        'remove_special_chars',
        'normalize_whitespace',
        'fix_unicode'
    ]
    
    BLOCKCHAIN_SPECIFIC_TERMS = [
        'smart contract', 'consensus mechanism',
        'distributed ledger', 'hash function'
    ]



# 优化器配置
OPTIMIZER_CONFIG = {
    'weight_decay': 0.01,
    'adam_epsilon': 1e-6,
    'max_grad_norm': 1.0
}