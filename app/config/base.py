from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ConfigDict, Field, field_validator

import os
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        protected_namespaces=()
    )
    
    # 项目根路径
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # 日志配置
    LOG_DIR: Path = ROOT_DIR / 'logs'
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 服务器配置
    SERVER_HOST: str = '0.0.0.0'
    SERVER_PORT: int = 5000

    # 缓存目录
    CACHE_DIR: Path = ROOT_DIR / 'cache'
    MODEL_CACHE_DIR: Path = CACHE_DIR / 'models'
    HF_CACHE_DIR: Path = CACHE_DIR / 'huggingface'



    # 添加缺失的配置字段
    flask_debug: bool = False
    flask_app: str = 'app'
    max_file_size: int = 52428800
    upload_folder: str = './app/static/uploads'
    allowed_extensions: List[str] = Field(default=['pdf', 'docx', 'txt'], description="允许上传的文件后缀列表")

    @field_validator('allowed_extensions', mode='before')
    def split_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(',')]
        if isinstance(v, list):
            return v
        raise ValueError(f"Invalid allowed_extensions format: {v}")
    database_url: str = 'sqlite:///app.db'
    vector_db_path: str = './static/vector_db'
    faiss_index_path: str = './static/outputs/faiss_index.bin'
    log_file: str = './logs/app.log'
    cache_type: str = 'simple'
    cache_default_timeout: int = 300
    api_version: str = 'v1'
    api_prefix: str = '/api/v1'
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    use_gpu: bool = False
    blockchain_paper_path: str = './static/papers'
    blockchain_model_path: str = './models'
    blockchain_index_path: str = './static/outputs/blockchain_index.bin'
    blockchain_embedding_dimension: int = 384
    pdf_chunk_size: int = 1000
    secret_key: str = 'your-dev-secret-key'
    pytorch_cuda_alloc_conf: Optional[str] = None
    transformers_cache: str = './cache/transformers'
    transformers_offline: bool = False
    hf_home: str = './cache/huggingface'
    hf_hub_download_timeout: int = 300



base_config = BaseConfig()