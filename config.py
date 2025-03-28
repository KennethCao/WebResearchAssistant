from typing import List
from pydantic import BaseSettings, validator, Field
from pathlib import Path
from pydantic import field_validator

class BaseConfig(BaseSettings):
    ROOT_DIR: Path = Path(__file__).parent.parent
    
    flask_debug: bool = Field(False, env='FLASK_DEBUG')
    flask_app: str = Field('app', env='FLASK_APP')
    max_file_size: int = Field(52428800, env='MAX_FILE_SIZE')
    upload_folder: Path = Field(ROOT_DIR/'app/static/uploads', env='UPLOAD_FOLDER')
    allowed_extensions: List[str] = Field(..., env='ALLOWED_EXTENSIONS')

    @field_validator('allowed_extensions', mode='before')
    def validate_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(',')]
        return v
    
    database_url: str = Field('sqlite:///app.db', env='DATABASE_URL')
    vector_db_path: Path = Field(ROOT_DIR/'static/vector_db', env='VECTOR_DB_PATH')
    faiss_index_path: Path = Field(ROOT_DIR/'static/outputs/faiss_index.bin', env='FAISS_INDEX_PATH')
    log_file: Path = Field(ROOT_DIR/'logs/app.log', env='LOG_FILE')
    cache_type: str = Field('simple', env='CACHE_TYPE')
    cache_default_timeout: int = Field(300, env='CACHE_DEFAULT_TIMEOUT')
    api_version: str = Field('v1', env='API_VERSION')
    api_prefix: str = Field('/api/v1', env='API_PREFIX')
    model_name: str = Field('sentence-transformers/all-MiniLM-L6-v2', env='MODEL_NAME')
    use_gpu: bool = Field(False, env='USE_GPU')
    blockchain_paper_path: Path = Field(ROOT_DIR/'static/papers', env='BLOCKCHAIN_PAPER_PATH')
    blockchain_model_path: Path = Field(ROOT_DIR/'models', env='BLOCKCHAIN_MODEL_PATH')
    blockchain_index_path: Path = Field(ROOT_DIR/'static/outputs/blockchain_index.bin', env='BLOCKCHAIN_INDEX_PATH')
    blockchain_embedding_dimension: int = Field(384, env='BLOCKCHAIN_EMBEDDING_DIMENSION')
    pdf_chunk_size: int = Field(1000, env='PDF_CHUNK_SIZE')
    secret_key: str = Field('your-dev-secret-key', env='SECRET_KEY')
    pytorch_cuda_alloc_conf: str = Field('max_split_size_mb:128', env='PYTORCH_CUDA_ALLOC_CONF')
    transformers_cache: Path = Field(ROOT_DIR/'cache/transformers', env='TRANSFORMERS_CACHE')
    transformers_offline: bool = Field(False, env='TRANSFORMERS_OFFLINE')
    hf_home: Path = Field(ROOT_DIR/'cache/huggingface', env='HF_HOME')
    hf_hub_download_timeout: int = Field(300, env='HF_HUB_DOWNLOAD_TIMEOUT')

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

config = BaseConfig()