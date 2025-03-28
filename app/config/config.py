# config/config.py
import os
from app.config.environment import (
    MODEL_CACHE_DIR,
    MODEL_NAME,
    MODEL_DIMENSION,
    UPLOAD_FOLDER,
    FAISS_INDEX_PATH,
    MAX_FILE_SIZE
)
from app.config.constants import ALLOWED_EXTENSIONS, EMBEDDING_DIMENSION, FAISS_INDEX_FILE, UPLOAD_FOLDER as CONSTANTS_UPLOAD_FOLDER, OUTPUT_FOLDER, LOG_FILE_PATH, SUMMARY_MAX_LENGTH, KEYWORDS_MAX_COUNT, KMEANS_NUM_CLUSTERS, TOPIC_NETWORK_FILE, SQLITE_DB_PATH, MODEL_PATH, FINE_TUNED_MODEL_PATH, USE_GPU, API_VERSION, DEFAULT_PORT
from app.config.settings import Config, DevelopmentConfig, ProductionConfig, TestingConfig
from app.config.validator import validate_config

class BaseConfig:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    UPLOAD_FOLDER = UPLOAD_FOLDER
    FAISS_INDEX_PATH = FAISS_INDEX_PATH
    SQLITE_DB_PATH = SQLITE_DB_PATH
    LOG_FILE_PATH = LOG_FILE_PATH
    DEFAULT_PORT = DEFAULT_PORT
    USE_GPU = USE_GPU
    MODEL_NAME = MODEL_NAME
    MODEL_DIMENSION = MODEL_DIMENSION
    MAX_FILE_SIZE = MAX_FILE_SIZE
    ALLOWED_EXTENSIONS = ALLOWED_EXTENSIONS
    EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
    FAISS_INDEX_FILE = FAISS_INDEX_FILE
    CONSTANTS_UPLOAD_FOLDER = CONSTANTS_UPLOAD_FOLDER
    OUTPUT_FOLDER = OUTPUT_FOLDER
    SUMMARY_MAX_LENGTH = SUMMARY_MAX_LENGTH
    KEYWORDS_MAX_COUNT = KEYWORDS_MAX_COUNT
    KMEANS_NUM_CLUSTERS = KMEANS_NUM_CLUSTERS
    TOPIC_NETWORK_FILE = TOPIC_NETWORK_FILE
    MODEL_PATH = MODEL_PATH
    FINE_TUNED_MODEL_PATH = FINE_TUNED_MODEL_PATH
    API_VERSION = API_VERSION
    UPLOAD_FOLDER = os.path.join(os.getenv('BLOCKCHAIN_ASSISTANT_DATA_DIR', 'data'), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = False

class ProductionConfig(BaseConfig):
    DEBUG = False
    TESTING = False

class TestingConfig(BaseConfig):
    DEBUG = True
    TESTING = True
    SQLITE_DB_PATH = ':memory:'

def get_config():
    """
    根据环境变量返回适配的配置类实例。
    """
    env = os.getenv('FLASK_DEBUG', 'development')

    if env == 'development':
        return DevelopmentConfig()
    elif env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        raise ValueError(f"未知的 FLASK_DEBUG 值: {env}")

def initialize_config():
    """
    初始化配置并验证。
    :return: 配置类实例
    """
    try:
        config = get_config()
        if validate_config(config):
            return config
        else:
            raise ValueError("配置验证失败")
    except Exception as e:
        raise RuntimeError(f"配置初始化失败: {str(e)}")

class ConfigLoader:
    def __init__(self):
        self.config = initialize_config()

    def load_config(self):
        return self.config