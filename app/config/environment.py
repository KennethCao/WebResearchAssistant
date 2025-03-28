from pydantic import Field, AnyUrl, field_validator
from app.config.base import BaseConfig
import os
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# 从环境配置实例获取路径配置
db_dir = os.getenv('DB_DIR', ROOT_DIR / 'instance')
MODEL_CACHE_DIR = Path(db_dir) / 'models'
LOG_DIR = Path(os.getenv('LOG_DIR', ROOT_DIR / 'logs'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class EnvironmentConfig(BaseConfig):
    # 服务器配置
    server_host: str = Field('localhost', env='SERVER_HOST')
    server_port: int = Field(5000, env='SERVER_PORT')
    log_level: str = Field('INFO', env='LOG_LEVEL')
    model_cache_dir: Path = Field(ROOT_DIR / 'models', env='MODEL_CACHE_DIR')

    # Flask基础配置
    # 修复 Field 类型标注，确保与布尔类型兼容
    FLASK_DEBUG: bool = Field(default_factory=lambda: bool(os.getenv('FLASK_DEBUG', True)), env='FLASK_DEBUG')
    # 数据库连接配置
    SECRET_KEY: str = Field(default='dev-secret-key', env='SECRET_KEY')
    
    # 路径配置
    DATABASE_URL: AnyUrl = Field(
        default='sqlite:///' + str(ROOT_DIR / 'instance' / 'app.db'),
        env='DATABASE_URL'
    )
    CACHE_DIR: Path = ROOT_DIR / 'app' / 'cache'
    DATA_DIR: Path = ROOT_DIR / 'data'
    UPLOAD_DIR: Path = ROOT_DIR / 'uploads'
    MODEL_DIR: Path = ROOT_DIR / 'models'
    NLTK_DATA_DIR: Path = DATA_DIR / 'nltk_data'
    DB_PATH: Path = ROOT_DIR / 'instance' / 'app.db'

    # 设备配置
    DEVICE: str = Field('cpu', env='DEVICE')
    MODEL_DEVICE: str = Field('cpu', env='MODEL_DEVICE')

    # 文件上传配置
    MAX_CONTENT_MB: int = Field(16, env='MAX_CONTENT_MB')
    ALLOWED_EXTENSIONS: list = ['pdf', 'docx', 'txt']

    @field_validator('CACHE_DIR', 'DATA_DIR', 'UPLOAD_DIR', 'MODEL_DIR', mode='before')
    def validate_paths(cls, value):
        return ROOT_DIR / value

    @field_validator('*', mode='before')
    def create_dirs(cls, value, info):
        if info.field_name and info.field_name.endswith(('_DIR', '_PATH')) and isinstance(value, Path):
            value.mkdir(parents=True, exist_ok=True)
        return value

    model_config = {
        'env_prefix': 'BLOCKCHAIN_',
        'protected_namespaces': ('settings_',)
    }

env_config = EnvironmentConfig()

# 保持原有全局变量访问方式
__all__ = ['env_config']

def __getattr__(name):
    return getattr(env_config, name)

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_DEVICE'] = env_config.DEVICE

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置后添加 PDF 处理配置
# ... (保持现有配置不变)

# PDF 处理配置
PDF_CHUNK_SIZE = int(os.getenv('PDF_CHUNK_SIZE', 1000))  # 每个文本块的大小
PDF_OVERLAP = int(os.getenv('PDF_OVERLAP', 200))  # 文本块之间的重叠大小
PDF_MAX_PAGES = int(os.getenv('PDF_MAX_PAGES', 100))  # 处理的最大页数
PDF_ENCODING = 'utf-8'  # PDF 文本编码
PDF_EXTRACTION_TIMEOUT = 300  # PDF 文本提取超时时间（秒）

# PDF 解析设置
PDF_PARSER_SETTINGS = {
    'line_margin': 0.3,  # 行间距阈值
    'char_margin': 2.0,  # 字符间距阈值
    'word_margin': 0.1,  # 单词间距阈值
    'boxes_flow': 0.5,   # 文本框流向阈值
    'detect_vertical': True,  # 检测垂直文本
    'all_texts': True    # 提取所有文本
}

# PDF 文本清理设置
PDF_CLEANING_SETTINGS = {
    'remove_urls': True,          # 移除 URL
    'remove_emails': True,        # 移除邮箱
    'remove_numbers': False,      # 保留数字
    'remove_punctuation': False,  # 保留标点
    'remove_whitespace': True,    # 移除多余空白
    'normalize_unicode': True,    # Unicode 标准化
    'min_word_length': 2,        # 最小词长度
    'max_word_length': 100       # 最大词长度
}

# 确保必要的目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'data', 'knowledge_base'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'cache', 'models'), exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 其他配置...

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'
# 从 EnvironmentConfig 类中获取 CACHE_DIR
# 使用本地env_config实例
HF_CACHE_DIR = os.path.join(ROOT_DIR / 'app' / 'cache', 'huggingface')
FAISS_INDEX_PATH = ROOT_DIR / 'cache' / 'faiss' / 'index.faiss'

# 确保缓存目录存在
for cache_dir in [MODEL_CACHE_DIR, HF_CACHE_DIR, FAISS_INDEX_PATH]:
    os.makedirs(cache_dir, exist_ok=True)

# 设置 HuggingFace 缓存目录
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_DIR)
os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = MODEL_CACHE_DIR
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# 日志配置
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = ROOT_DIR / 'logs' / 'app.log'

# API 配置
API_PREFIX = '/api/v1'

# 打印关键配置信息
if FLASK_DEBUG == '1':
    print(f"DEVICE: {env_config.DEVICE}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"API_PREFIX: {API_PREFIX}")

# Celery settings
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')

# 新增日志目录配置
LOG_DIR = ROOT_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TIMEZONE = 'UTC'
CELERY_ENABLE_UTC = True

# 在现有配置中添加或修改
MAX_BATCH_SIZE = 32  # 限制批处理大小
TORCH_DTYPE = 'float32'  # 使用 float32 而不是 float64

# 内存限制
import torch
torch.set_num_threads(1)  # 限制线程数
torch.backends.cudnn.enabled = False  # 禁用 cuDNN

# CORS 配置
CORS_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.1.43:5000"  # 添加你的 IP
]

CORS_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
CORS_ALLOW_HEADERS = ['Content-Type']
CORS_SUPPORTS_CREDENTIALS = True

# 从配置类获取设置
FLASK_DEBUG = str(env_config.FLASK_DEBUG)
SECRET_KEY = env_config.SECRET_KEY
SERVER_HOST = env_config.server_host
SERVER_PORT = env_config.server_port

# 数据库配置
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URI = f'sqlite:///{os.path.join(DB_DIR, "app.db")}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 模型配置
MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用更小的模型
# 假设 ROOT_DIR 就是我们需要的基础目录
MODEL_CACHE_DIR = env_config.model_cache_dir
MODEL_PATH = MODEL_CACHE_DIR / MODEL_NAME
MODEL_BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', 4))
MODEL_MAX_LENGTH = int(os.getenv('MODEL_MAX_LENGTH', 512))

# 确保模型缓存目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
BLOCKCHAIN_EMBEDDING_DIMENSION = 384  # 减小嵌入维度
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
EMBEDDING_MAX_LENGTH = int(os.getenv('EMBEDDING_MAX_LENGTH', 512))

# 摘要模型配置
SUMMARY_MODEL = os.getenv('SUMMARY_MODEL', 'facebook/bart-large-cnn')
SUMMARY_MAX_LENGTH = int(os.getenv('SUMMARY_MAX_LENGTH', 1024))
SUMMARY_MIN_LENGTH = int(os.getenv('SUMMARY_MIN_LENGTH', 50))

# 聚类配置
CLUSTERING_N_CLUSTERS = int(os.getenv('CLUSTERING_N_CLUSTERS', 5))
CLUSTERING_RANDOM_STATE = int(os.getenv('CLUSTERING_RANDOM_STATE', 42))

# 向量检索配置
VECTOR_SIMILARITY_METRIC = 'cosine'
VECTOR_TOPK = int(os.getenv('VECTOR_TOPK', 5))

# CPU 优化配置
import multiprocessing
CPU_THREADS = multiprocessing.cpu_count()
BATCH_SIZE = 4  # 较小的批处理大小
MAX_LENGTH = 512  # 限制文本长度

# 缓存配置
CACHE_TYPE = 'simple'