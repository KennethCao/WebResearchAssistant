import os
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 基础配置
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 服务器配置
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000

# 缓存目录配置
HF_CACHE_DIR = os.path.join(ROOT_DIR, 'cache', 'huggingface')
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, 'cache', 'models') 