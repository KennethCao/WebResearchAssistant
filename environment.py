import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.resolve()

class EnvConfig:
    def __init__(self):
        # 先加载环境变量
        self._load_environment()
        
        # 验证必需的环境变量
        self._validate_environment()
        
        # 初始化路径配置
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        self.DB_PATH = ROOT_DIR / 'instance' / 'database.db'
        self.LOG_DIR = ROOT_DIR / 'logs'
        self.MIGRATIONS_DIR = ROOT_DIR / 'migrations'

    def _load_environment(self):
        # 从.env文件加载环境变量
        env_path = ROOT_DIR / '.env'
        if env_path.exists():
            load_dotenv(env_path)
        
        # 设置默认开发环境变量
        os.environ.setdefault('FLASK_ENV', 'development')
        os.environ.setdefault('FLASK_DEBUG', '1')

    def _validate_environment(self):
        required_vars = ['SECRET_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logging.error(error_msg)
            raise EnvironmentError(error_msg)

# 在模块加载时立即初始化配置
env_config = EnvConfig()