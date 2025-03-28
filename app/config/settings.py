import os
from dotenv import load_dotenv

# 定义 .env 文件的路径
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")

# 加载 .env 文件中的环境变量
if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH)
else:
    print(f"警告: 未找到 .env 文件, 路径: {ENV_FILE_PATH}")


class Config:
    """
    基础配置类
    """
    # 通用配置
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')  # 默认密钥
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv('UPLOAD_FOLDER', 'static/pdfs'))  # 文件上传目录
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, os.getenv('FAISS_INDEX_PATH', 'static/outputs/faiss_index.bin'))  # FAISS 索引文件路径
    SQLITE_DB_PATH = os.path.join(BASE_DIR, os.getenv('SQLITE_DB_PATH', 'app/db.sqlite3'))  # SQLite 数据库路径
    LOG_FILE_PATH = os.path.join(BASE_DIR, os.getenv('LOG_FILE_PATH', 'logs/app.log'))  # 日志文件路径

    # 通用功能配置
    DEBUG = False  # 是否开启调试模式
    TESTING = False  # 是否开启测试模式
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # 禁用 SQLAlchemy 的事件系统（提高性能）

    # 新增 GPU 使用的环境变量标志，默认禁用 GPU
    USE_GPU = os.getenv('USE_GPU', 'False').lower() == 'true'

    # Flask 默认端口配置
    DEFAULT_PORT = int(os.getenv('DEFAULT_PORT', 5000))


class DevelopmentConfig(Config):
    """
    开发环境配置
    """
    DEBUG = True  # 开启调试模式
    TESTING = False  # 关闭测试模式


class ProductionConfig(Config):
    """
    生产环境配置
    """
    DEBUG = False  # 关闭调试模式
    TESTING = False  # 关闭测试模式


class TestingConfig(Config):
    """
    测试环境配置
    """
    DEBUG = True  # 开启调试模式
    TESTING = True  # 开启测试模式
    # 测试环境使用内存数据库（如 SQLite 的内存模式）
    SQLITE_DB_PATH = ':memory:'