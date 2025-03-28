# app/config/__init__.py
import os
from dotenv import load_dotenv

# 加载环境变量
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, '../../.env')
load_dotenv(ENV_PATH)

from .config import ConfigLoader

def get_config():
    """
    根据环境变量返回适配的配置类实例。
    """
    env = os.getenv('FLASK_DEBUG', '1')

    if env == 'development':
        from app.config.settings import DevelopmentConfig
        config = DevelopmentConfig()
        config.USE_GPU = False  # 强制禁用 GPU
        return config
    elif env == 'production':
        from app.config.settings import ProductionConfig
        config = ProductionConfig()
        config.USE_GPU = False  # 强制禁用 GPU
        return config
    elif env == 'testing':
        from app.config.settings import TestingConfig
        config = TestingConfig()
        config.USE_GPU = False  # 强制禁用 GPU
        return config
    else:
        raise ValueError(f"无效的 FLASK_DEBUG 值: {env}")

def validate_config(config):
    """
    验证配置的有效性。
    :param config: 配置类实例
    :return: 验证结果
    """
    try:
        from app.config.validator import validate_config as validate
        if validate(config):
            return True
        else:
            raise ValueError("配置验证失败")
    except Exception as e:
        raise RuntimeError(f"配置验证失败: {str(e)}")

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