import os
from app.utils.logger import setup_logger
from scripts.init_db import init_database
from scripts.init_directories import init_directories

logger = setup_logger(__name__)

def initialize_all():
    """初始化所有组件"""
    try:
        # 1. 初始化目录
        logger.info("Initializing directories...")
        init_directories()
        
        # 2. 初始化数据库
        logger.info("Initializing database...")
        init_database()
        
        logger.info("All components initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False

if __name__ == "__main__":
    initialize_all() 