import os
from pathlib import Path
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def init_directories():
    """初始化必要的目录"""
    try:
        # 获取项目根目录
        root_dir = Path(__file__).parent.parent
        
        # 需要创建的目录
        directories = [
            'uploads',
            'cache',
            'logs',
            'data/knowledge_base'
        ]
        
        # 创建目录
        for dir_name in directories:
            dir_path = os.path.join(root_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize directories: {e}")
        return False

if __name__ == "__main__":
    init_directories() 