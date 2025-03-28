import os
from pathlib import Path
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def setup_directories():
    """设置必要的目录结构"""
    try:
        # 获取项目根目录
        root_dir = Path(__file__).parent.parent
        
        # 需要创建的目录
        directories = [
            15: 'uploads',           # Uploaded files directory
            16: 'instance',          # Instance directory (database)
            17: 'logs',             # Logs directory
            18: 'cache/models',      # Model cache
            19: 'cache/huggingface', # HuggingFace cache
            20: 'data/knowledge_base', # Knowledge base
            21: 'static/pdfs'        # PDF storage
        ]
        
        # 创建目录
        for dir_path in directories:
            full_path = root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup directories: {e}")
        return False

if __name__ == "__main__":
    setup_directories()