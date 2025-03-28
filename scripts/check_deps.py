import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT_DIR)

import importlib
import logging

# 设置基本的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查关键依赖是否正确安装"""
    required_packages = [
        'huggingface_hub',
        'transformers',
        'sentence_transformers',
        'torch',
        'flask',
        'flask_sqlalchemy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} is installed")
        except ImportError as e:
            logger.error(f"✗ {package} is missing: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        return False
    return True

if __name__ == "__main__":
    try:
        logger.info(f"Checking dependencies from {ROOT_DIR}")
        if not check_dependencies():
            sys.exit(1)
        logger.info("All dependencies are installed correctly")
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        sys.exit(1) 