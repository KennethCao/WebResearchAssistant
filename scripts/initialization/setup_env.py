import subprocess
import sys
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def install_dependencies():
    """安装和更新依赖包"""
    try:
        # 安装/更新特定版本的包
        packages = [
            'huggingface-hub==0.16.4',
            'sentence-transformers==2.0.0',
            'transformers==4.35.2'
        ]
        
        for package in packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', package])
            
        logger.info("Dependencies installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

if __name__ == "__main__":
    if install_dependencies():
        print("Environment setup completed successfully")
    else:
        print("Environment setup failed")
        sys.exit(1) 