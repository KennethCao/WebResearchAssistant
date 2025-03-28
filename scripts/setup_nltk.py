import nltk
import ssl
import os
from app.utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

def setup_nltk():
    """Initialize NLTK data"""
    try:
        # Set NLTK data directory
        nltk_data_dir = os.path.join(ROOT_DIR, 'data', 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK datasets
        packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
        for package in packages:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Successfully downloaded {package}")
            except Exception as e:
                logger.error(f"Failed to download NLTK package {package}: {e}")
                return False
        
        # 验证下载
        try:
            from nltk.corpus import stopwords
            stopwords.words('english')
            logger.info("Stopwords test passed")
        except Exception as e:
            logger.error(f"Stopwords test failed: {e}")
            return False
            
        logger.info("NLTK数据验证成功")
        return True
        
    except Exception as e:
        logger.error(f"NLTK设置失败: {e}")
        return False

if __name__ == "__main__":
    # 先清理现有的NLTK数据（如果需要重新下载）
    nltk_data_dir = os.path.join(Path(__file__).parent.parent, 'data', 'nltk_data')
    if os.path.exists(nltk_data_dir):
        import shutil
        shutil.rmtree(nltk_data_dir)
        logger.info(f"已清理旧的NLTK数据目录: {nltk_data_dir}")
    
    # 重新下载和设置
    if setup_nltk():
        logger.info("NLTK设置成功")
    else:
        logger.error("NLTK设置失败")