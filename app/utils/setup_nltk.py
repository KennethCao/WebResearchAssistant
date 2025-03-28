import nltk
import os
from app.config.environment import NLTK_DATA_DIR

def setup_nltk():
    """下载并设置NLTK数据"""
    # 设置NLTK数据目录
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    nltk.data.path.append(str(NLTK_DATA_DIR))
    
    try:
        # 下载必要的NLTK数据
        nltk.download('punkt', download_dir=str(NLTK_DATA_DIR))
        nltk.download('stopwords', download_dir=str(NLTK_DATA_DIR))
        nltk.download('averaged_perceptron_tagger', download_dir=str(NLTK_DATA_DIR))
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        raise 