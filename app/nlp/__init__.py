# __init__.py
import warnings
from app.utils.logger import setup_logger
from app.nlp.summarizer import TextSummarizer
from app.nlp.embedding_generator import EmbeddingGenerator
import nltk
import os
import ssl
from .analyzer import analyze_paper
from app.nlp.keyword_extractor import extract_keywords
from app.nlp.text_processor import TextProcessor

# 设置日志记录器
logger = setup_logger(__name__)

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA
os.environ['TORCH_DEVICE'] = 'cpu'

# 初始化组件
summarizer = None
embedding_generator = None

def download_nltk_data():
    """下载 NLTK 数据"""
    try:
        # 处理 SSL 证书问题
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # 设置 NLTK 数据目录
        nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # 设置 NLTK 数据路径
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        
        # 直接下载数据包
        packages = ['punkt', 'stopwords']
        for package in packages:
            try:
                nltk.download(package, download_dir=nltk_data_dir, quiet=True)
                logger.info(f"Successfully downloaded {package}")
            except Exception as e:
                logger.error(f"Error downloading {package}: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise

def initialize_nltk():
    """初始化 NLTK"""
    try:
        # 检查是否已经下载了所需数据
        nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'nltk_data')
        required_data = ['tokenizers/punkt', 'corpora/stopwords']
        
        # 检查每个必需的数据包
        missing_data = []
        for data in required_data:
            try:
                nltk.data.find(data)
            except LookupError:
                missing_data.append(data.split('/')[-1])
        
        # 如果有缺失的数据包，下载它们
        if missing_data:
            logger.info(f"Downloading missing NLTK data: {missing_data}")
            download_nltk_data()
        
        # 设置 NLTK 数据路径
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
            
        logger.info("NLTK initialization completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize NLTK: {e}")
        raise

def initialize_nlp():
    """初始化所有 NLP 组件"""
    global summarizer, embedding_generator
    
    try:
        # 初始化 NLTK（只在需要时初始化）
        if summarizer is None or embedding_generator is None:
            initialize_nltk()
            
            # 初始化组件
            if summarizer is None:
                summarizer = TextSummarizer()
                logger.info("Text summarizer initialized")
            
            if embedding_generator is None:
                embedding_generator = EmbeddingGenerator()
                logger.info("Embedding generator initialized")
            
            logger.info("NLP components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize NLP components: {e}")
        raise

def get_summarizer() -> TextSummarizer:
    """
    获取文本摘要生成器实例。
    """
    global summarizer
    if summarizer is None:
        initialize_nlp()
    if summarizer is None:
        raise RuntimeError("Failed to initialize summarizer")
    return summarizer

def get_embedding_generator():
    """
    获取文本嵌入生成器实例。
    """
    global embedding_generator
    if embedding_generator is None:
        initialize_nlp()
    if embedding_generator is None:
        raise RuntimeError("Failed to initialize embedding generator")
    return embedding_generator

def load_default_model():
    """
    加载默认模型
    """
    try:
        initialize_nlp()
        logger.info("Default models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load default models: {e}")
        raise

def get_model():
    """
    获取当前加载的模型
    """
    try:
        return {
            'summarizer': get_summarizer(),
            'embedding_generator': get_embedding_generator()
        }
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise

def init_models():
    """初始化模型"""
    global summarizer, embedding_generator
    try:
        summarizer = TextSummarizer()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise

# 导出函数
__all__ = ['analyze_paper', 'extract_keywords', 'TextProcessor', 'TextSummarizer', 'get_embedding_generator'] 