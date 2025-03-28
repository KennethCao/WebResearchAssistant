import re
from typing import List, Optional
from app.utils.logger import setup_logger
import nltk

logger = setup_logger(__name__)

class TextProcessor:
    """文本处理器类"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        try:
            # 初始化 NLTK 资源
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            self._initialized = True
            logger.info("Text processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TextProcessor: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        清理文本
        - 移除特殊字符
        - 标准化空白字符
        - 移除多余的空格
        """
        try:
            # 替换换行符为空格
            text = text.replace('\n', ' ')
            
            # 移除特殊字符，保留字母、数字、空格和基本标点
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            
            # 标准化空白字符
            text = ' '.join(text.split())
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text

    def split_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子
        """
        try:
            sentences = nltk.sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Error splitting sentences: {e}")
            return [text]

    def tokenize(self, text: str) -> List[str]:
        """
        将文本分词
        """
        try:
            return nltk.word_tokenize(text)
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return text.split()

    def remove_stopwords(self, text: str) -> str:
        """
        移除停用词
        """
        try:
            stopwords = set(nltk.corpus.stopwords.words('english'))
            words = self.tokenize(text)
            filtered_words = [word for word in words if word.lower() not in stopwords]
            return ' '.join(filtered_words)
        except Exception as e:
            logger.error(f"Error removing stopwords: {e}")
            return text

    def preprocess(self, text: str, remove_stops: bool = True) -> str:
        """
        完整的文本预处理流程
        """
        try:
            # 清理文本
            text = self.clean_text(text)
            
            # 移除停用词（可选）
            if remove_stops:
                text = self.remove_stopwords(text)
                
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text

# 示例用法
if __name__ == "__main__":
    processor = TextProcessor()
    
    test_text = """
    This is a test text! It contains some basic
    punctuation and    multiple   spaces...
    
    Let's see how it handles preprocessing.
    """
    
    print("Original text:")
    print(test_text)
    print("\nPreprocessed text:")
    print(processor.preprocess(test_text)) 