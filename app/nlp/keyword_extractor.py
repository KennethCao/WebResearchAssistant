# keyword_extractor.py
from rake_nltk import Rake
from app.utils.logger import setup_logger
import nltk
from typing import List, Optional
import torch
from collections import Counter
import re

# 设置日志记录器
logger = setup_logger(__name__)

class KeywordExtractor:
    """关键词提取器类"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KeywordExtractor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        try:
            # 初始化 RAKE
            self.rake = Rake(
                min_length=2,  # 最小词长度
                max_length=4,   # 最大词长度
                ranking_metric='degree'
            )
            
            self._initialized = True
            logger.info("Keyword extractor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize KeywordExtractor: {e}")
            raise

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_n: 返回的关键词数量
            
        Returns:
            关键词列表
        """
        try:
            # 如果文本为空，返回空列表
            if not text or not text.strip():
                return []
                
            # 使用 RAKE 提取关键词
            self.rake.extract_keywords_from_text(text)
            rake_keywords = self.rake.get_ranked_phrases()[:top_n*2]
            
            # 使用词频统计补充关键词
            freq_keywords = self._extract_by_frequency(text, top_n)
            
            # 合并关键词
            combined_keywords = list(set(rake_keywords + freq_keywords))
            
            # 过滤和排序关键词
            filtered_keywords = self._filter_keywords(combined_keywords)
            sorted_keywords = sorted(
                filtered_keywords,
                key=lambda x: (self._keyword_score(x, text), -len(x))
            )
            
            return sorted_keywords[:top_n]
            
        except Exception as e:
            logger.error(f"Failed to extract keywords: {e}")
            return self._extract_by_frequency(text, top_n)  # 使用简单的词频统计作为后备方案

    def _extract_by_frequency(self, text: str, top_n: int) -> List[str]:
        """使用词频统计提取关键词"""
        try:
            # 分词
            words = re.findall(r'\w+', text.lower())
            
            # 过滤停用词
            try:
                stopwords = set(nltk.corpus.stopwords.words('english'))
            except:
                stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
            
            words = [w for w in words if w not in stopwords and len(w) > 2]
            
            # 统计词频
            word_freq = Counter(words)
            
            # 返回最常见的词
            return [word for word, _ in word_freq.most_common(top_n)]
        except Exception as e:
            logger.error(f"Error in frequency-based extraction: {e}")
            return []

    def _filter_keywords(self, keywords: List[str]) -> List[str]:
        """过滤无关的关键词"""
        try:
            stopwords = set(nltk.corpus.stopwords.words('english'))
        except:
            stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
            
        filtered = []
        
        for kw in keywords:
            # 转换为小写
            kw = kw.lower().strip()
            
            # 跳过太短或包含停用词的关键词
            if len(kw) < 3 or any(word in stopwords for word in kw.split()):
                continue
                
            # 跳过包含数字或特殊字符的关键词
            if any(c.isdigit() or not c.isalnum() for c in kw):
                continue
                
            filtered.append(kw)
            
        return filtered

    def _keyword_score(self, keyword: str, text: str) -> float:
        """计算关键词的相关性分数"""
        # 频率分数
        freq = text.lower().count(keyword.lower())
        
        # 位置分数（越早出现分数越高）
        pos = text.lower().find(keyword.lower())
        pos_score = 1.0 / (pos + 1) if pos != -1 else 0
        
        # 长度分数（适当偏好较长的关键词）
        length_score = min(len(keyword) / 10.0, 1.0)
        
        return freq * 0.5 + pos_score * 0.3 + length_score * 0.2

def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
        num_keywords: 要提取的关键词数量
    
    Returns:
        关键词列表
    """
    try:
        extractor = KeywordExtractor()
        return extractor.extract_keywords(text, num_keywords)
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

# 示例用法
if __name__ == "__main__":
    # 测试代码
    test_text = """
    Blockchain technology has revolutionized the way we think about digital transactions
    and decentralized systems. Smart contracts enable automated, trustless execution of
    agreements, while cryptocurrencies provide a new paradigm for financial transactions.
    """
    
    print("Testing keyword extraction:")
    print(extract_keywords(test_text))