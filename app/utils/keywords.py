from typing import List, Optional
import importlib
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class KeywordExtractor:
    def __init__(self):
        self._jieba = None
        
    @property
    def jieba(self):
        """延迟加载 jieba"""
        if self._jieba is None:
            try:
                self._jieba = importlib.import_module('jieba')
                # 设置小词典模式
                self._jieba.initialize(dictionary=None)
                logger.info("Jieba initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize jieba: {e}")
                raise
        return self._jieba

def extract_keywords(text: str, method: str = 'hybrid', num_keywords: int = 10) -> List[str]:
    """
    从文本中提取关键词
    
    Args:
        text: 要分析的文本
        method: 提取方法 ('tfidf', 'textrank', 'hybrid')
        num_keywords: 要提取的关键词数量
        
    Returns:
        关键词列表
    """
    try:
        extractor = KeywordExtractor()
        jieba = extractor.jieba
        
        # 区块链相关关键词列表
        blockchain_keywords = {
            "区块链", "智能合约", "去中心化", "共识机制", "加密货币",
            "比特币", "以太坊", "分布式账本", "数字货币", "挖矿",
            "公链", "私链", "联盟链", "跨链", "侧链",
            "代币", "钱包", "哈希", "节点", "交易"
        }
        
        keywords = []
        
        if method in ['tfidf', 'hybrid']:
            # 使用 TF-IDF 提取关键词
            tfidf_keywords = jieba.analyse.extract_tags(
                text,
                topK=num_keywords,
                withWeight=False
            )
            keywords.extend(tfidf_keywords)
            
        if method in ['textrank', 'hybrid']:
            # 使用 TextRank 提取关键词
            textrank_keywords = jieba.analyse.textrank(
                text,
                topK=num_keywords,
                withWeight=False
            )
            keywords.extend(textrank_keywords)
            
        # 过滤区块链相关关键词
        filtered = [
            kw for kw in keywords 
            if any(bk in kw or kw in bk for bk in blockchain_keywords)
        ]
        
        return filtered[:num_keywords] or keywords[:num_keywords]
        
    except Exception as e:
        logger.error(f"关键词提取失败: {e}")
        return []

def filter_blockchain_keywords(keywords: List[str]) -> List[str]:
    """
    过滤出区块链相关的关键词
    
    Args:
        keywords: 关键词列表
        
    Returns:
        过滤后的关键词列表
    """
    try:
        blockchain_keywords = {
            "区块链", "智能合约", "去中心化", "共识机制", "加密货币",
            "比特币", "以太坊", "分布式账本", "数字货币", "挖矿",
            "公链", "私链", "联盟链", "跨链", "侧链",
            "代币", "钱包", "哈希", "节点", "交易"
        }
        
        filtered = [
            kw for kw in keywords 
            if any(bk in kw or kw in bk for bk in blockchain_keywords)
        ]
        
        return filtered or keywords  # 如果没有匹配项，返回原始关键词
        
    except Exception as e:
        logger.error(f"关键词过滤失败: {e}")
        return keywords

def get_keyword_frequency(texts: List[str], top_n: int = 10) -> List[tuple]:
    """
    获取多个文本中关键词的出现频率
    
    Args:
        texts: 文本列表
        top_n: 返回前N个高频关键词
        
    Returns:
        (关键词, 频率) 元组的列表
    """
    try:
        keyword_count = {}
        
        for text in texts:
            keywords = extract_keywords(text)
            for kw in keywords:
                keyword_count[kw] = keyword_count.get(kw, 0) + 1
                
        # 按频率排序
        sorted_keywords = sorted(
            keyword_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_keywords[:top_n]
        
    except Exception as e:
        logger.error(f"获取关键词频率失败: {e}")
        return [] 