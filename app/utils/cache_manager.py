from functools import wraps
from app.extensions import cache
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def cache_key(*args, **kwargs):
    """生成缓存键"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return ":".join(key_parts)

def cached(timeout=300):
    """缓存装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = cache_key(f.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            rv = cache.get(key)
            if rv is not None:
                return rv
                
            # 执行函数
            rv = f(*args, **kwargs)
            
            # 存入缓存
            try:
                cache.set(key, rv, timeout=timeout)
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")
                
            return rv
        return decorated_function
    return decorator

def clear_cache_for(key_pattern):
    """清除指定模式的缓存"""
    try:
        cache.delete_pattern(key_pattern)
        logger.info(f"Cleared cache for pattern: {key_pattern}")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

class CacheManager:
    """缓存管理器"""
    
    @staticmethod
    def cache_paper_analysis(paper_id, results, timeout=3600):
        """缓存论文分析结果"""
        key = f"paper:analysis:{paper_id}"
        try:
            cache.set(key, results, timeout=timeout)
            logger.info(f"Cached analysis results for paper {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache analysis results: {e}")
            return False
    
    @staticmethod
    def get_paper_analysis(paper_id):
        """获取缓存的论文分析结果"""
        key = f"paper:analysis:{paper_id}"
        try:
            return cache.get(key)
        except Exception as e:
            logger.error(f"Failed to get cached analysis: {e}")
            return None
    
    @staticmethod
    def cache_paper_embedding(paper_id, embedding, timeout=86400):
        """缓存论文向量嵌入"""
        key = f"paper:embedding:{paper_id}"
        try:
            cache.set(key, embedding, timeout=timeout)
            logger.info(f"Cached embedding for paper {paper_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")
            return False
    
    @staticmethod
    def clear_paper_cache(paper_id):
        """清除论文相关的所有缓存"""
        patterns = [
            f"paper:analysis:{paper_id}",
            f"paper:embedding:{paper_id}",
            f"paper:qa:{paper_id}:*"
        ]
        for pattern in patterns:
            clear_cache_for(pattern) 