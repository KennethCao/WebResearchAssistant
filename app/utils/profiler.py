import time
import functools
from flask import request
from app.utils.logger import setup_logger
from app.extensions import cache

logger = setup_logger(__name__)

def profile_performance(func):
    """性能分析装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 计算执行时间
        execution_time = time.time() - start_time
        
        # 记录性能数据
        endpoint = request.endpoint if request else func.__name__
        _record_performance(endpoint, execution_time)
        
        return result
    return wrapper

def _record_performance(endpoint: str, execution_time: float):
    """记录性能数据"""
    try:
        # 获取现有统计数据
        stats_key = f"perf:stats:{endpoint}"
        stats = cache.get(stats_key) or {
            'count': 0,
            'total_time': 0,
            'avg_time': 0,
            'min_time': float('inf'),
            'max_time': 0
        }
        
        # 更新统计数据
        stats['count'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        # 保存统计数据
        cache.set(stats_key, stats, timeout=86400)  # 24小时
        
        # 记录慢请求
        if execution_time > 1.0:  # 超过1秒的请求
            logger.warning(
                f"Slow request detected: {endpoint} took {execution_time:.2f}s"
            )
            
    except Exception as e:
        logger.error(f"Failed to record performance data: {e}")

def get_performance_stats():
    """获取性能统计数据"""
    try:
        stats = {}
        for key in cache.keys("perf:stats:*"):
            endpoint = key.split(":")[-1]
            stats[endpoint] = cache.get(key)
        return stats
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {} 