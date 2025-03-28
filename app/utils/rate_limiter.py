from functools import wraps
from flask import request, jsonify
from app.extensions import cache
from app.utils.logger import setup_logger
import time

logger = setup_logger(__name__)

class RateLimiter:
    def __init__(self, requests=100, window=60):
        self.requests = requests  # 请求次数限制
        self.window = window     # 时间窗口(秒)
    
    def __call__(self, f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # 获取客户端IP
            client_ip = request.remote_addr
            
            # 生成缓存键
            key = f"rate_limit:{client_ip}:{f.__name__}"
            
            try:
                # 获取当前请求记录
                request_history = cache.get(key) or []
                
                # 清理过期记录
                current_time = time.time()
                request_history = [t for t in request_history 
                                 if current_time - t < self.window]
                
                # 检查是否超过限制
                if len(request_history) >= self.requests:
                    return jsonify({
                        'error': 'Too Many Requests',
                        'message': 'Rate limit exceeded'
                    }), 429
                
                # 添加新请求记录
                request_history.append(current_time)
                cache.set(key, request_history, timeout=self.window)
                
                return f(*args, **kwargs)
                
            except Exception as e:
                logger.error(f"Rate limiter error: {e}")
                # 发生错误时允许请求通过
                return f(*args, **kwargs)
                
        return decorated

# 使用示例:
# @RateLimiter(requests=100, window=60)
# def some_api_endpoint():
#     pass 