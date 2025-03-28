from time import perf_counter
from typing import Callable
from flask import Request, Response
from threading import Lock
from ..metrics_storage import MetricsStorage

class RequestMetricsCollector:
    def __init__(self):
        self._lock = Lock()
        self.request_count = 0
        self.error_count = 0
        self.latency_buckets = {}

    def record_request(self, duration: float, status_code: int):
        with self._lock:
            self.request_count += 1
            if status_code >= 400:
                self.error_count += 1
            
            # 按100ms间隔分桶
            bucket = int(duration * 10) * 100
            self.latency_buckets[bucket] = self.latency_buckets.get(bucket, 0) + 1


def request_monitor(func: Callable) -> Callable:
    """API请求监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        response = func(*args, **kwargs)
        duration = (perf_counter() - start_time) * 1000  # 转换为毫秒
        
        # 获取状态码
        status_code = getattr(response, 'status_code', 500)
        
        # 记录指标
        MetricsStorage.get_instance().record_request(duration, status_code)
        return response
    return wrapper


class MetricsMiddleware:
    """Flask指标收集中间件"""
    def __init__(self, app):
        self.app = app
        self.app.before_request(self._before_request)
        self.app.after_request(self._after_request)

    def _before_request(self):
        request._start_time = perf_counter()

    def _after_request(self, response: Response) -> Response:
        duration = (perf_counter() - request._start_time) * 1000
        MetricsStorage.get_instance().record_request(duration, response.status_code)
        return response