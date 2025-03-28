from threading import Lock
from typing import Dict
from datetime import datetime

class MetricsStorage:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._init_storage()
            return cls._instance

    def _init_storage(self):
        self.request_counts = {'total': 0, 'errors': 0}
        self.latency_data = {}
        self._data_lock = Lock()

    def record_request(self, duration_ms: float, status_code: int):
        with self._data_lock:
            # 记录总请求数
            self.request_counts['total'] += 1
            
            # 记录错误请求
            if status_code >= 400:
                self.request_counts['errors'] += 1
            
            # 按时间窗口存储延迟数据（每分钟一个桶）
            time_bucket = datetime.utcnow().strftime('%Y-%m-%dT%H:%M')
            bucket = self.latency_data.setdefault(time_bucket, {
                'count': 0,
                'sum': 0.0,
                'max': 0.0,
                'buckets': {}
            })
            
            # 更新统计信息
            bucket['count'] += 1
            bucket['sum'] += duration_ms
            bucket['max'] = max(bucket['max'], duration_ms)
            
            # 按100ms间隔分桶
            latency_bucket = int(duration_ms // 100) * 100
            bucket['buckets'][latency_bucket] = bucket['buckets'].get(latency_bucket, 0) + 1

    def get_current_metrics(self) -> Dict:
        with self._data_lock:
            return {
                'request_counts': self.request_counts.copy(),
                'latency_data': self.latency_data.copy()
            }

    @classmethod
    def get_instance(cls):
        return cls()