import threading
import time
from datetime import datetime, timedelta
from .storage import MetricsStorage
from ..logger import logger

class MetricsReporter:
    def __init__(self, interval=300):
        self.interval = interval
        self._timer = None
        self._lock = threading.Lock()
        self._stopped = False

    def _run(self):
        try:
            while not self._stopped:
                self.report()
                time.sleep(self.interval)
        except Exception as e:
            logger.error(f"指标报告线程异常: {str(e)}")

    def start(self):
        with self._lock:
            if not self._timer:
                self._stopped = False
                self._timer = threading.Thread(target=self._run, daemon=True)
                self._timer.start()
                logger.info("启动指标报告服务")

    def stop(self):
        with self._lock:
            self._stopped = True
            if self._timer:
                self._timer.join()
            logger.info("停止指标报告服务")

    def report(self):
        try:
            storage = MetricsStorage.get_instance()
            metrics = storage.get_current_metrics()
            
            # 生成基础报告
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'request_count': metrics['request_counts']['total'],
                'error_rate': metrics['request_counts']['errors'] / max(metrics['request_counts']['total'], 1),
                'avg_latency': self._calculate_avg_latency(metrics['latency_data'])
            }
            
            logger.info("系统性能报告:\n"
                       f"总请求量: {report['request_count']}次\n"
                       f"错误率: {report['error_rate']:.2%}\n"
                       f"平均延迟: {report['avg_latency']:.2f}ms")
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {str(e)}")

    def _calculate_avg_latency(self, latency_data):
        total = 0
        count = 0
        for time_bucket in latency_data.values():
            count += time_bucket['count']
            total += time_bucket['sum']
        return total / count if count > 0 else 0

def start_metrics_reporter(app):
    reporter = MetricsReporter(interval=app.config.get('METRICS_INTERVAL', 300))
    reporter.start()
    return reporter