try:
    from prometheus_client import make_wsgi_app, CollectorRegistry, Gauge, Counter, Histogram
except ImportError:
    print("无法解析导入 'prometheus_client'，请确保已经安装了该库。可以使用以下命令进行安装：")
    print("pip install prometheus_client")
    raise
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import os
from flask import request
import time
from flask import current_app
from flask_sqlalchemy import SQLAlchemy

class PrometheusMonitor:
    def __init__(self, app=None):
        self.app = app
        self.registry = CollectorRegistry()
        self._init_metrics()
        self.similarity_model = None
        
        if app:
            self.init_app(app)

    def init_app(self, app):
        """延迟初始化应用"""
        self.app = app
        self._setup_middleware()
        app.add_url_rule('/health', 'health_check', self.health_check)

    def _init_metrics(self):
        """初始化核心监控指标"""
        self.request_count = Counter(
            'flask_http_request_total',
            'Total HTTP Requests',
            ['method', 'endpoint', 'http_status'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'flask_http_request_duration_seconds',
            'HTTP Request Latency',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.active_requests = Gauge(
            'flask_http_active_requests',
            'Active HTTP Requests',
            ['method', 'endpoint'],
            registry=self.registry
        )

    def _setup_middleware(self):
        """配置监控中间件"""
        if current_app.config.get('MONITORING', True) and self.app:
            self.app.wsgi_app = DispatcherMiddleware(self.app.wsgi_app, {
                '/metrics': make_wsgi_app(registry=self.registry)
            })

    def health_check(self):
        """统一健康检查端点"""
        return jsonify({
            'status': 'OK',
            'components': {
                'database': self._check_database(),
                'models': self._check_models()
            }
        })

    db = SQLAlchemy()

    def _check_database(self):
        try:
            db.session.execute('SELECT 1')
            return 'available'
        except Exception as e:
            current_app.logger.error(f"Database check failed: {e}")
            return 'unavailable'

    def _check_models(self):
        return 'loaded' if hasattr(current_app, 'faiss_retriever') else 'unloaded'

    def monitor(self):
        """装饰器方法用于业务指标监控"""
        def decorator(f):
            def wrapped_function(*args, **kwargs):
                start_time = time.time()
                self.active_requests.labels(request.method, request.path).inc()
                
                try:
                    response = f(*args, **kwargs)
                    status_code = response.status_code
                except Exception as e:
                    status_code = 500
                    raise e
                finally:
                    latency = time.time() - start_time
                    self.request_latency.labels(
                        request.method, request.path
                    ).observe(latency)
                    
                    self.request_count.labels(
                        request.method, 
                        request.path, 
                        status_code
                    ).inc()
                    
                    self.active_requests.labels(request.method, request.path).dec()
                
                return response
            return wrapped_function
        return decorator


def setup_prometheus_monitoring(app):
    """工厂方法初始化监控系统"""
    if os.getenv('ENABLE_PROMETHEUS_MONITORING', 'true').lower() == 'true':
        return PrometheusMonitor(app)
    return None