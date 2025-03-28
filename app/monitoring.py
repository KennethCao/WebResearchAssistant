from flask import Blueprint, jsonify
from prometheus_client import generate_latest, REGISTRY
import time
import threading
from flask_sse import sse

monitoring_blueprint = Blueprint('monitoring', __name__)

@monitoring_blueprint.route('/metrics')
def metrics():
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain'}

@monitoring_blueprint.route('/stream')
def stream():
    return sse.stream()

class MetricsCollector(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True

    def run(self):
        while self.running:
            with app.app_context():
                data = {
                    'rps': self.calculate_rps(),
                    'latency': self.get_latency_metrics()
                }
                sse.publish(data, type='metrics')
            time.sleep(1)

    def calculate_rps(self):
        return REGISTRY.get_sample_value('http_requests_total')

    def get_latency_metrics(self):
        return {
            'p99': REGISTRY.get_sample_value('http_request_duration_seconds_bucket', {'le': '0.1'}),
            'avg': REGISTRY.get_sample_value('http_request_duration_seconds_sum') / REGISTRY.get_sample_value('http_request_duration_seconds_count')
        }