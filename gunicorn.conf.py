import multiprocessing
import os

# Server socket
bind = os.getenv('GUNICORN_BIND', '0.0.0.0:5000')
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = 'blockchain-research-assistant'

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# SSL (uncomment for HTTPS)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Server hooks
def on_starting(server):
    """Server startup hook"""
    pass

def on_reload(server):
    """Server reload hook"""
    pass

def when_ready(server):
    """Server ready hook"""
    pass

def on_exit(server):
    """Server exit hook"""
    pass 