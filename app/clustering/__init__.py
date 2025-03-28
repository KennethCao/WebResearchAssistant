# __init__.py
from app.utils.logger import setup_logger
from app.clustering.clustering_utils import perform_kmeans_clustering

# 设置日志记录器
logger = setup_logger(__name__) 