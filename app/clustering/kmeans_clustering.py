# kmeans_clustering.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from app.utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)  # 使用 __name__ 作为日志记录器名称

def perform_kmeans_clustering(embeddings, n_clusters=5):
    """
    对文档嵌入向量进行 K-means 聚类。
    
    Args:
        embeddings (np.ndarray): 文档嵌入向量数组
        n_clusters (int): 聚类数量
        
    Returns:
        dict: 聚类结果，包含标签和中心点
    """
    try:
        # 数据标准化
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # 执行 K-means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_embeddings)
        
        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_
        
        # 计算每个样本到其聚类中心的距离
        distances = np.linalg.norm(scaled_embeddings - cluster_centers[cluster_labels], axis=1)
        
        logger.info(f"Successfully performed clustering with {n_clusters} clusters")
        
        return {
            'labels': cluster_labels.tolist(),
            'centers': cluster_centers.tolist(),
            'distances': distances.tolist(),
            'inertia': float(kmeans.inertia_)
        }
        
    except Exception as e:
        logger.error(f"Failed to perform clustering: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    sample_embeddings = np.random.rand(10, 384)  # 10个文档，每个384维
    
    # 执行聚类
    results = perform_kmeans_clustering(sample_embeddings, n_clusters=3)
    
    # 打印结果
    print("Cluster Labels:", results['labels'])
    print("Number of documents in each cluster:", 
          [results['labels'].count(i) for i in range(3)])
    print("Inertia:", results['inertia'])

# 添加别名以保持向后兼容性
perform_clustering = perform_kmeans_clustering