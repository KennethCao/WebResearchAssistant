try:
    import os
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import pairwise_distances
    import matplotlib
    matplotlib.use('Agg')  # 设置后端为 Agg
    import matplotlib.pyplot as plt
    import seaborn as sns
    from app.utils.logger import setup_logger

    # 设置日志记录器
    logger = setup_logger(__name__)
    logger.info("Successfully imported all required packages")
    
except ImportError as e:
    print(f"Error importing required packages: {e}")
    raise

def normalize_embeddings(embeddings):
    """
    标准化嵌入向量。
    
    Args:
        embeddings: 输入的嵌入向量数组
        
    Returns:
        标准化后的嵌入向量数组
    """
    try:
        # 创建标准化器
        scaler = StandardScaler()
        
        # 标准化嵌入向量
        normalized_embeddings = scaler.fit_transform(embeddings)
        
        logger.info(f"Normalized embeddings with shape: {normalized_embeddings.shape}")
        return normalized_embeddings
        
    except Exception as e:
        logger.error(f"Failed to normalize embeddings: {e}")
        raise

def determine_optimal_k(embeddings, max_k=10):
    """
    使用肘部法则确定最佳聚类数。
    
    Args:
        embeddings: 输入的嵌入向量数组
        max_k: 最大聚类数
        
    Returns:
        最佳聚类数
    """
    try:
        # 计算不同 k 值的 SSE（误差平方和）
        distortions = []
        K = range(1, max_k + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            distortions.append(kmeans.inertia_)
        
        # 计算斜率变化
        slopes = np.diff(distortions)
        slope_changes = np.diff(slopes)
        
        # 找到斜率变化最大的点
        optimal_k = np.argmin(slope_changes) + 2
        
        logger.info(f"Determined optimal number of clusters: {optimal_k}")
        return optimal_k
        
    except Exception as e:
        logger.error(f"Failed to determine optimal k: {e}")
        raise

def perform_kmeans_clustering(embeddings, n_clusters=None):
    """
    执行 K-means 聚类。
    
    Args:
        embeddings: 输入的嵌入向量数组
        n_clusters: 聚类数量，如果为 None 则自动确定
        
    Returns:
        聚类标签和聚类中心
    """
    try:
        # 标准化嵌入向量
        normalized_embeddings = normalize_embeddings(embeddings)
        
        # 如果没有指定聚类数，自动确定
        if n_clusters is None:
            n_clusters = determine_optimal_k(normalized_embeddings)
        
        # 执行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(normalized_embeddings)
        centers = kmeans.cluster_centers_
        
        logger.info(f"Performed clustering with {n_clusters} clusters")
        return labels, centers
        
    except Exception as e:
        logger.error(f"Failed to perform clustering: {e}")
        raise

def calculate_cluster_metrics(embeddings, labels):
    """
    计算聚类指标。
    
    Args:
        embeddings: 输入的嵌入向量数组
        labels: 聚类标签
        
    Returns:
        包含各种聚类指标的字典
    """
    try:
        # 计算轮廓系数
        distances = pairwise_distances(embeddings)
        n_samples = len(embeddings)
        n_clusters = len(np.unique(labels))
        
        # 计算每个样本的轮廓系数
        silhouette_scores = []
        for i in range(n_samples):
            # 计算样本到同类其他样本的平均距离
            same_cluster = labels == labels[i]
            same_cluster[i] = False
            if np.any(same_cluster):
                a = np.mean(distances[i][same_cluster])
            else:
                a = 0
            
            # 计算样本到其他类样本的最小平均距离
            other_clusters = labels != labels[i]
            if np.any(other_clusters):
                b = np.min([np.mean(distances[i][labels == j]) 
                          for j in range(n_clusters) if j != labels[i]])
            else:
                b = 0
            
            # 计算轮廓系数
            if a == 0 and b == 0:
                s = 0
            else:
                s = (b - a) / max(a, b)
            silhouette_scores.append(s)
        
        # 计算平均轮廓系数
        avg_silhouette = np.mean(silhouette_scores)
        
        # 计算簇内距离
        intra_distances = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            if np.sum(cluster_mask) > 1:
                cluster_distances = distances[cluster_mask][:, cluster_mask]
                intra_distances.append(np.mean(cluster_distances))
        avg_intra_distance = np.mean(intra_distances) if intra_distances else 0
        
        metrics = {
            'silhouette_score': avg_silhouette,
            'intra_cluster_distance': avg_intra_distance,
            'n_clusters': n_clusters,
            'n_samples': n_samples
        }
        
        logger.info(f"Calculated clustering metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate clustering metrics: {e}")
        raise

def plot_clustering_results(embeddings, labels, save_path=None):
    """
    可视化聚类结果。
    
    Args:
        embeddings: 输入的嵌入向量数组
        labels: 聚类标签
        save_path: 图像保存路径
    """
    try:
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 如果维度大于2，使用 PCA 降维
        if embeddings.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        # 绘制散点图
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6
        )
        
        # 添加图例
        plt.colorbar(scatter)
        
        # 设置标题和标签
        plt.title('Clustering Results')
        plt.xlabel('First Component')
        plt.ylabel('Second Component')
        
        # 保存或显示图形
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved clustering plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot clustering results: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 300
    n_features = 10
    n_clusters = 3
    
    # 创建模拟的嵌入向量
    embeddings = np.random.randn(n_samples, n_features)
    
    try:
        # 执行聚类
        labels, centers = perform_kmeans_clustering(embeddings, n_clusters)
        
        # 计算聚类指标
        metrics = calculate_cluster_metrics(embeddings, labels)
        print("\nClustering Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # 可视化结果
        plot_clustering_results(embeddings, labels, "clustering_results.png")
        
    except Exception as e:
        print(f"Error: {e}")