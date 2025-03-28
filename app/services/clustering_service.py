from app.clustering.kmeans_clustering import perform_kmeans_clustering, compute_silhouette_score
from app.clustering.topic_network import build_topic_network
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def perform_clustering(data, num_clusters=5):
    """
    执行 KMeans 聚类分析。
    :param data: 输入数据（嵌入向量或特征矩阵）
    :param num_clusters: 聚类数量
    :return: 聚类标签列表
    """
    if not data:
        logger.error("输入数据为空")
        raise ValueError("输入数据不能为空")

    if num_clusters <= 0:
        logger.error("聚类数量必须大于 0")
        raise ValueError("聚类数量必须大于 0")

    try:
        logger.info(f"开始执行 KMeans 聚类分析，聚类数量: {num_clusters}")
        labels = perform_kmeans_clustering(data, num_clusters=num_clusters)
        logger.info("KMeans 聚类分析完成")
        return labels
    except Exception as e:
        logger.error(f"KMeans 聚类分析失败: {e}")
        raise RuntimeError(f"KMeans 聚类分析失败: {str(e)}")

def evaluate_clustering(data, labels):
    """
    评估聚类质量。
    :param data: 输入数据（嵌入向量或特征矩阵）
    :param labels: 聚类结果标签
    :return: 轮廓系数（Silhouette Score）
    """
    if not data:
        logger.error("输入数据为空")
        raise ValueError("输入数据不能为空")

    if not labels:
        logger.error("聚类结果标签为空")
        raise ValueError("聚类结果标签不能为空")

    try:
        logger.info("开始评估聚类质量")
        score = compute_silhouette_score(data, labels)
        logger.info(f"聚类质量评估完成，轮廓系数: {score}")
        return score
    except Exception as e:
        logger.error(f"聚类质量评估失败: {e}")
        raise RuntimeError(f"聚类质量评估失败: {str(e)}")

def generate_topic_network(data, labels):
    """
    构建主题网络图。
    :param data: 输入数据（嵌入向量或特征矩阵）
    :param labels: 聚类结果的标签
    :return: NetworkX 图对象
    """
    if not data:
        logger.error("输入数据为空")
        raise ValueError("输入数据不能为空")

    if not labels:
        logger.error("聚类结果标签为空")
        raise ValueError("聚类结果标签不能为空")

    try:
        logger.info("开始构建主题网络图")
        network = build_topic_network(data, labels)
        logger.info("主题网络图构建完成")
        return network
    except Exception as e:
        logger.error(f"构建主题网络图失败: {e}")
        raise RuntimeError(f"构建主题网络图失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例数据（嵌入向量或特征矩阵）
    data = np.random.rand(100, 10)  # 示例数据

    # 示例聚类数量
    num_clusters = 3

    # 执行聚类分析
    labels = perform_clustering(data, num_clusters=num_clusters)
    print("聚类标签:", labels)

    # 评估聚类质量
    silhouette_score = evaluate_clustering(data, labels)
    print("轮廓系数:", silhouette_score)

    # 构建主题网络图
    network = generate_topic_network(data, labels)
    print("主题网络图:", network)