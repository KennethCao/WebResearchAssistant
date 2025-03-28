from app.utils.logger import setup_logger
from app.clustering.kmeans_clustering import perform_kmeans_clustering
from app.clustering.topic_network import build_topic_network
from app.nlp.embedding_generator import generate_embeddings
import numpy as np

# 设置日志记录
logger = setup_logger()

def cluster_blockchain_papers(paper_texts, num_clusters=5):
    """
    对区块链论文进行聚类分析。
    :param paper_texts: 区块链论文的文本列表
    :param num_clusters: 聚类数量
    :return: 聚类结果和主题网络图
    """
    try:
        # 生成嵌入向量
        embeddings = [generate_embeddings(text) for text in paper_texts]
        
        # 确保 embeddings 是 numpy 数组
        embeddings = np.array([e.cpu().numpy() for e in embeddings])
        
        # 执行 KMeans 聚类
        logger.info(f"执行 KMeans 聚类分析，聚类数量: {num_clusters}")
        cluster_labels = perform_kmeans_clustering(embeddings, num_clusters)
        logger.info("KMeans 聚类分析完成")
        
        # 生成主题网络图
        logger.info("生成主题网络图")
        topic_network = build_topic_network(embeddings, cluster_labels)
        logger.info("主题网络图生成完成")
        
        return cluster_labels, topic_network
    except Exception as e:
        logger.error(f"区块链论文聚类失败: {e}")
        raise RuntimeError(f"区块链论文聚类失败: {str(e)}")

def get_cluster_summary(cluster_labels, paper_texts):
    """
    获取每个聚类的摘要信息。
    :param cluster_labels: 聚类结果的标签
    :param paper_texts: 区块链论文的文本列表
    :return: 每个聚类的摘要信息
    """
    try:
        cluster_summaries = {}
        unique_clusters = set(cluster_labels)
        
        for cluster in unique_clusters:
            cluster_texts = [paper_texts[i] for i, label in enumerate(cluster_labels) if label == cluster]
            cluster_summaries[cluster] = generate_cluster_summary(cluster_texts)
        
        return cluster_summaries
    except Exception as e:
        logger.error(f"获取聚类摘要失败: {e}")
        raise RuntimeError(f"获取聚类摘要失败: {str(e)}")

def generate_cluster_summary(cluster_texts):
    """
    生成单个聚类的摘要。
    :param cluster_texts: 单个聚类中的论文文本列表
    :return: 聚类的摘要
    """
    try:
        from app.nlp.summarizer import generate_summary
        summary = generate_summary(" ".join(cluster_texts))
        return summary
    except Exception as e:
        logger.error(f"生成聚类摘要失败: {e}")
        raise RuntimeError(f"生成聚类摘要失败: {str(e)}")