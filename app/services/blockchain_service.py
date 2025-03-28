from app.nlp.keyword_extractor import extract_keywords
from app.nlp.trend_analysis import analyze_trend
from app.clustering.kmeans_clustering import perform_kmeans_clustering
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def analyze_blockchain_keywords(text, top_n=10):
    """
    提取区块链领域文献的关键词。
    :param text: 文献文本内容
    :param top_n: 提取的关键词数量
    :return: 区块链相关关键词列表
    """
    try:
        logger.info("提取区块链领域关键词")
        keywords = extract_keywords(text, top_n=top_n)
        logger.info(f"提取的关键词: {keywords}")
        return keywords
    except Exception as e:
        logger.error(f"提取区块链关键词失败: {e}")
        raise RuntimeError(f"提取区块链关键词失败: {str(e)}")


def classify_blockchain_documents(data, num_clusters=5):
    """
    使用 KMeans 对区块链文献进行聚类分析。
    :param data: 输入文献的嵌入向量或特征矩阵
    :param num_clusters: 聚类数量
    :return: 聚类结果标签
    """
    try:
        logger.info("开始对区块链文献进行聚类分析")
        labels = perform_kmeans_clustering(data, num_clusters=num_clusters)
        logger.info(f"聚类分析完成，聚类结果: {labels}")
        return labels
    except Exception as e:
        logger.error(f"区块链文献聚类分析失败: {e}")
        raise RuntimeError(f"区块链文献聚类分析失败: {str(e)}")


def detect_blockchain_trends(data):
    """
    分析区块链领域的研究趋势。
    :param data: 区块链文献的时间序列数据（如年份与论文数量）
    :return: 趋势分析结果
    """
    try:
        logger.info("开始分析区块链领域的研究趋势")
        trend_result = analyze_trend(data)
        logger.info(f"趋势分析结果: {trend_result}")
        return trend_result
    except Exception as e:
        logger.error(f"区块链领域趋势分析失败: {e}")
        raise RuntimeError(f"区块链领域趋势分析失败: {str(e)}")


def evaluate_blockchain_relevance(text, blockchain_keywords):
    """
    评估文献与区块链领域的相关性。
    :param text: 文献文本内容
    :param blockchain_keywords: 区块链领域的关键词列表
    :return: 相关性得分
    """
    try:
        logger.info("评估文献与区块链领域的相关性")
        text_keywords = extract_keywords(text)
        relevance_score = len(set(text_keywords) & set(blockchain_keywords)) / len(blockchain_keywords)
        logger.info(f"文献相关性得分: {relevance_score}")
        return relevance_score
    except Exception as e:
        logger.error(f"文献相关性评估失败: {e}")
        raise RuntimeError(f"文献相关性评估失败: {str(e)}")