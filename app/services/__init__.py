from app.nlp.keyword_extractor import extract_keywords
from app.nlp.trend_analysis import analyze_trend
from app.clustering.kmeans_clustering import perform_kmeans_clustering
from app.utils.logger import setup_logger

# 设置日志记录器，提供模块名称
logger = setup_logger(__name__)

# 导出服务
__all__ = [
    'extract_keywords',
    'analyze_trend'
]

def analyze_blockchain_keywords(text, top_n=10):
    """
    提取区块链领域文献的关键词。
    :param text: 文献文本内容
    :param top_n: 提取的关键词数量
    :return: 区块链相关关键词列表
    """
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    if top_n <= 0:
        logger.error("提取的关键词数量必须大于 0")
        raise ValueError("提取的关键词数量必须大于 0")

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
    if not data:
        logger.error("输入数据为空")
        raise ValueError("输入数据不能为空")

    if num_clusters <= 0:
        logger.error("聚类数量必须大于 0")
        raise ValueError("聚类数量必须大于 0")

    try:
        logger.info("开始对区块链文献进行聚类分析")
        labels = perform_kmeans_clustering(data, num_clusters=num_clusters)
        logger.info(f"聚类分析完成，聚类结果: {labels}")
        return labels
    except Exception as e:
        logger.error(f"区块链文献聚类分析失败: {e}")
        raise RuntimeError(f"区块链文献聚类分析失败: {str(e)}")

def detect_blockchain_trends(data, keyword_list, time_column="timestamp", text_column="text"):
    """
    分析区块链领域的研究趋势。
    :param data: 区块链文献的时间序列数据（如年份与论文数量）
    :param keyword_list: 需要分析的关键词列表
    :param time_column: 时间字段名称（默认为 "timestamp"）
    :param text_column: 文本字段名称（默认为 "text"）
    :return: 趋势分析结果
    """
    if not data:
        logger.error("输入数据为空")
        raise ValueError("输入数据不能为空")

    if not keyword_list:
        logger.error("关键词列表为空")
        raise ValueError("关键词列表不能为空")

    if not isinstance(keyword_list, list):
        logger.error("关键词列表必须是列表类型")
        raise TypeError("关键词列表必须是列表类型")

    try:
        logger.info("开始分析区块链领域的研究趋势")
        trend_result = analyze_trend(data, keyword_list, time_column=time_column, text_column=text_column)
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
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    if not blockchain_keywords:
        logger.error("区块链关键词列表为空")
        raise ValueError("区块链关键词列表不能为空")

    if not isinstance(blockchain_keywords, list):
        logger.error("区块链关键词列表必须是列表类型")
        raise TypeError("区块链关键词列表必须是列表类型")

    try:
        logger.info("评估文献与区块链领域的相关性")
        text_keywords = extract_keywords(text)
        relevance_score = len(set(text_keywords) & set(blockchain_keywords)) / len(blockchain_keywords)
        logger.info(f"文献相关性得分: {relevance_score}")
        return relevance_score
    except Exception as e:
        logger.error(f"文献相关性评估失败: {e}")
        raise RuntimeError(f"文献相关性评估失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例文本
    text = "区块链技术在金融领域的应用越来越广泛。智能合约是区块链技术的核心组成部分。区块链技术提供了去中心化的解决方案。区块链技术的安全性和透明性是其主要优势。"

    # 示例关键词列表
    blockchain_keywords = ["区块链技术", "智能合约", "去中心化", "安全性", "透明性"]

    # 提取关键词
    keywords = analyze_blockchain_keywords(text, top_n=5)
    print("提取的关键词:", keywords)

    # 示例数据（嵌入向量或特征矩阵）
    data = np.random.rand(100, 10)  # 示例数据

    # 聚类分析
    labels = classify_blockchain_documents(data, num_clusters=3)
    print("聚类结果:", labels)

    # 示例时间序列数据
    trend_data = [
        {"timestamp": "2020-01-01", "text": "区块链技术在金融领域的应用越来越广泛。"},
        {"timestamp": "2020-02-01", "text": "智能合约是区块链技术的核心组成部分。"},
        {"timestamp": "2020-03-01", "text": "区块链技术提供了去中心化的解决方案。"},
        {"timestamp": "2020-04-01", "text": "区块链技术的安全性和透明性是其主要优势。"},
    ]

    # 趋势分析
    trend_result = detect_blockchain_trends(trend_data, keyword_list=blockchain_keywords)
    print("趋势分析结果:", trend_result)

    # 相关性评估
    relevance_score = evaluate_blockchain_relevance(text, blockchain_keywords)
    print("文献相关性得分:", relevance_score)