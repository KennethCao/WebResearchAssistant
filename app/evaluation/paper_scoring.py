from app.nlp.keyword_extractor import extract_keywords
from app.nlp.similarity_checker import compute_similarity
from app.utils.logger import setup_logger

# 设置日志记录
# 假设 setup_logger 函数需要 name 参数，这里添加一个默认的名称
logger = setup_logger(name="paper_scoring")

def score_paper(text):
    """
    根据文献内容计算综合评分。
    :param text: 文献文本内容
    :return: 文献评分（0-100）
    """
    if not text:
        logger.error("Input text is empty")
        raise ValueError("Input text cannot be empty")

    try:
        logger.info("开始对文献进行评分")

        # 评分维度
        structure_score = evaluate_structure(text)       # 结构评分
        language_score = evaluate_language_quality(text) # 语言质量评分
        relevance_score = evaluate_relevance(text)       # 领域相关性评分
        innovation_score = evaluate_innovation(text)     # 创新性评分

        # 综合评分计算
        total_score = (
            0.3 * structure_score +
            0.3 * language_score +
            0.2 * relevance_score +
            0.2 * innovation_score
        )
        logger.info(f"文献评分完成，总分: {total_score:.2f}")
        return round(total_score, 2)
    except Exception as e:
        logger.error(f"文献评分失败: {e}")
        raise RuntimeError(f"文献评分失败: {str(e)}")


def evaluate_structure(text):
    """
    评估文献的结构，包括标题、摘要、章节等内容是否完整。
    :param text: 文献文本内容
    :return: 结构评分（0-100）
    """
    if not text:
        logger.error("Input text is empty")
        raise ValueError("Input text cannot be empty")

    try:
        logger.info("评估文献结构完整性")
        has_title = "标题" in text or "title" in text.lower()
        has_abstract = "摘要" in text or "abstract" in text.lower()
        has_sections = any(section in text.lower() for section in ["introduction", "methods", "results", "discussion"])

        score = 0
        if has_title:
            score += 30
        if has_abstract:
            score += 30
        if has_sections:
            score += 40

        logger.info(f"文献结构评分: {score}")
        return score
    except Exception as e:
        logger.error(f"结构评分失败: {e}")
        raise RuntimeError(f"结构评分失败: {str(e)}")


def evaluate_language_quality(text):
    """
    评估文献的语言质量，包括语法、拼写和表达清晰度。
    :param text: 文献文本内容
    :return: 语言质量评分（0-100）
    """
    if not text:
        logger.error("Input text is empty")
        raise ValueError("Input text cannot be empty")

    try:
        logger.info("评估文献语言质量")
        word_count = len(text.split())
        typo_count = detect_typos(text)  # 假设有一个拼写错误检测函数

        # 计算语法评分
        grammar_score = 100 - (typo_count / word_count) * 100 if word_count > 0 else 0

        # 语言质量评分
        score = max(0, min(100, grammar_score))
        logger.info(f"文献语言质量评分: {score}")
        return score
    except Exception as e:
        logger.error(f"语言质量评分失败: {e}")
        raise RuntimeError(f"语言质量评分失败: {str(e)}")


def evaluate_relevance(text):
    """
    评估文献与区块链领域的相关性。
    :param text: 文献文本内容
    :return: 领域相关性评分（0-100）
    """
    if not text:
        logger.error("Input text is empty")
        raise ValueError("Input text cannot be empty")

    try:
        logger.info("评估文献领域相关性")
        blockchain_keywords = ["blockchain", "smart contract", "decentralized", "distributed ledger"]
        text_keywords = extract_keywords(text)
        match_count = len(set(text_keywords) & set(blockchain_keywords))

        score = (match_count / len(blockchain_keywords)) * 100 if blockchain_keywords else 0
        logger.info(f"文献领域相关性评分: {score}")
        return score
    except Exception as e:
        logger.error(f"领域相关性评分失败: {e}")
        raise RuntimeError(f"领域相关性评分失败: {str(e)}")


def evaluate_innovation(text):
    """
    评估文献的创新性。
    :param text: 文献文本内容
    :return: 创新性评分（0-100）
    """
    if not text:
        logger.error("Input text is empty")
        raise ValueError("Input text cannot be empty")

    try:
        logger.info("评估文献创新性")
        # 模拟创新性评估：基于文献内容与已有文献的相似性
        existing_papers = [
            "区块链是一种分布式账本技术，它允许多个参与者安全地记录交易信息。",
            "比特币是区块链技术的第一个实际应用。",
        ]

        # 计算与现有文献的相似性
        similarity_scores = [compute_similarity(text, paper) for paper in existing_papers]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        # 创新性评分 = 100 - 平均相似度（越低相似度，创新性越高）
        score = max(0, min(100, 100 - avg_similarity * 100))
        logger.info(f"文献创新性评分: {score}")
        return score
    except Exception as e:
        logger.error(f"创新性评分失败: {e}")
        raise RuntimeError(f"创新性评分失败: {str(e)}")


def detect_typos(text):
    """
    检测文本中的拼写错误数量（占位函数）。
    :param text: 文本内容
    :return: 拼写错误数量
    """
    # 模拟拼写检测（实际应替换为语言工具，如 Hunspell 或 pyspellchecker）
    return 5  # 假设检测到 5 个拼写错误