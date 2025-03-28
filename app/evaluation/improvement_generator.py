from app.evaluation.paper_scoring import evaluate_structure, evaluate_language_quality, evaluate_relevance, evaluate_innovation
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def generate_improvement_suggestions(text):
    """
    根据文献内容生成改进建议。
    :param text: 文献文本内容
    :return: 改进建议列表
    """
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    try:
        logger.info("开始生成文献改进建议")

        # 分析各个评分维度
        structure_score = evaluate_structure(text)
        language_score = evaluate_language_quality(text)
        relevance_score = evaluate_relevance(text)
        innovation_score = evaluate_innovation(text)

        # 生成建议
        suggestions = []

        if structure_score < 100:
            suggestions.append(suggest_structure_improvement(text, structure_score))
        if language_score < 100:
            suggestions.append(suggest_language_improvement(language_score))
        if relevance_score < 100:
            suggestions.append(suggest_relevance_improvement(relevance_score))
        if innovation_score < 100:
            suggestions.append(suggest_innovation_improvement(innovation_score))

        logger.info("文献改进建议生成完成")
        return suggestions
    except Exception as e:
        logger.error(f"生成文献改进建议失败: {e}")
        raise RuntimeError(f"生成文献改进建议失败: {str(e)}")


def suggest_structure_improvement(text, score):
    """
    根据结构评分生成改进建议。
    :param text: 文献文本内容
    :param score: 结构评分
    :return: 改进建议
    """
    try:
        logger.info("生成结构改进建议")
        missing_elements = []

        if "标题" not in text and "title" not in text.lower():
            missing_elements.append("标题")
        if "摘要" not in text and "abstract" not in text.lower():
            missing_elements.append("摘要")
        if not any(section in text.lower() for section in ["introduction", "methods", "results", "discussion"]):
            missing_elements.append("章节（如引言、方法、结果或讨论）")

        suggestion = (f"文献结构评分为 {score} 分，建议补充以下缺失部分: {', '.join(missing_elements)}。"
                      if missing_elements else "文献结构完整，无需改进。")
        logger.info(f"结构改进建议: {suggestion}")
        return suggestion
    except Exception as e:
        logger.error(f"生成结构改进建议失败: {e}")
        raise RuntimeError(f"生成结构改进建议失败: {str(e)}")


def suggest_language_improvement(score):
    """
    根据语言评分生成改进建议。
    :param score: 语言评分
    :return: 改进建议
    """
    try:
        logger.info("生成语言质量改进建议")
        suggestion = (f"语言评分为 {score} 分，建议检查拼写错误、语法问题，并确保表达清晰简洁。"
                      "适当使用专业术语，提高学术性。" if score < 100 else "语言质量优秀，无需改进。")
        logger.info(f"语言改进建议: {suggestion}")
        return suggestion
    except Exception as e:
        logger.error(f"生成语言改进建议失败: {e}")
        raise RuntimeError(f"生成语言改进建议失败: {str(e)}")


def suggest_relevance_improvement(score):
    """
    根据领域相关性评分生成改进建议。
    :param score: 相关性评分
    :return: 改进建议
    """
    try:
        logger.info("生成领域相关性改进建议")
        suggestion = (f"领域相关性评分为 {score} 分，建议在文献中增加更多与区块链相关的关键词和内容，"
                      "例如 '智能合约'、'去中心化'、'分布式账本' 等。" if score < 100 else "文献领域相关性很高，无需改进。")
        logger.info(f"相关性改进建议: {suggestion}")
        return suggestion
    except Exception as e:
        logger.error(f"生成领域相关性改进建议失败: {e}")
        raise RuntimeError(f"生成领域相关性改进建议失败: {str(e)}")


def suggest_innovation_improvement(score):
    """
    根据创新性评分生成改进建议。
    :param score: 创新性评分
    :return: 改进建议
    """
    try:
        logger.info("生成创新性改进建议")
        suggestion = (f"创新性评分为 {score} 分，建议在文献中提出新的研究问题或方法，引用最新文献，"
                      "或结合实际案例，展示研究的独特价值。" if score < 100 else "文献具有较高的创新性，无需改进。")
        logger.info(f"创新性改进建议: {suggestion}")
        return suggestion
    except Exception as e:
        logger.error(f"生成创新性改进建议失败: {e}")
        raise RuntimeError(f"生成创新性改进建议失败: {str(e)}")