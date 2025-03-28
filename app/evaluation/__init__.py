from app.evaluation.paper_scoring import score_paper
from app.evaluation.improvement_generator import generate_improvement_suggestions
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def evaluate_paper(text):
    """
    评估文献质量并生成评分。
    :param text: 文献文本内容
    :return: 文献评分
    """
    try:
        logger.info("开始评估文献质量")
        score = score_paper(text)
        logger.info(f"文献质量评分完成: {score}")
        return score
    except Exception as e:
        logger.error(f"文献质量评估失败: {e}")
        raise RuntimeError(f"文献质量评估失败: {str(e)}")


def provide_improvement_suggestions(text):
    """
    为文献生成改进建议。
    :param text: 文献文本内容
    :return: 改进建议
    """
    try:
        logger.info("开始生成文献改进建议")
        suggestions = generate_improvement_suggestions(text)
        logger.info("文献改进建议生成完成")
        return suggestions
    except Exception as e:
        logger.error(f"生成文献改进建议失败: {e}")
        raise RuntimeError(f"生成文献改进建议失败: {str(e)}")