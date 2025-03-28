from app.nlp.summarizer import generate_summary
from app.nlp.keyword_extractor import extract_keywords
from app.nlp.similarity_checker import compute_similarity
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def extract_summary(text, max_length=150):
    """
    生成文本摘要。
    :param text: 输入文本
    :param max_length: 摘要的最大长度
    :return: 生成的摘要
    """
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    if max_length <= 0:
        logger.error("摘要的最大长度必须大于 0")
        raise ValueError("摘要的最大长度必须大于 0")

    try:
        logger.info("开始生成文本摘要")
        summary = generate_summary(text, max_length=max_length)
        logger.info("文本摘要生成完成")
        return summary
    except Exception as e:
        logger.error(f"生成文本摘要失败: {e}")
        raise RuntimeError(f"生成文本摘要失败: {str(e)}")

def extract_keywords(text, top_n=10):
    """
    提取文本中的关键词。
    :param text: 输入文本
    :param top_n: 提取的关键词数量
    :return: 关键词列表
    """
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    if top_n <= 0:
        logger.error("提取的关键词数量必须大于 0")
        raise ValueError("提取的关键词数量必须大于 0")

    try:
        logger.info("开始提取文本关键词")
        keywords = extract_keywords(text, top_n=top_n)
        logger.info("文本关键词提取完成")
        return keywords
    except Exception as e:
        logger.error(f"提取文本关键词失败: {e}")
        raise RuntimeError(f"提取文本关键词失败: {str(e)}")

def analyze_similarity(text1, text2):
    """
    计算两个文本的相似度。
    :param text1: 第一个文本
    :param text2: 第二个文本
    :return: 相似度分数（0 到 1）
    """
    if not text1:
        logger.error("第一个文本为空")
        raise ValueError("第一个文本不能为空")

    if not text2:
        logger.error("第二个文本为空")
        raise ValueError("第二个文本不能为空")

    try:
        logger.info("开始计算文本相似度")
        similarity = compute_similarity(text1, text2)
        logger.info(f"文本相似度计算完成: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"计算文本相似度失败: {e}")
        raise RuntimeError(f"计算文本相似度失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例文本
    text1 = "区块链技术在金融领域的应用越来越广泛。"
    text2 = "智能合约是区块链技术的核心组成部分。"

    # 生成文本摘要
    summary = extract_summary(text1, max_length=100)
    print("文本摘要:", summary)

    # 提取文本关键词
    keywords = extract_keywords(text1, top_n=5)
    print("关键词:", keywords)

    # 计算文本相似度
    similarity = analyze_similarity(text1, text2)
    print("文本相似度:", similarity)