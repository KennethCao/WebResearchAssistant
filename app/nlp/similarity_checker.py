import torch
from sentence_transformers import util
from app.nlp.embedding_generator import get_embedding_generator
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def compute_similarity(text1, text2):
    """
    计算两个文本的相似度。
    :param text1: 第一个文本
    :param text2: 第二个文本
    :return: 相似度分数（0到1之间）
    """
    if not text1:
        logger.error("第一个文本为空")
        raise ValueError("第一个文本不能为空")

    if not text2:
        logger.error("第二个文本为空")
        raise ValueError("第二个文本不能为空")

    try:
        logger.info("生成文本的嵌入向量")

        # 获取嵌入生成器实例
        embedding_generator = get_embedding_generator()

        # 强制使用 CPU
        with torch.no_grad():
            embedding1 = embedding_generator.generate_embeddings(text1, normalize=True).cpu()
            embedding2 = embedding_generator.generate_embeddings(text2, normalize=True).cpu()

        logger.info("计算文本相似度")
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        logger.info(f"文本相似度: {similarity}")
        return similarity
    except Exception as e:
        logger.error(f"计算文本相似度失败: {e}")
        raise RuntimeError(f"计算文本相似度失败: {str(e)}")

def compute_similarity_batch(text, text_list):
    """
    计算一个文本与一组文本的相似度。
    :param text: 输入文本
    :param text_list: 文本列表
    :return: 相似度分数列表
    """
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    if not text_list:
        logger.error("文本列表为空")
        raise ValueError("文本列表不能为空")

    try:
        logger.info("生成输入文本的嵌入向量")

        # 获取嵌入生成器实例
        embedding_generator = get_embedding_generator()

        # 强制使用 CPU
        with torch.no_grad():
            embedding1 = embedding_generator.generate_embeddings(text, normalize=True).cpu()

            logger.info("生成文本列表的嵌入向量")
            embeddings2 = embedding_generator.generate_embeddings(text_list, normalize=True).cpu()

        logger.info("计算文本批量相似度")
        similarities = util.pytorch_cos_sim(embedding1, embeddings2).squeeze().tolist()
        logger.info(f"批量相似度: {similarities}")
        return similarities
    except Exception as e:
        logger.error(f"批量计算文本相似度失败: {e}")
        raise RuntimeError(f"批量计算文本相似度失败: {str(e)}")