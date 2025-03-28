from app.rag.retriever import search_documents
from app.rag.generator import generate_text
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def retrieve_and_generate_response(query, top_k=5):
    """
    执行检索增强生成（RAG）。
    :param query: 用户查询文本
    :param top_k: 检索的最相似文档数量
    :return: 基于上下文生成的回答
    """
    if not query:
        logger.error("查询文本为空")
        raise ValueError("查询文本不能为空")

    if top_k <= 0:
        logger.error("检索的文档数量必须大于 0")
        raise ValueError("检索的文档数量必须大于 0")

    try:
        logger.info(f"开始处理查询: {query}")

        # 检索相关文档
        logger.info("检索相关文档")
        indices, distances = search_documents(query, top_k=top_k)
        logger.info(f"检索到的文档索引: {indices}, 相似度: {distances}")

        # 获取文档内容（根据项目数据集的实际实现）
        # 这里假设有一个函数 get_document_content 从文档存储中获取具体内容
        def get_document_content(indices):
            # 示例：从文档存储中获取具体内容
            # 这里使用占位符文档内容
            documents = [f"文档内容 {i}" for i in indices]
            return documents

        documents = get_document_content(indices)

        # 基于上下文生成回答
        logger.info("基于上下文生成回答")
        response = generate_text(query, documents)
        logger.info("回答生成完成")
        return response
    except Exception as e:
        logger.error(f"检索与生成失败: {e}")
        raise RuntimeError(f"检索与生成失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例查询
    query = "区块链技术的发展趋势"

    # 执行检索增强生成
    response = retrieve_and_generate_response(query, top_k=3)
    print("生成的回答:", response)