import os
import faiss
import torch
from sentence_transformers import SentenceTransformer
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

# 模型和检索存储路径
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "./models/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")

# 全局变量
embedding_model = None
index = None

def load_embedding_model():
    """
    加载嵌入向量生成模型。
    """
    global embedding_model
    if embedding_model is None:
        try:
            logger.info(f"加载嵌入向量模型: {EMBEDDING_MODEL_NAME}")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")  # 使用 CPU
        except Exception as e:
            logger.error(f"加载嵌入向量模型失败: {e}")
            raise RuntimeError(f"加载嵌入向量模型失败: {str(e)}")

def load_or_create_faiss_index(dimension=384):
    """
    加载或创建 FAISS 索引。
    :param dimension: 嵌入向量的维度（默认 384）
    :return: FAISS 索引
    """
    global index
    os.makedirs(INDEX_DIR, exist_ok=True)
    if os.path.exists(INDEX_FILE):
        try:
            logger.info("加载现有的 FAISS 索引")
            index = faiss.read_index(INDEX_FILE)
        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {e}")
            raise RuntimeError(f"加载 FAISS 索引失败: {str(e)}")
    else:
        try:
            logger.info("创建新的 FAISS 索引")
            index = faiss.IndexFlatL2(dimension)
        except Exception as e:
            logger.error(f"创建 FAISS 索引失败: {e}")
            raise RuntimeError(f"创建 FAISS 索引失败: {str(e)}")
    return index

def add_to_index(documents):
    """
    将文档嵌入向量添加到 FAISS 索引。
    :param documents: 文档列表
    """
    if not documents:
        logger.error("文档列表为空")
        raise ValueError("文档列表不能为空")

    if not all(isinstance(doc, str) for doc in documents):
        logger.error("文档列表中的每个元素必须是字符串")
        raise TypeError("文档列表中的每个元素必须是字符串")

    if index is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    if embedding_model is None:
        load_embedding_model()

    try:
        logger.info("生成文档嵌入向量")
        embeddings = embedding_model.encode(documents, convert_to_tensor=False)  # 不转换为张量，加速 CPU 处理
        index.add(embeddings)
        logger.info(f"已向索引中添加 {len(documents)} 条记录")
    except Exception as e:
        logger.error(f"向索引中添加记录失败: {e}")
        raise RuntimeError(f"向索引中添加记录失败: {str(e)}")

def search_documents(query, top_k=5, use_gpu=False):
    """
    在 FAISS 索引中搜索与查询最相似的文档。
    :param query: 查询文本
    :param top_k: 返回的最相似文档数量
    :param use_gpu: 是否使用 GPU（默认 False）
    :return: 最相似文档的索引和距离
    """
    if not query:
        logger.error("查询文本为空")
        raise ValueError("查询文本不能为空")

    if top_k <= 0:
        logger.error("检索的文档数量必须大于 0")
        raise ValueError("检索的文档数量必须大于 0")

    if index is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    if embedding_model is None:
        load_embedding_model()

    try:
        logger.info("生成查询嵌入向量")
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        distances, indices = index.search(query_embedding, top_k)
        logger.info(f"检索到 {top_k} 个最相似的文档")
        return indices[0], distances[0]
    except Exception as e:
        logger.error(f"搜索索引失败: {e}")
        raise RuntimeError(f"搜索索引失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 初始化检索器和索引
    load_embedding_model()
    load_or_create_faiss_index()

    # 示例文档
    documents = [
        "区块链技术在金融领域的应用越来越广泛。",
        "智能合约是区块链技术的核心组成部分。",
        "区块链技术提供了去中心化的解决方案。",
        "区块链技术的安全性和透明性是其主要优势。"
    ]

    # 添加文档到索引
    add_to_index(documents)

    # 示例查询
    query = "区块链技术的发展趋势"
    indices, distances = search_documents(query, top_k=3)
    print("检索到的文档索引:", indices)
    print("检索到的文档距离:", distances)