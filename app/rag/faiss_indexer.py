import os
import faiss
from sentence_transformers import SentenceTransformer
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

# 全局变量
INDEX_DIR = "./models/faiss_index"
FAISS_INDEX = None
EMBEDDING_MODEL = None
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_or_create_faiss_index(dimension=384):
    """
    加载或创建 FAISS 索引。
    :param dimension: 嵌入向量的维度（默认 384）
    :return: FAISS 索引对象
    """
    global FAISS_INDEX

    try:
        os.makedirs(INDEX_DIR, exist_ok=True)
        index_path = os.path.join(INDEX_DIR, "faiss.index")

        if os.path.exists(index_path):
            logger.info("加载现有的 FAISS 索引")
            FAISS_INDEX = faiss.read_index(index_path)
        else:
            logger.info("创建新的 FAISS 索引")
            FAISS_INDEX = faiss.IndexFlatL2(dimension)

        logger.info("FAISS 索引加载或创建完成")
        return FAISS_INDEX
    except Exception as e:
        logger.error(f"加载或创建 FAISS 索引失败: {e}")
        raise RuntimeError(f"加载或创建 FAISS 索引失败: {str(e)}")

def load_embedding_model():
    """
    加载嵌入向量生成模型。
    """
    global EMBEDDING_MODEL

    try:
        logger.info(f"加载嵌入模型: {EMBEDDING_MODEL_NAME}")
        # 强制使用 CPU
        EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("嵌入模型加载完成，运行于 CPU")
    except Exception as e:
        logger.error(f"加载嵌入模型失败: {e}")
        raise RuntimeError(f"加载嵌入模型失败: {str(e)}")

def add_to_index(texts):
    """
    将新文本的嵌入向量添加到索引中。
    :param texts: 文本列表
    """
    global FAISS_INDEX, EMBEDDING_MODEL

    if not texts:
        logger.error("文本列表为空")
        raise ValueError("文本列表不能为空")

    if not all(isinstance(doc, str) for doc in texts):
        logger.error("文本列表中的每个元素必须是字符串")
        raise TypeError("文本列表中的每个元素必须是字符串")

    if FAISS_INDEX is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    if EMBEDDING_MODEL is None:
        logger.warning("嵌入模型未加载，正在加载嵌入模型")
        load_embedding_model()

    try:
        logger.info("生成文本的嵌入向量")
        embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
        FAISS_INDEX.add(embeddings)
        logger.info(f"已向索引中添加 {len(texts)} 条记录")
    except Exception as e:
        logger.error(f"添加记录到索引失败: {e}")
        raise RuntimeError(f"添加记录到索引失败: {str(e)}")

def search_index(query, top_k=5):
    """
    在 FAISS 索引中检索与查询最相关的文本。
    :param query: 查询文本
    :param top_k: 返回的最相关记录数量
    :return: 检索结果的索引和距离
    """
    global FAISS_INDEX, EMBEDDING_MODEL

    if not query:
        logger.error("查询文本为空")
        raise ValueError("查询文本不能为空")

    if top_k <= 0:
        logger.error("检索的文档数量必须大于 0")
        raise ValueError("检索的文档数量必须大于 0")

    if FAISS_INDEX is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    if EMBEDDING_MODEL is None:
        logger.warning("嵌入模型未加载，正在加载嵌入模型")
        load_embedding_model()

    try:
        logger.info("生成查询文本的嵌入向量")
        query_embedding = EMBEDDING_MODEL.encode([query], show_progress_bar=True)

        logger.info("在 FAISS 索引中检索最相关的记录")
        distances, indices = FAISS_INDEX.search(query_embedding, top_k)
        return indices[0].tolist(), distances[0].tolist()
    except Exception as e:
        logger.error(f"检索失败: {e}")
        raise RuntimeError(f"检索失败: {str(e)}")

def save_index():
    """
    保存当前的 FAISS 索引到磁盘。
    """
    global FAISS_INDEX

    if FAISS_INDEX is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    try:
        index_path = os.path.join(INDEX_DIR, "faiss.index")
        faiss.write_index(FAISS_INDEX, index_path)
        logger.info(f"FAISS 索引已保存到: {index_path}")
    except Exception as e:
        logger.error(f"保存索引失败: {e}")
        raise RuntimeError(f"保存索引失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 初始化 FAISS 索引和嵌入模型
    load_or_create_faiss_index()
    load_embedding_model()

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
    indices, distances = search_index(query, top_k=3)
    print("检索到的文档索引:", indices)
    print("检索到的文档距离:", distances)

    # 保存索引
    save_index()