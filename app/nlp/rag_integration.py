import os
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

# 模型和检索存储路径
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "facebook/bart-large-cnn"
INDEX_DIR = "./models/faiss_index"

# 全局变量
embedding_model = None
generation_model = None
tokenizer = None
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

def load_generation_model():
    """
    加载生成模型及其分词器。
    """
    global generation_model, tokenizer
    if generation_model is None or tokenizer is None:
        try:
            logger.info(f"加载生成模型: {GENERATION_MODEL_NAME}")
            generation_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME).to("cpu")  # 使用 CPU
            tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        except Exception as e:
            logger.error(f"加载生成模型失败: {e}")
            raise RuntimeError(f"加载生成模型失败: {str(e)}")

def load_or_create_faiss_index(dimension=384):
    """
    加载或创建 FAISS 索引。
    :param dimension: 嵌入向量的维度（默认 384）。
    :return: FAISS 索引
    """
    global index
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    if os.path.exists(index_path):
        try:
            logger.info("加载现有的 FAISS 索引")
            index = faiss.read_index(index_path)
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

def add_to_index(texts):
    """
    将文本嵌入向量添加到 FAISS 索引。
    :param texts: 文本列表
    """
    if index is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    if embedding_model is None:
        load_embedding_model()

    try:
        logger.info("生成文本嵌入向量")
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)  # 不转换为张量，加速 CPU 处理
        index.add(embeddings)
        logger.info(f"已向索引中添加 {len(texts)} 条记录")
    except Exception as e:
        logger.error(f"向索引中添加记录失败: {e}")
        raise RuntimeError(f"向索引中添加记录失败: {str(e)}")

def search_index(query, top_k=5):
    """
    在 FAISS 索引中搜索与查询最相似的文本。
    :param query: 查询文本
    :param top_k: 返回的最相似记录数量
    :return: 最相似记录的索引和距离
    """
    if index is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    if embedding_model is None:
        load_embedding_model()

    try:
        logger.info("生成查询嵌入向量")
        query_embedding = embedding_model.encode([query], convert_to_tensor=False)
        distances, indices = index.search(query_embedding, top_k)
        return indices[0], distances[0]
    except Exception as e:
        logger.error(f"搜索索引失败: {e}")
        raise RuntimeError(f"搜索索引失败: {str(e)}")

def generate_response(query, top_k=5):
    """
    基于查询和检索的上下文生成响应。
    :param query: 查询文本
    :param top_k: 检索的最相似记录数量
    :return: 生成的响应文本
    """
    if not query:
        logger.error("查询文本为空")
        raise ValueError("查询文本不能为空")

    if generation_model is None or tokenizer is None:
        load_generation_model()

    try:
        logger.info("检索相关上下文")
        indices, distances = search_index(query, top_k=top_k)
        context = "\n".join([f"相关文档 {i}: 距离 {distances[i]}" for i in indices])

        logger.info("生成响应")
        input_text = f"上下文: {context}\n问题: {query}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to("cpu")  # 确保使用 CPU
        outputs = generation_model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response
    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        raise RuntimeError(f"生成响应失败: {str(e)}")

def save_index():
    """
    将 FAISS 索引保存到磁盘。
    """
    if index is None:
        logger.error("FAISS 索引未加载")
        raise RuntimeError("FAISS 索引未加载")

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    try:
        faiss.write_index(index, index_path)
        logger.info(f"FAISS 索引已保存到: {index_path}")
    except Exception as e:
        logger.error(f"保存 FAISS 索引失败: {e}")
        raise RuntimeError(f"保存 FAISS 索引失败: {str(e)}")