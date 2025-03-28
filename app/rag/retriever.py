# retriever.py
import os
import faiss
import numpy as np
import torch
from app.utils.logger import setup_logger
from app.nlp.embedding_generator import EmbeddingGenerator, get_embedding_generator
from app.config.environment import (
    FAISS_INDEX_PATH,
    BLOCKCHAIN_EMBEDDING_DIMENSION
)

# 设置日志记录器
logger = setup_logger(__name__)

class DocumentRetriever:
    """
    文档检索器类。
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentRetriever, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 从环境变量获取应用数据目录
        app_data_dir = os.getenv('BLOCKCHAIN_ASSISTANT_DATA_DIR')
        if not app_data_dir:
            # 使用默认目录
            app_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
            os.environ['BLOCKCHAIN_ASSISTANT_DATA_DIR'] = app_data_dir
        
        # 使用应用数据目录中的 faiss 索引路径
        self.index_path = os.path.join(app_data_dir, 'cache', 'faiss', 'index.faiss')
        
        # 如果已经初始化过，直接返回
        if self._initialized:
            return
            
        try:
            # 初始化索引
            self.index = None
            self.load_index()
            
            logger.info("Document retriever initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize document retriever: {e}")
            raise
    
    def load_index(self):
        """加载或创建 FAISS 索引"""
        try:
            # 确保索引目录存在
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    logger.info("Loaded existing FAISS index")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}")
            
            # 创建新索引
            self.create_new_index()
            
        except Exception as e:
            logger.error(f"Failed to load/create index: {e}")
            raise
    
    def create_new_index(self):
        """创建新的 FAISS 索引"""
        try:
            # 创建索引
            self.index = faiss.IndexFlatL2(BLOCKCHAIN_EMBEDDING_DIMENSION)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # 保存索引
            faiss.write_index(self.index, self.index_path)
            logger.info("Created and saved new FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to create new index: {e}")
            raise
    
    def add_documents(self, embeddings: np.ndarray):
        """
        将文档向量添加到索引中。
        
        Args:
            embeddings (np.ndarray): 文档向量数组
        """
        try:
            if self.index is None:
                raise RuntimeError("Index not initialized")
                
            # 确保输入是 numpy 数组
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            # 添加向量到索引
            self.index.add(embeddings)
            
            # 保存更新后的索引
            faiss.write_index(self.index, self.index_path)
            
            logger.info(f"Added {len(embeddings)} vectors to index")
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5):
        """
        搜索最相似的文档。
        """
        try:
            if self.index is None:
                self.load_index()  # 确保索引已加载
            
            if self.index.ntotal == 0:
                # 如果索引为空，返回空结果而不是错误
                logger.warning("Search called on empty index")
                return np.array([]), np.array([])
            
            # 确保查询向量是正确的形状
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # 执行搜索
            distances, indices = self.index.search(query_vector.astype('float32'), k)
            
            logger.info(f"Search completed, found {len(indices[0])} results")
            return distances, indices
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            raise

class FAISSRetriever:
    def __init__(self, dimension, index_path, create_new=False):
        self.dimension = dimension
        self.index_path = index_path
        
        try:
            if create_new:
                # 创建新索引
                self.index = faiss.IndexFlatL2(dimension)
                # 保存索引
                self._save_index()
            else:
                # 加载现有索引
                self._load_index()
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
            
    def _save_index(self):
        """保存索引到文件"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"FAISS index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            raise
            
    def _load_index(self):
        """从文件加载索引"""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"FAISS index loaded from {self.index_path}")
            else:
                logger.warning(f"Index file not found at {self.index_path}, creating new index")
                self.index = faiss.IndexFlatL2(self.dimension)
                self._save_index()
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
    
    def search(self, query: str, k: int = 5):
        """
        搜索相似向量
        """
        try:
            if self.index is None:
                raise RuntimeError("FAISS index not initialized")
            
            # 获取查询的嵌入向量
            embedding_generator = get_embedding_generator()
            query_vector = embedding_generator.generate_embedding(query)
            query_np = query_vector.cpu().numpy().reshape(1, -1)
            
            # 执行搜索
            distances, indices = self.index.search(query_np, k)
            return distances, indices
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise

# 示例用法
if __name__ == "__main__":
    # 创建检索器实例
    retriever = DocumentRetriever()
    
    # 创建一些测试向量
    test_vectors = np.random.random((10, BLOCKCHAIN_EMBEDDING_DIMENSION)).astype('float32')
    
    # 添加向量到索引
    retriever.add_documents(test_vectors)
    
    # 测试搜索
    query = np.random.random((1, BLOCKCHAIN_EMBEDDING_DIMENSION)).astype('float32')
    distances, indices = retriever.search(query, k=3)
    
    print("Search results:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        print(f"{i}. Distance: {dist:.4f}, Index: {idx}")