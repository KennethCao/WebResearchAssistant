from typing import List, Dict
import faiss
import numpy as np
from .base_retriever import BaseRetriever
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class HierarchicalFAISSRetriever(BaseRetriever):
    def __init__(self, index_path: str = None, nlist: int = 100, m: int = 32):
        self.index = None
        self.nlist = nlist
        self.m = m
        self.index_path = index_path
        self.dimension = 768  # 与嵌入维度匹配

    def _build_index(self, embeddings: np.ndarray):
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
        self.index.train(embeddings)
        self.index.add(embeddings)

    def add_documents(self, documents: List[Dict]):
        embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
        if not self.index:
            self._build_index(embeddings)
        else:
            self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        distances, indices = self.index.search(query_embedding, k)
        return [{
            'document': self.documents[idx],
            'score': float(1 - distance),
            'match_type': 'hierarchical'
        } for distance, idx in zip(distances[0], indices[0])]

    def save_index(self):
        if self.index_path:
            faiss.write_index(self.index, self.index_path)

    def load_index(self):
        if self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)