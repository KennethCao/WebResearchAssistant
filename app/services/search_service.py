from typing import List, Dict, Any
import numpy as np
from app.models.paper import Paper
from app.nlp.blockchain_analyzer import BlockchainAnalyzer
from app.utils.logger import setup_logger
from app.utils.cache_manager import cached

logger = setup_logger(__name__)

class SearchService:
    def __init__(self):
        self.analyzer = BlockchainAnalyzer()
    
    @cached(timeout=3600)
    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相关论文"""
        try:
            # 生成查询向量
            query_embedding = self.analyzer.embedding_model.encode(query)
            
            # 获取所有论文
            papers = Paper.query.filter_by(analysis_status='completed').all()
            if not papers:
                return []
                
            # 计算相似度
            similarities = []
            for paper in papers:
                if paper.vector_embedding:
                    paper_embedding = np.frombuffer(paper.vector_embedding)
                    similarity = np.dot(query_embedding, paper_embedding)
                    similarities.append((paper, similarity))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回结果
            results = []
            for paper, similarity in similarities[:limit]:
                result = paper.to_dict()
                result['similarity'] = float(similarity)
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    @cached(timeout=3600)
    def search_by_topic(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """按主题搜索论文"""
        try:
            papers = Paper.query.filter(
                Paper.blockchain_aspects.contains([{'aspect': topic}])
            ).limit(limit).all()
            
            return [paper.to_dict() for paper in papers]
            
        except Exception as e:
            logger.error(f"Topic search failed: {e}")
            return []
    
    @cached(timeout=3600)
    def get_similar_papers(self, paper_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """获取相似论文"""
        try:
            # 获取目标论文
            paper = Paper.query.get(paper_id)
            if not paper or not paper.vector_embedding:
                return []
                
            # 获取目标论文的向量
            paper_embedding = np.frombuffer(paper.vector_embedding)
            
            # 获取其他论文
            other_papers = Paper.query.filter(
                Paper.id != paper_id,
                Paper.analysis_status == 'completed'
            ).all()
            
            # 计算相似度
            similarities = []
            for other in other_papers:
                if other.vector_embedding:
                    other_embedding = np.frombuffer(other.vector_embedding)
                    similarity = np.dot(paper_embedding, other_embedding)
                    similarities.append((other, similarity))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 返回结果
            results = []
            for similar_paper, similarity in similarities[:limit]:
                result = similar_paper.to_dict()
                result['similarity'] = float(similarity)
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Failed to get similar papers: {e}")
            return [] 