from celery import shared_task
from datetime import datetime
import numpy as np
from app.models.paper import Paper
from app.nlp.blockchain_analyzer import BlockchainAnalyzer
from app.nlp.text_extractor import TextExtractor
from app.extensions import db
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

@shared_task
def analyze_paper(paper_id: int):
    """异步分析论文任务"""
    try:
        # 获取论文
        paper = Paper.query.get(paper_id)
        if not paper:
            logger.error(f"Paper not found: {paper_id}")
            return False
            
        # 更新状态
        paper.analysis_status = 'processing'
        db.session.commit()
        
        # 初始化分析器
        analyzer = BlockchainAnalyzer()
        extractor = TextExtractor()
        
        try:
            # 提取文本
            text_data = extractor.extract_from_pdf(paper.file_path)
            
            # 分析论文
            results = analyzer.analyze_blockchain_paper(text_data)
            
            # 更新论文信息
            paper.title = text_data.get('title')
            paper.abstract = text_data.get('abstract')
            paper.keywords = text_data.get('keywords')
            paper.content = text_data.get('full_text')
            
            # 更新分析结果
            paper.summary = results.get('summary')
            paper.key_findings = results.get('key_findings')
            paper.methodology = results.get('methodology')
            paper.innovations = results.get('innovations')
            paper.technical_details = results.get('technical_details')
            paper.blockchain_aspects = results.get('blockchain_aspects')
            paper.challenges = results.get('challenges')
            
            # 保存向量嵌入
            if 'embedding' in results:
                paper.vector_embedding = results['embedding'].tobytes()
            
            # 更新状态
            paper.analysis_status = 'completed'
            paper.analyzed_at = datetime.utcnow()
            
            db.session.commit()
            logger.info(f"Successfully analyzed paper: {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed for paper {paper_id}: {e}")
            paper.analysis_status = 'failed'
            db.session.commit()
            return False
            
    except Exception as e:
        logger.error(f"Task failed for paper {paper_id}: {e}")
        return False

@shared_task
def update_paper_embeddings():
    """更新所有论文的向量嵌入"""
    try:
        analyzer = BlockchainAnalyzer()
        papers = Paper.query.filter_by(analysis_status='completed').all()
        
        for paper in papers:
            try:
                # 生成新的嵌入
                text = f"{paper.title} {paper.abstract} {paper.keywords}"
                embedding = analyzer.embedding_model.encode(text)
                paper.vector_embedding = embedding.tobytes()
                
            except Exception as e:
                logger.error(f"Failed to update embedding for paper {paper.id}: {e}")
                continue
                
        db.session.commit()
        logger.info("Successfully updated paper embeddings")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update paper embeddings: {e}")
        return False 