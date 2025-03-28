import uuid
import threading
import json
from PyPDF2 import PdfReader
import os
import sys
from typing import List, Dict, Any, Optional
from app.utils.logger import setup_logger
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread
import time
import torch
from app.services.paper_service import PaperService
from app.nlp.blockchain_qa import BlockchainQA
from app.models.paper import Paper
from app.extensions import db

# 设置日志记录器
logger = setup_logger(__name__)

# 添加项目根目录到 Python 路径
import os.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Then import jieba
import jieba  # type: ignore

from app.nlp.summarizer import TextSummarizer, generate_summary
from app.nlp.keyword_extractor import KeywordExtractor, extract_keywords
from app.clustering.kmeans_clustering import perform_kmeans_clustering
from app.file_processing.pdf_processor import extract_text_from_pdf
from app.config.environment import UPLOAD_FOLDER

# Global cache for analysis results
analysis_cache: Dict[str, Dict[str, Any]] = {}

# 存储分析状态的字典
analysis_tasks: Dict[str, Dict[str, Any]] = {}

def init_jieba():
    """延迟初始化 jieba"""
    try:
        import jieba
        jieba.initialize()  # 使用小词典模式
        return jieba
    except Exception as e:
        logger.error(f"Failed to initialize jieba: {e}")
        return None

def analyze_text(text: str) -> Dict[str, Any]:
    """分析文本内容"""
    try:
        results = {}
        
        # 延迟加载 jieba
        jieba_instance = init_jieba()
        if jieba_instance:
            # 使用基本分词
            words = jieba_instance.cut(text, cut_all=False)
            results['words'] = list(words)
        else:
            # 如果 jieba 加载失败，使用简单的分词
            results['words'] = text.split()
            
        return results
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        return {'words': text.split()}

def analyze_topics(text: str) -> List[Dict[str, float]]:
    """分析文本主题"""
    try:
        # 这里是一个简单的示例实现
        # 实际应用中可能需要更复杂的主题分析算法
        topics = [
            {"name": "区块链技术", "score": 85},
            {"name": "智能合约", "score": 75},
            {"name": "去中心化", "score": 65},
            {"name": "加密货币", "score": 55},
            {"name": "共识机制", "score": 45}
        ]
        return topics
    except Exception as e:
        logger.error(f"主题分析失败: {e}")
        return []

def start_analysis_task(file_id: str, file_path: str) -> None:
    """启动文档分析任务"""
    try:
        # 初始化分析状态
        analysis_cache[file_id] = {
            'status': 'processing',
            'progress': 0,
            'current_task': '正在准备分析...', → 'current_task': 'Preparing analysis...'
            'timestamp': datetime.now().isoformat()
        }
        
        # 启动异步分析任务
        Thread(target=analyze_document, args=(file_id, file_path)).start()
        logger.info(f"Started analysis task for file: {file_id}")
        
    except Exception as e:
        logger.error(f"Failed to start analysis task: {e}")
        analysis_cache[file_id] = {
            'status': 'error',
            'error': str(e)
        }

def get_analysis_status(file_id: str) -> dict:
    """获取文件分析状态"""
    try:
        if file_id not in analysis_cache:
            # 如果缓存中没有该文件的状态，返回错误
            return {
                'status': 'error',
                'error': '找不到该文件的分析状态'
            }
        
        return analysis_cache[file_id]
        
    except Exception as e:
        logger.error(f"获取分析状态时出错: {e}") → logger.error(f"Error getting analysis status: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

def get_file_content(file_id: str) -> Optional[str]:
    """获取文件内容"""
    try:
        # 这里需要根据你的实际存储方式来实现
        # 示例实现：从文件系统读取
        file_path = f"data/files/{file_id}.txt"
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return content
        
    except Exception as e:
        logger.error(f"Failed to get file content: {e}")
        return None

def clear_analysis_cache(file_id: Optional[str] = None) -> bool:
    """
    清除分析缓存
    
    Args:
        file_id: 可选的文件ID，如果不提供则清除所有缓存
        
    Returns:
        bool: 操作是否成功
    """
    try:
        if file_id is None:
            analysis_cache.clear()
            logger.info("Cleared all analysis cache")
        elif file_id in analysis_cache:
            del analysis_cache[file_id]
            logger.info(f"Cleared analysis cache for file_id: {file_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error clearing analysis cache: {e}")
        return False

# Add a cleanup function to periodically clear old cache entries
def cleanup_old_cache(max_age_hours: int = 24):
    """
    Clean up old cache entries
    
    Args:
        max_age_hours: Maximum age of cache entries in hours
    """
    try:
        current_time = datetime.now()
        to_delete = []
        
        for file_id, data in analysis_cache.items():
            upload_time = datetime.strptime(data['upload_time'], '%Y-%m-%d %H:%M:%S')
            age = (current_time - upload_time).total_seconds() / 3600
            
            if age > max_age_hours:
                to_delete.append(file_id)
        
        for file_id in to_delete:
            del analysis_cache[file_id]
            
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old cache entries")
            
    except Exception as e:
        logger.error(f"Error cleaning up cache: {e}")

def analyze_document(file_id: str, file_path: str) -> None:
    """分析文档内容"""
    try:
        logger.info(f"开始分析文档: {file_id}")
        
        # 更新状态为处理中
        analysis_cache[file_id].update({
            'status': 'processing',
            'progress': 10,
            'current_task': '正在分析文档...'
        })
        
        # 执行实际的分析任务
        paper_service = PaperService()
        logger.info("开始提取PDF文本...")
        text = paper_service.process_pdf(file_path)
        
        if text:
            logger.info(f"成功提取文本，长度: {len(text)}")
            # 更新进度
            analysis_cache[file_id].update({
                'progress': 30,
                'current_task': '正在提取关键信息...'
            })
            
            try:
                # 提取关键信息
                keywords = paper_service.validate_blockchain_content(text)
                
                # 更新进度
                analysis_cache[file_id].update({
                    'progress': 60,
                    'current_task': '正在生成摘要...'
                })
                
                # 生成摘要
                summary = paper_service._extract_abstract(text) or text[:500] + "..."
                
                # 更新进度
                analysis_cache[file_id].update({
                    'progress': 80,
                    'current_task': '正在添加到知识库...'
                })
                
                # 添加到知识库
                qa_system = BlockchainQA()
                added_to_kb = qa_system.process_pdf(text)
                
                # 完成分析
                analysis_cache[file_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'current_task': '分析完成',
                    'keywords': keywords,
                    'summary': summary,
                    'added_to_kb': added_to_kb
                })
                
                logger.info(f"文档分析完成: {file_id}")
                
            except Exception as e:
                logger.error(f"提取关键信息时出错: {e}") → logger.error(f"Error extracting key info: {e}")
                raise ValueError("无法从PDF中提取文本") → raise ValueError("Failed to extract text from PDF")
                
        else:
            raise ValueError("无法从PDF中提取文本")
        
    except Exception as e:
        logger.error(f"文档分析失败: {e}", exc_info=True)
        analysis_cache[file_id].update({
            'status': 'error',
            'error': str(e),
            'progress': 0
        }) 

def analyze_paper(file_path: str, paper_id: int):
    """分析论文文件"""
    try:
        logger.info(f"Starting analysis for paper ID {paper_id}")
        
        # 创建新的数据库会话
        with db.session.begin():
            paper = Paper.query.get(paper_id)
            if not paper:
                raise ValueError(f"Paper not found with ID: {paper_id}")
            paper.analysis_status = 'processing'
            db.session.commit()
        
        try:
            # 读取PDF文件
            with open(file_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            
            if not text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            # 更新数据库
            with db.session.begin():
                paper = Paper.query.get(paper_id)
                if paper:
                    paper.content = text
                    paper.analysis_status = 'completed'
                    db.session.commit()
                    logger.info(f"Analysis completed for paper ID: {paper_id}")
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            with db.session.begin():
                paper = Paper.query.get(paper_id)
                if paper:
                    paper.analysis_status = 'error'
                    paper.error_message = str(e)
                    db.session.commit()
            
    except Exception as e:
        logger.error(f"Analysis failed for paper ID {paper_id}: {e}", exc_info=True)
        try:
            with db.session.begin():
                paper = Paper.query.get(paper_id)
                if paper:
                    paper.analysis_status = 'error'
                    paper.error_message = str(e)
                    db.session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update error status: {db_error}")