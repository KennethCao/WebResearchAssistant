from app.utils.logger import setup_logger
from app.models.paper import Paper
from app.nlp.text_analyzer import TextAnalyzer
from app.extensions import db

logger = setup_logger(__name__)

def analyze_paper(paper_id: int) -> bool:
    """分析论文内容
    
    Args:
        paper_id: 论文ID
        
    Returns:
        bool: 分析是否成功
    """
    try:
        # 获取论文记录
        paper = Paper.query.get(paper_id)
        if not paper:
            logger.error(f"Paper not found: {paper_id}")
            return False
            
        # 更新分析状态
        paper.analysis_status = 'processing'
        db.session.commit()
        
        # 初始化文本分析器
        analyzer = TextAnalyzer()
        
        try:
            # 提取关键信息
            paper.title = analyzer.extract_title(paper.content)
            paper.summary = analyzer.generate_summary(paper.content)
            paper.keywords = analyzer.extract_keywords(paper.content)
            paper.topics = analyzer.extract_topics(paper.content)
            
            # 更新状态
            paper.analysis_status = 'completed'
            paper.error_message = None
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            paper.analysis_status = 'failed'
            paper.error_message = str(e)
            
        # 保存更改
        db.session.commit()
        return paper.analysis_status == 'completed'
        
    except Exception as e:
        logger.error(f"Failed to analyze paper: {str(e)}")
        return False

# 确保函数可以被导入
__all__ = ['analyze_paper'] 