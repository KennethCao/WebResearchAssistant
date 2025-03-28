# __init__.py
from app.utils.logger import setup_logger
from app.rag.retriever import DocumentRetriever

# 设置日志记录器
logger = setup_logger(__name__)

# 初始化组件
document_retriever = None

def initialize_rag():
    """
    初始化 RAG 组件。
    """
    try:
        global document_retriever
        
        # 初始化文档检索器
        if document_retriever is None:
            document_retriever = DocumentRetriever()
            logger.info("Document retriever initialized")
        
        logger.info("RAG components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise

def get_document_retriever() -> DocumentRetriever:
    """
    获取文档检索器实例。
    
    Returns:
        DocumentRetriever: 文档检索器实例
    """
    global document_retriever
    if document_retriever is None:
        initialize_rag()
    return document_retriever 