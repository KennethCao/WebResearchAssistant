import os
from app.utils.logger import setup_logger
from scripts.reset_knowledge_base import reset_knowledge_base
from scripts.init_db import init_database
from scripts.process_existing_pdfs import process_existing_pdfs
from scripts.check_knowledge_base import check_knowledge_base

logger = setup_logger(__name__)

def initialize_all():
    """初始化所有组件"""
    try:
        logger.info("Initialization started...")
        logger.info("Step 1: Reset knowledge base")
        reset_knowledge_base()
        logger.info("Step 2: Initialize database")
        init_database()
        logger.info("Step 3: Process PDF files")
        process_existing_pdfs()
        logger.info("Step 4: Verify knowledge base")
        check_knowledge_base()
        logger.info("Initialization completed!")
    except Exception as e:
        logger.error(f"初始化过程出错: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_all()