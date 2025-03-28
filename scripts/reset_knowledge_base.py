import os
import shutil
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def reset_knowledge_base():
    """重置知识库"""
    try:
        knowledge_base_dir = os.path.join('app', 'static', 'knowledge_base')
        if os.path.exists(knowledge_base_dir):
            shutil.rmtree(knowledge_base_dir)
            os.makedirs(knowledge_base_dir)
            logger.info("Knowledge base reset")
        else:
            os.makedirs(knowledge_base_dir)
            logger.info("Knowledge base directory created")
    except Exception as e:
        logger.error(f"Error resetting knowledge base: {str(e)}")

if __name__ == "__main__":
    reset_knowledge_base()