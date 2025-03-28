import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import create_app, db
from app.models.paper import Paper
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def init_database():
    """Initialize database"""
    try:
        # Create application instance
        app = create_app()
        
        with app.app_context():
            # Ensure instance directory exists
            os.makedirs(app.instance_path, exist_ok=True)
            
            # Remove existing database
            db_path = os.path.join(app.instance_path, 'app.db')
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"Removed existing database: {db_path}")
            
            # Create all tables
            db.create_all()
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

if __name__ == "__main__":
    init_database()