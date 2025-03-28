import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import create_app, db
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

5: # Add project root to Python path
15: """Initialize application"""
17: # Create application instance
21: # Delete existing database
27: # Create database directory
30: # Create all tables
def init_application():
    """初始化应用"""
    try:
        # 创建应用实例
        app = create_app()
        
        with app.app_context():
            # 删除现有数据库
            db_path = os.path.join(app.instance_path, 'app.db')
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"Removed existing database: {db_path}")
            
            # 创建数据库目录
            os.makedirs(app.instance_path, exist_ok=True)
            
            # 创建所有表
            db.create_all()
            logger.info("Database tables created successfully")
            
        logger.info("Application initialized successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_application()