import os
from app import create_app, db
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def reset_database():
    """重置数据库"""
    try:
        app = create_app()
        with app.app_context():
            # 删除数据库文件
            db_path = os.path.join(app.instance_path, 'app.db')
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.info(f"Removed existing database: {db_path}")
            
            # 创建新的数据库和表
            db.create_all()
            logger.info("Created new database and tables")
            
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise

if __name__ == "__main__":
    reset_database() 