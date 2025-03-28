import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import create_app, db
from flask_migrate import Migrate, upgrade
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def setup_database():
    """Setup database"""
    try:
        app = create_app()
        
        with app.app_context():
            # Ensure instance directory exists
            os.makedirs(app.instance_path, exist_ok=True)
            
            # Initialize Flask-Migrate
            Migrate(app, db)
            
            # Create database directory
            db_dir = os.path.dirname(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
            os.makedirs(db_dir, exist_ok=True)
            
            # Create database tables
            logger.info("Creating database tables...")
            db.create_all()
            
            # 运行迁移并添加事务管理
            try:
                logger.info("Running database migrations...")
                db.session.begin()
                upgrade()
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logger.error(f"Database migration failed: {e}")
                raise
            
            logger.info("Database setup completed successfully")
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_database()