import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 从本地 utils 包导入
from utils import setup_script_logger

logger = setup_script_logger(__name__)

def check_database():
    """检查数据库状态"""
    try:
        from app import create_app, db
        
        app = create_app()
        with app.app_context():
            # 检查数据库连接
            db.engine.connect()
            logger.info("Database connection successful")
            
            # 检查必要的表是否存在
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            logger.info(f"Found tables: {tables}")
            
            # 验证数据库文件权限
            db_path = os.path.join(app.instance_path, 'app.db')
            if os.path.exists(db_path):
                logger.info(f"Database file exists at: {db_path}")
                # 检查文件权限
                try:
                    with open(db_path, 'a'):
                        pass
                    logger.info("Database file is writable")
                except IOError as e:
                    logger.error(f"Database file permission error: {e}")
                    return False
            else:
                logger.warning("Database file does not exist")
            
            return True
            
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False

if __name__ == "__main__":
    if check_database():
        print("Database check passed")
        sys.exit(0)
    else:
        print("Database check failed")
        sys.exit(1) 