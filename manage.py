import os
import sys
from pathlib import Path
from flask import Flask
try:
    try:
        from flask_sqlalchemy import SQLAlchemy
        from flask_migrate import Migrate
    except ImportError:
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flask_sqlalchemy"])
            from flask_sqlalchemy import SQLAlchemy
            from flask_migrate import Migrate
        except subprocess.CalledProcessError:
            print("安装 flask_sqlalchemy 失败，请手动安装。")
            raise
        except ImportError:
            print("导入 flask_sqlalchemy 仍然失败，请检查安装情况。")
            import site
            print(f"Python 站点包目录: {site.getsitepackages()}")
            print(f"当前 Python 环境: {sys.executable}")
            raise
except ImportError:
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask_sqlalchemy"])
        from flask_sqlalchemy import SQLAlchemy
        from flask_migrate import Migrate
    except subprocess.CalledProcessError:
        print("安装 flask_sqlalchemy 失败，请手动安装。")
        raise
    except ImportError:
        print("导入 flask_sqlalchemy 仍然失败，请检查安装情况。可能需要确认pip源是否可用，或者Python环境是否正确配置。")
        import site
        print(f"Python 站点包目录: {site.getsitepackages()}")
        print(f"当前 Python 环境: {sys.executable}")
        raise
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask_sqlalchemy"])
        # 再次尝试导入
        from flask_sqlalchemy import SQLAlchemy
        from flask_migrate import Migrate
    except subprocess.CalledProcessError:
        print("安装 flask_sqlalchemy 失败，请手动安装。")
        raise
    except ImportError:
        print("导入 flask_sqlalchemy 仍然失败，请检查安装情况。可能需要确认pip源是否可用，或者Python环境是否正确配置。")
        # 可以添加更多调试信息
        import site
        print(f"Python 站点包目录: {site.getsitepackages()}")
        print(f"当前 Python 环境: {sys.executable}")
        raise
from app.utils.logger import setup_logger

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

logger = setup_logger(__name__)

def create_minimal_app():
    """创建最小化的 Flask 应用"""
    app = Flask(__name__)
    
    # 配置
    app.config.update(
        SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', ''),
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{os.path.join(ROOT_DIR, "instance", "app.db")}',
        SQLALCHEMY_TRACK_MODIFICATIONS=False
    )
    
    # 检查密钥配置
    if not app.config['SECRET_KEY']:
        raise ValueError('FLASK_SECRET_KEY environment variable must be set in production')
    
    return app

def init_db():
    """初始化数据库"""
    try:
        app = create_minimal_app()
        db = SQLAlchemy(app)
        migrate = Migrate(app, db)
        
        # 定义模型
        class Paper(db.Model):
            __tablename__ = 'papers'
            
            id = db.Column(db.Integer, primary_key=True)
            filename = db.Column(db.String(255), nullable=False)
            title = db.Column(db.String(500))
            summary = db.Column(db.Text)
            keywords = db.Column(db.Text)
            topics = db.Column(db.Text)
            content = db.Column(db.Text)
            analysis_status = db.Column(db.String(50))
            error_message = db.Column(db.Text)
            uploaded_at = db.Column(db.DateTime, default=db.func.current_timestamp())
            added_to_kb = db.Column(db.Boolean, default=False)
        
        with app.app_context():
            # 确保实例目录存在
            os.makedirs(os.path.join(ROOT_DIR, 'instance'), exist_ok=True)
            
            # 创建所有表
            db.create_all()
            db.session.commit()
            
            # 验证表是否创建成功
            if 'papers' in db.inspect(db.engine).get_table_names():
                logger.info("Database initialized successfully")
                return True
            
            logger.error("Failed to create papers table")
            return False
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def verify_db():
    """验证数据库结构"""
    try:
        app = create_minimal_app()
        db = SQLAlchemy(app)
        migrate = Migrate(app, db)
        
        with app.app_context():
            # 检查数据库文件是否存在
            db_path = os.path.join(ROOT_DIR, 'instance', 'app.db')
            if not os.path.exists(db_path):
                logger.error(f"Database file not found: {db_path}")
                return False
            
            # 检查表是否存在
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            logger.info(f"Found tables: {tables}")
            
            if 'papers' not in tables:
                logger.error("Papers table not found")
                return False
            
            # 验证表结构
            columns = [col['name'] for col in inspector.get_columns('papers')]
            logger.info(f"Papers table columns: {columns}")
            
            required_columns = [
                'id', 'filename', 'title', 'summary', 'keywords',
                'topics', 'content', 'analysis_status', 'error_message',
                'uploaded_at', 'added_to_kb'
            ]
            
            missing_columns = set(required_columns) - set(columns)
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            
            # 尝试执行简单查询
            try:
                db.session.query(Paper).count()
                db.session.commit()
                logger.info("Database query test passed")
            except Exception as e:
                logger.error(f"Database query test failed: {e}")
                return False
                
            logger.info("Database structure verified successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to verify database: {e}")
        return False

# 导出函数供 main.py 使用
setup_database = init_db

if __name__ == '__main__':
    if init_db():
        print("Database initialized successfully!")
        sys.exit(0)
    else:
        print("Database initialization failed!")
        sys.exit(1)