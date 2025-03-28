# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from app.utils.logger import setup_logger
import os
from .routes.upload import upload_bp  # 导入上传蓝图

# 初始化扩展
db = SQLAlchemy()
migrate = Migrate()
logger = setup_logger(__name__)

def create_app():
    """创建 Flask 应用"""
    try:
        logger.info("Starting to create Flask application...")
        app = Flask(__name__)
        
        # 配置应用
        app.config.update(
            SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
            SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(app.instance_path, 'app.db')}",
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
            UPLOAD_FOLDER=os.path.join(os.getenv('BLOCKCHAIN_ASSISTANT_DATA_DIR', 'data'), 'uploads')
        )
        
        # 确保目录存在
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # 初始化扩展
        db.init_app(app)
        migrate.init_app(app, db)  # 添加迁移初始化
        
        # 注册蓝图
        with app.app_context():
            # 确保数据库表存在
            db.create_all()
            
            from app.routes import web, api
            app.register_blueprint(web.bp)
            app.register_blueprint(api.bp, url_prefix='/api')
        
        logger.info("Flask application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise
