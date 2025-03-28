from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS

# 创建数据库实例
db = SQLAlchemy()
migrate = Migrate()
cors = CORS()

def init_extensions(app):
    """初始化所有扩展"""
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app) 