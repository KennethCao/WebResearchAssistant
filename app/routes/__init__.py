# __init__.py
from flask import Blueprint
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def register_blueprints(app):
    """注册所有蓝图"""
    try:
        logger.info("Starting blueprint registration...")
        
        # 导入蓝图
        from app.routes.api import bp as api_bp
        from app.routes.web import bp as web_bp
        
        # 注册蓝图
        app.register_blueprint(api_bp, url_prefix='/api')
        logger.info(f"Registered API blueprint with prefix: /api")
        
        app.register_blueprint(web_bp)
        logger.info("Registered Web blueprint")
        
        # 打印所有注册的蓝图
        logger.info("Registered blueprints:")
        for blueprint in app.blueprints:
            logger.info(f"- {blueprint}")
        
        logger.info("Blueprint registration completed")
        
    except Exception as e:
        logger.error(f"Failed to register blueprints: {e}", exc_info=True)
        raise
    return app 