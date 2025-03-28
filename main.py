import os
from flask import Flask
try:
    from prometheus_client import make_wsgi_app
except ImportError:
    print("未找到 'prometheus_client' 模块，请使用以下命令安装：")
    print("pip install prometheus_client")
    raise
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app import create_app
from app.extensions import db
from app.utils.logger import setup_logger
from app.monitoring.prometheus_monitor import setup_prometheus_monitoring
from app.utils.startup_checks import perform_startup_checks
from app.middlewares.health_check import HealthCheckMiddleware
from app.middlewares.security_headers import SecurityHeadersMiddleware  # 补充安全头中间件导入
from pydantic_settings import BaseSettings

# 环境变量配置
os.environ.update({
    'HF_ENDPOINT': 'https://hf-mirror.com',
    'TRANSFORMERS_OFFLINE': '1'
})

logger = setup_logger(__name__)

def initialize_application():
    """Unified application initialization entry"""
    try:
        # 1. 前置环境检查
        perform_startup_checks()

        # 2. 初始化基础配置
        os.environ.update({
            'HF_ENDPOINT': 'https://hf-mirror.com',
            'TRANSFORMERS_OFFLINE': '1',
            'FLASK_DEBUG': '0'
        })

        # 3. 创建增强型应用实例
        app = create_app(settings={
            'ENV': os.getenv('FLASK_ENV', 'production'),
            'MONITORING': True,
            'PRELOAD_MODELS': os.getenv('SKIP_MODEL_PRELOAD', '0') == '0'
        })

        # 4. 数据库初始化流程
        with app.app_context():
            # 移除重复的db.create_all()调用
            # 保留必要的扩展初始化
            from flask_migrate import Migrate
            migrate = Migrate(app, db)
            
            # 数据库版本校验
            current_version = db.engine.execute('PRAGMA user_version').fetchone()[0]
            expected_version = 20240501  # 设置当前数据库版本号
            
            if current_version != expected_version:
                # 开启事务进行迁移
                try:
                    connection = db.engine.connect()
                    trans = connection.begin()
                    db.engine.execute(f'PRAGMA user_version={expected_version}')
                    migrate = Migrate(app, db)
                    trans.commit()
                except Exception as e:
                    trans.rollback()
                    logger.error(f"数据库迁移失败，已回滚: {str(e)}")
                    raise
            
            # 模型预加载系统（保持原有逻辑）
            try:
                from app.services.model_manager import check_and_initialize_models
                check_and_initialize_models()
                app.faiss_retriever = load_faiss_retriever()
except ImportError:
    logger.error("未找到 'app.services.model_manager' 模块，请检查模块路径。")
    raise
except Exception as e:
    logger.error(f"模型预加载失败: {str(e)}")
    raise

# 5. 中间件注册系统
# 初始化Prometheus监控系统
try:
    prometheus_monitor = setup_prometheus_monitoring(app)
except Exception as e:
    logger.error(f"Prometheus监控系统初始化失败: {str(e)}")
    raise

# 中间件注册系统
try:
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        '/health': HealthCheckMiddleware()
    })
except Exception as e:
    logger.error(f"中间件注册失败: {str(e)}")
    raise

# 6. 全局错误处理链
try:
    from app.utils.error_handler import (
        setup_global_error_handlers,
        register_shutdown_hooks
    )
    setup_global_error_handlers(app)
    register_shutdown_hooks(app)
except ImportError:
    logger.error("未找到 'app.utils.error_handler' 模块，请检查模块路径。")
    raise
except Exception as e:
    logger.error(f"全局错误处理链初始化失败: {str(e)}")
    raise

# 7. 安全响应头设置
try:
    # 检查 'app.middlewares.security_headers' 是否存在
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'app', 'middlewares', 'security_headers.py')):
        logger.error("未找到 'app.middlewares.security_headers' 模块，请确保该文件存在。")
        raise ImportError("未找到 'app.middlewares.security_headers' 模块")
    app.wsgi_app = SecurityHeadersMiddleware(app.wsgi_app)
except ImportError:
    logger.error("未找到 'app.middlewares.security_headers' 模块，请检查模块路径。")
    raise
except Exception as e:
    logger.error(f"安全响应头设置失败: {str(e)}")
    raise

return app

except Exception as e:
    logger.critical(f"应用初始化失败: {str(e)}", exc_info=True)
    raise


# 模型生命周期管理系统
try:
    from app.services.model_lifecycle import (
        check_base_models,
        handle_model_finetuning,
        update_knowledge_base
    )
    check_base_models(['all-MiniLM-L6-v2', 'bert-base-uncased'])
except ImportError:
    logger.error("未找到 'app.services.model_lifecycle' 模块")
    raise
        check_base_models,
        handle_model_finetuning,
        update_knowledge_base
    )
except ImportError:
    print("未找到 'app.services.model_lifecycle' 模块，请检查模块路径。")
    raise
    check_base_models,
    handle_model_finetuning,
    update_knowledge_base
)

# 文件上传处理系统
try:
    from app.services.file_processor import (
        process_uploaded_files,
        setup_file_monitoring
    )
except ImportError:
    print("未找到 'app.services.file_processor' 模块，请检查模块路径。")
    raise
    process_uploaded_files,

    setup_file_monitoring
)

# 在初始化流程中添加模型检查
try:
    # Check and download base models
    check_base_models(['all-MiniLM-L6-v2', 'bert-base-uncased'])
    
    # 初始化文件监控系统
    setup_file_monitoring(
        watch_dir=os.path.join('static', 'pdfs'),
        handler=lambda: handle_model_finetuning(update_knowledge_base)
    )

except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}", exc_info=True)
    raise

if __name__ == '__main__':
    try:
        app = initialize_application()
        # 增强型启动配置
        if app:
            app.run(
                host=os.getenv('SERVER_HOST', '0.0.0.0'),
                port=int(os.getenv('SERVER_PORT', 5000)),
                use_reloader=False,
                threaded=True,
                ssl_context='adhoc' if os.getenv('ENABLE_HTTPS') else None,
                server='waitress'
            )
            host=os.getenv('SERVER_HOST', '127.0.0.1'),
            port=int(os.getenv('SERVER_PORT', 5000)),
            use_reloader=False,
            threaded=True,
            ssl_context='adhoc' if os.getenv('ENABLE_HTTPS') else None,
            server='waitress'
        )
    except KeyboardInterrupt:
        logger.info("服务正常终止")
    except Exception as e:
        # 这里添加具体的异常处理逻辑，假设需要打印异常信息
        print(f"An error occurred: {e}")
        logger.error(f"服务异常终止: {str(e)}", exc_info=True)
        raise SystemExit(1)

# 安全头中间件配置
class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def custom_start_response(status, headers, exc_info=None):
            headers.extend([
                ('Content-Security-Policy', "default-src 'self'"),
                ('X-Content-Type-Options', 'nosniff'),
                ('X-Frame-Options', 'DENY')
            ])
            return start_response(status, headers, exc_info)
        return self.app(environ, custom_start_response)

# 移除整个数据库初始化区块
# 保留中间件注册和监控系统

# 应用启动流程
import sys  # 确保sys模块已导入
if __name__ == "__main__":
    if os.getenv('INITIALIZATION_DONE') != '1':
        logger.error("请通过set_env.bat启动应用以执行完整初始化流程")
        sys.exit(1)
    
    app = create_app(settings={
        'ENV': 'production',
        'DEBUG': False,
        'TESTING': False
    })
    # 检查app是否为Flask应用实例
    if isinstance(app, Flask):
        # 仅保留WSGI配置和运行命令
        app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
            '/metrics': make_wsgi_app()
        })
    else:
        logger.error("app不是Flask应用实例，无法设置wsgi_app。")
        sys.exit(1)
    
    app.run(host='127.0.0.1', port=5000)
