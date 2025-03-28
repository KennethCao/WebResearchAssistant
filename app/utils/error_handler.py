from flask import jsonify, current_app
from app.utils.logger import setup_logger
from functools import wraps
from werkzeug.exceptions import HTTPException

# 设置日志记录器
logger = setup_logger(__name__)

class APIError(Exception):
    """API错误基类（状态码自动设置）"""
    status_code = 500
    
    def __init__(self, message, payload=None):
        super().__init__()
        self.message = message
        self.payload = payload

    def to_dict(self):
        """生成标准错误响应格式"""
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'code': self.status_code,
            **self.payload
        }

# 派生具体异常类型
class BadRequestError(APIError):
    status_code = 400

class UnauthorizedError(APIError):
    status_code = 401

class ForbiddenError(APIError):
    status_code = 403

class NotFoundError(APIError):
    status_code = 404

class MethodNotAllowedError(APIError):
    status_code = 405

class InternalServerError(APIError):
    status_code = 500

class ServiceUnavailableError(APIError):
    status_code = 503

def error_handler(error):
    """统一错误处理器（自动识别APIError）"""
    if isinstance(error, APIError):
        logger.error(f"APIError: {error.message}")
        return jsonify(error.to_dict()), error.status_code
    
    # 处理框架原生错误
    if isinstance(error, HTTPException):
        logger.error(f"HTTPException: {error.description}")
        return jsonify({
            'error': error.name,
            'message': error.description,
            'code': error.code
        }), error.code
    
    # 未处理异常
    logger.exception("Unhandled Exception")
    return jsonify({
        'error': 'InternalServerError',
        'message': 'An unexpected error occurred',
        'code': 500
    }), 500

def bad_request_error(error):
    """
    处理 400 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"400 错误: {str(error)}")
    response = {
        'error': 'Bad Request',
        'message': str(error),
        'status_code': 400
    }
    return jsonify(response), 400

def unauthorized_error(error):
    """
    处理 401 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"401 错误: {str(error)}")
    response = {
        'error': 'Unauthorized',
        'message': str(error),
        'status_code': 401
    }
    return jsonify(response), 401

def forbidden_error(error):
    """
    处理 403 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"403 错误: {str(error)}")
    response = {
        'error': 'Forbidden',
        'message': str(error),
        'status_code': 403
    }
    return jsonify(response), 403

def not_found_error(error):
    """
    处理 404 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"404 错误: {str(error)}")
    response = {
        'error': 'Not Found',
        'message': str(error),
        'status_code': 404
    }
    return jsonify(response), 404

def method_not_allowed_error(error):
    """
    处理 405 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"405 错误: {str(error)}")
    response = {
        'error': 'Method Not Allowed',
        'message': str(error),
        'status_code': 405
    }
    return jsonify(response), 405

def internal_server_error(error):
    """
    处理 500 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"500 错误: {str(error)}")
    response = {
        'error': 'Internal Server Error',
        'message': str(error),
        'status_code': 500
    }
    return jsonify(response), 500

def service_unavailable_error(error):
    """
    处理 503 错误。
    :param error: 错误对象
    :return: JSON 响应
    """
    logger.error(f"503 错误: {str(error)}")
    response = {
        'error': 'Service Unavailable',
        'message': str(error),
        'status_code': 503
    }
    return jsonify(response), 503

def register_error_handlers(app):
    """注册错误处理器装饰器"""
    def handler(status_code):
        """自动创建错误处理装饰器"""
        def decorator(cls):
            if not issubclass(cls, APIError):
                raise ValueError('必须继承自APIError')
            
            @app.errorhandler(cls)
            def handle_error(error):
                return error_handler(error)
            
            return cls
        return decorator
    
    # 注册HTTP状态码对应异常
    @handler(400)
    class HTTPBadRequestError(BadRequestError):
        """自动转换400错误"""
        def __init__(self, description):
            super().__init__(description)
    
    @handler(404)
    class HTTPNotFoundError(NotFoundError):
        """自动转换404错误"""
        def __init__(self, description):
            super().__init__(description)
    
    # 注册框架错误基类
    app.register_error_handler(HTTPException, error_handler)
    app.register_error_handler(Exception, error_handler)

def setup_error_handlers(app):
    """
    设置错误处理器
    """
    try:
        register_error_handlers(app)
        logger.info("Error handlers setup completed")
    except Exception as e:
        logger.error(f"Failed to setup error handlers: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    from flask import Flask
    
    # 创建 Flask 应用
    app = Flask(__name__)
    
    # 注册错误处理器
    register_error_handlers(app)
    
    # 测试路由
    @app.route('/test_400')
    def test_400():
        return bad_request_error("测试 400 错误")
    
    @app.route('/test_404')
    def test_404():
        return not_found_error("测试 404 错误")
    
    @app.route('/test_500')
    def test_500():
        return internal_server_error("测试 500 错误")
    
    # 运行应用
    app.run(debug=True)