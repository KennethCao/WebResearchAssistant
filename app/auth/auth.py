from functools import wraps
from flask import request, jsonify, current_app
import jwt
from datetime import datetime, timedelta
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def generate_token(user_id: int, expires_in: int = 3600) -> str:
    """生成JWT令牌"""
    try:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(
            payload,
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise

def verify_token(token: str) -> dict:
    """验证JWT令牌"""
    try:
        return jwt.decode(
            token,
            current_app.config['SECRET_KEY'],
            algorithms=['HS256']
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"Invalid token: {e}")

def token_required(f):
    """Token验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # 获取token
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Invalid token format'}), 401
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
            
        try:
            # 验证token
            payload = verify_token(token)
            request.user_id = payload['user_id']
            
        except ValueError as e:
            return jsonify({'message': str(e)}), 401
            
        return f(*args, **kwargs)
        
    return decorated

def admin_required(f):
    """管理员权限装饰器"""
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        # 这里可以添加管理员验证逻辑
        if not is_admin(request.user_id):
            return jsonify({'message': 'Admin privilege required'}), 403
        return f(*args, **kwargs)
    return decorated

def is_admin(user_id: int) -> bool:
    """检查用户是否是管理员"""
    # 这里添加管理员检查逻辑
    return True  # 临时返回True 