import os
from app.config.constants import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from app.utils.logger import setup_logger
from app.file_processing.file_validator import validate_file
from app.file_processing.pdf_parser import parse_pdf

# 设置日志记录器
logger = setup_logger(__name__)

def is_allowed_file(filename):
    """
    检查文件是否具有允许的扩展名
    :param filename: 文件名
    :return: 如果文件扩展名合法，返回 True；否则返回 False
    """
    if not filename:
        logger.error("输入文件名为空")
        raise ValueError("输入文件名不能为空")
    
    try:
        logger.info(f"检查文件扩展名: {filename}")
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    except Exception as e:
        logger.error(f"检查文件扩展名失败: {e}")
        raise RuntimeError(f"检查文件扩展名失败: {str(e)}")


def get_secure_filepath(filename):
    """
    获取文件在安全目录中的完整路径
    :param filename: 文件名
    :return: 安全的完整文件路径
    """
    if not filename:
        logger.error("输入文件名为空")
        raise ValueError("输入文件名不能为空")
    
    try:
        logger.info(f"获取文件的安全路径: {filename}")
        secure_filepath = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"文件的安全路径: {secure_filepath}")
        return secure_filepath
    except Exception as e:
        logger.error(f"获取文件安全路径失败: {e}")
        raise RuntimeError(f"获取文件安全路径失败: {str(e)}")


def ensure_upload_folder_exists():
    """
    确保上传文件夹存在，如果不存在则创建
    """
    try:
        logger.info(f"确保上传文件夹存在: {UPLOAD_FOLDER}")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        logger.info(f"上传文件夹已确保存在: {UPLOAD_FOLDER}")
    except Exception as e:
        logger.error(f"创建上传文件夹失败: {e}")
        raise RuntimeError(f"创建上传文件夹失败: {str(e)}")