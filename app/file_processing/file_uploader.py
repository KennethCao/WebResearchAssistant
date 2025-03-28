import os
from werkzeug.utils import secure_filename
from app.config.constants import UPLOAD_FOLDER
from app.file_processing import is_allowed_file, get_secure_filepath, ensure_upload_folder_exists
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def upload_file(file):
    """
    处理文件上传逻辑，包括文件验证、保存以及错误处理。
    :param file: Flask `request.files` 中的文件对象
    :return: (状态, 信息) 二元组，状态为 True/False，信息为文件路径或错误消息
    """
    try:
        ensure_upload_folder_exists()  # 确保上传目录存在

        # 检查是否上传了文件
        if not file:
            logger.error("没有选择文件上传")
            return False, "没有选择文件"

        # 检查文件是否有合法的名称和扩展名
        if not is_allowed_file(file.filename):
            logger.error(f"文件扩展名不合法: {file.filename}")
            return False, "文件类型不被支持"

        # 确保文件名安全
        filename = secure_filename(file.filename)
        if not filename:
            logger.error("无效的文件名")
            return False, "无效的文件名"

        file_path = get_secure_filepath(filename)

        # 保存文件
        file.save(file_path)
        logger.info(f"文件上传成功: {file_path}")
        return True, file_path

    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        return False, f"文件上传失败: {str(e)}"

def validate_file(file):
    """
    验证上传的文件。
    :param file: 上传的文件对象
    :return: 如果文件有效返回 True，否则返回 False
    """
    if not file or not file.filename:
        return False
    return True

def save_uploaded_file(file, upload_folder, filename):
    """
    保存上传的文件。
    :param file: 上传的文件对象
    :param upload_folder: 保存目录
    :param filename: 文件名
    :return: 保存后的文件路径
    """
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filepath