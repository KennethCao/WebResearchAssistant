import os
from app.config.constants import ALLOWED_EXTENSIONS, UPLOAD_FOLDER
from app.file_processing import is_allowed_file, get_secure_filepath, ensure_upload_folder_exists
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

class BlockchainFileHandler:
    """
    处理区块链相关文件的上传和管理
    """

    def __init__(self):
        """
        初始化文件处理器
        """
        ensure_upload_folder_exists()

    def upload_file(self, file):
        """
        上传文件到指定目录
        :param file: 文件对象
        :return: 文件在服务器上的路径
        """
        if not file:
            logger.error("上传的文件对象为空")
            raise ValueError("上传的文件对象不能为空")

        filename = file.filename
        if not is_allowed_file(filename):
            logger.error(f"文件扩展名不合法: {filename}")
            raise ValueError(f"文件扩展名不合法: {filename}")

        try:
            logger.info(f"开始上传文件: {filename}")
            secure_filepath = get_secure_filepath(filename)
            file.save(secure_filepath)
            logger.info(f"文件上传成功: {secure_filepath}")
            return secure_filepath
        except Exception as e:
            logger.error(f"文件上传失败: {e}")
            raise RuntimeError(f"文件上传失败: {str(e)}")

    def delete_file(self, filename):
        """
        删除指定文件
        :param filename: 文件名
        """
        if not filename:
            logger.error("文件名为空")
            raise ValueError("文件名不能为空")

        try:
            logger.info(f"开始删除文件: {filename}")
            secure_filepath = get_secure_filepath(filename)
            if os.path.exists(secure_filepath):
                os.remove(secure_filepath)
                logger.info(f"文件删除成功: {secure_filepath}")
            else:
                logger.warning(f"文件不存在: {secure_filepath}")
        except Exception as e:
            logger.error(f"文件删除失败: {e}")
            raise RuntimeError(f"文件删除失败: {str(e)}")

    def list_files(self):
        """
        列出上传文件夹中的所有文件
        :return: 文件名列表
        """
        try:
            logger.info("列出上传文件夹中的所有文件")
            if not os.path.exists(UPLOAD_FOLDER):
                logger.warning(f"上传文件夹不存在: {UPLOAD_FOLDER}")
                return []

            files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
            logger.info(f"列出文件成功: {files}")
            return files
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            raise RuntimeError(f"列出文件失败: {str(e)}")

    def read_file(self, filename):
        """
        读取指定文件的内容
        :param filename: 文件名
        :return: 文件内容
        """
        if not filename:
            logger.error("文件名为空")
            raise ValueError("文件名不能为空")

        try:
            logger.info(f"开始读取文件: {filename}")
            secure_filepath = get_secure_filepath(filename)
            if not os.path.exists(secure_filepath):
                logger.error(f"文件不存在: {secure_filepath}")
                raise FileNotFoundError(f"文件不存在: {secure_filepath}")

            with open(secure_filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"文件读取成功: {filename}")
                return content
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            raise RuntimeError(f"文件读取失败: {str(e)}")