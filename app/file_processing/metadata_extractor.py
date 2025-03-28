import os
from PyPDF2 import PdfReader
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def extract_metadata(file_path):
    """
    从 PDF 文件中提取元信息。
    :param file_path: PDF 文件路径
    :return: 包含提取的元信息的字典
    """
    if not file_path:
        logger.error("文件路径为空")
        return {"error": "文件路径为空"}

    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件路径不存在: {file_path}")
            return {"error": f"文件路径不存在: {file_path}"}

        # 打开 PDF 文件
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)

            # 提取元信息
            metadata = reader.metadata
            extracted_metadata = {
                "title": metadata.title if metadata and metadata.title else "未知标题",
                "author": metadata.author if metadata and metadata.author else "未知作者",
                "subject": metadata.subject if metadata and metadata.subject else "未知主题",
                "producer": metadata.producer if metadata and metadata.producer else "未知生产者",
                "page_count": len(reader.pages),  # 提取 PDF 文件的总页数
            }

            logger.info(f"成功提取元信息: {extracted_metadata}")
            return extracted_metadata

    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return {"error": f"文件未找到: {file_path}"}
    except PermissionError:
        logger.error(f"权限不足: 无法读取文件 {file_path}")
        return {"error": f"权限不足: 无法读取文件 {file_path}"}
    except Exception as e:
        logger.error(f"提取 PDF 元信息失败: {e}")
        return {"error": f"提取元信息失败: {str(e)}"}