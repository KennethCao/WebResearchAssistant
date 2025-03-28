import os
import uuid
from app.file_processing.file_uploader import validate_file, save_uploaded_file
from app.file_processing.pdf_parser import parse_pdf
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

# 上传文件的存储路径
UPLOAD_FOLDER = "./static/pdfs/"

def handle_file_upload(file):
    """
    处理文件上传，包括验证和保存。
    :param file: 上传的文件对象
    :return: 保存后的文件路径
    """
    if not file:
        logger.error("上传的文件对象为空")
        raise ValueError("上传的文件对象不能为空")

    try:
        # 验证文件类型
        logger.info("验证上传的文件")
        if not validate_file(file):
            logger.error("文件验证失败: 不支持的文件类型")
            raise ValueError("文件验证失败: 不支持的文件类型")

        # 生成唯一文件名
        unique_filename = generate_unique_filename(file.filename)

        # 保存文件
        logger.info("保存上传的文件")
        saved_path = save_uploaded_file(file, UPLOAD_FOLDER, unique_filename)
        logger.info(f"文件已保存到: {saved_path}")
        return saved_path
    except Exception as e:
        logger.error(f"文件上传处理失败: {e}")
        raise RuntimeError(f"文件上传处理失败: {str(e)}")

def handle_pdf_parsing(file_path):
    """
    处理 PDF 文件解析。
    :param file_path: PDF 文件路径
    :return: 解析后的文本内容
    """
    if not file_path:
        logger.error("PDF 文件路径为空")
        raise ValueError("PDF 文件路径不能为空")

    try:
        logger.info(f"解析 PDF 文件: {file_path}")
        content = parse_pdf(file_path)
        logger.info("PDF 文件解析完成")
        return content
    except Exception as e:
        logger.error(f"PDF 文件解析失败: {e}")
        raise RuntimeError(f"PDF 文件解析失败: {str(e)}")

def generate_unique_filename(filename):
    """
    生成唯一的文件名。
    :param filename: 原始文件名
    :return: 唯一文件名
    """
    if not filename:
        logger.error("原始文件名为空")
        raise ValueError("原始文件名不能为空")

    try:
        logger.info("生成唯一文件名")
        unique_id = uuid.uuid4().hex
        extension = os.path.splitext(filename)[1]
        unique_filename = f"{unique_id}{extension}"
        logger.info(f"生成的唯一文件名: {unique_filename}")
        return unique_filename
    except Exception as e:
        logger.error(f"生成唯一文件名失败: {e}")
        raise RuntimeError(f"生成唯一文件名失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例文件对象（假设这是一个文件对象）
    class MockFile:
        def __init__(self, filename):
            self.filename = filename

        def save(self, path):
            with open(path, 'w') as f:
                f.write("示例文件内容")

    # 示例文件上传
    mock_file = MockFile("example.pdf")
    try:
        saved_path = handle_file_upload(mock_file)
        print("保存后的文件路径:", saved_path)

        # 示例 PDF 解析
        content = handle_pdf_parsing(saved_path)
        print("解析后的文本内容:", content)
    except Exception as e:
        print(f"示例处理失败: {e}")