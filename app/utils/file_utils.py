import os
import shutil
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def save_file(file_path, content):
    """
    将内容保存到文件。
    :param file_path: 文件路径
    :param content: 要保存的内容（字符串）
    """
    if not file_path:
        logger.error("文件路径为空")
        raise ValueError("文件路径不能为空")

    if not isinstance(content, str):
        logger.error("内容必须是字符串类型")
        raise TypeError("内容必须是字符串类型")

    try:
        logger.info(f"开始保存文件: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"文件保存成功: {file_path}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise RuntimeError(f"保存文件失败: {str(e)}")

def read_file(file_path):
    """
    读取文件内容。
    :param file_path: 文件路径
    :return: 文件内容（字符串）
    """
    if not file_path:
        logger.error("文件路径为空")
        raise ValueError("文件路径不能为空")

    try:
        logger.info(f"开始读取文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"文件读取成功: {file_path}")
        return content
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise RuntimeError(f"读取文件失败: {str(e)}")

def check_file_exists(file_path):
    """
    检查文件是否存在。
    :param file_path: 文件路径
    :return: 文件是否存在的布尔值
    """
    if not file_path:
        logger.error("文件路径为空")
        raise ValueError("文件路径不能为空")

    exists = os.path.exists(file_path)
    logger.info(f"检查文件是否存在: {file_path} - {'存在' if exists else '不存在'}")
    return exists

def delete_file(file_path):
    """
    删除指定文件。
    :param file_path: 文件路径
    """
    if not file_path:
        logger.error("文件路径为空")
        raise ValueError("文件路径不能为空")

    try:
        if check_file_exists(file_path):
            logger.info(f"开始删除文件: {file_path}")
            os.remove(file_path)
            logger.info(f"文件删除成功: {file_path}")
        else:
            logger.warning(f"文件不存在，无法删除: {file_path}")
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        raise RuntimeError(f"删除文件失败: {str(e)}")

def move_file(src_path, dest_path):
    """
    移动文件到新位置。
    :param src_path: 源文件路径
    :param dest_path: 目标文件路径
    """
    if not src_path:
        logger.error("源文件路径为空")
        raise ValueError("源文件路径不能为空")

    if not dest_path:
        logger.error("目标文件路径为空")
        raise ValueError("目标文件路径不能为空")

    try:
        logger.info(f"开始移动文件: {src_path} 到 {dest_path}")
        shutil.move(src_path, dest_path)
        logger.info(f"文件移动成功: {src_path} 到 {dest_path}")
    except Exception as e:
        logger.error(f"移动文件失败: {e}")
        raise RuntimeError(f"移动文件失败: {str(e)}")

def create_directory(directory_path):
    """
    创建目录（如果不存在）。
    :param directory_path: 目录路径
    """
    if not directory_path:
        logger.error("目录路径为空")
        raise ValueError("目录路径不能为空")

    try:
        logger.info(f"检查目录是否存在: {directory_path}")
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"目录已创建（或已存在）: {directory_path}")
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        raise RuntimeError(f"创建目录失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例文件路径
    file_path = "example.txt"
    content = "这是一个示例文件内容。"

    # 保存文件
    save_file(file_path, content)
    print(f"文件已保存到: {file_path}")

    # 读取文件
    read_content = read_file(file_path)
    print(f"读取的文件内容: {read_content}")

    # 检查文件是否存在
    exists = check_file_exists(file_path)
    print(f"文件是否存在: {exists}")

    # 删除文件
    delete_file(file_path)
    print(f"文件已删除: {file_path}")

    # 创建目录
    directory_path = "example_dir"
    create_directory(directory_path)
    print(f"目录已创建: {directory_path}")

    # 移动文件
    src_path = "example.txt"
    dest_path = os.path.join(directory_path, "example_moved.txt")
    save_file(src_path, content)  # 重新保存文件以便移动
    move_file(src_path, dest_path)
    print(f"文件已移动到: {dest_path}")