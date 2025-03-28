import datetime
import os
import uuid
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def get_timestamp():
    """
    获取当前时间的时间戳。
    :return: 当前时间的字符串时间戳（格式: YYYY-MM-DD_HH-MM-SS）
    """
    try:
        logger.info("生成时间戳")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logger.info(f"生成的时间戳: {timestamp}")
        return timestamp
    except Exception as e:
        logger.error(f"生成时间戳失败: {e}")
        raise RuntimeError(f"生成时间戳失败: {str(e)}")

def parse_date(date_str, format="%Y-%m-%d"):
    """
    将字符串日期解析为 datetime 对象。
    :param date_str: 字符串日期（如 '2025-01-01'）
    :param format: 日期格式（默认 '%Y-%m-%d'）
    :return: datetime 对象
    """
    if not date_str:
        logger.error("日期字符串为空")
        raise ValueError("日期字符串不能为空")

    try:
        logger.info(f"解析日期: {date_str}")
        date_obj = datetime.datetime.strptime(date_str, format)
        logger.info(f"解析成功: {date_obj}")
        return date_obj
    except ValueError as e:
        logger.error(f"解析日期失败: {e}")
        raise ValueError(f"解析日期失败: {str(e)}")
    except Exception as e:
        logger.error(f"解析日期时发生意外错误: {e}")
        raise RuntimeError(f"解析日期时发生意外错误: {str(e)}")

def generate_unique_id():
    """
    生成唯一的 UUID。
    :return: 唯一标识符字符串
    """
    try:
        logger.info("生成唯一 UUID")
        unique_id = str(uuid.uuid4())
        logger.info(f"生成的 UUID: {unique_id}")
        return unique_id
    except Exception as e:
        logger.error(f"生成唯一 UUID 失败: {e}")
        raise RuntimeError(f"生成唯一 UUID 失败: {str(e)}")

def file_size_in_mb(file_path):
    """
    获取文件大小（以 MB 为单位）。
    :param file_path: 文件路径
    :return: 文件大小（MB）
    """
    if not file_path:
        logger.error("文件路径为空")
        raise ValueError("文件路径不能为空")

    try:
        logger.info(f"获取文件大小: {file_path}")
        size_in_bytes = os.path.getsize(file_path)
        size_in_mb = size_in_bytes / (1024 * 1024)
        logger.info(f"文件大小: {size_in_mb:.2f} MB")
        return round(size_in_mb, 2)
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise FileNotFoundError(f"文件未找到: {file_path}")
    except Exception as e:
        logger.error(f"获取文件大小失败: {e}")
        raise RuntimeError(f"获取文件大小失败: {str(e)}")

def human_readable_size(size_in_bytes):
    """
    将字节大小转换为可读格式（如 KB、MB、GB）。
    :param size_in_bytes: 文件大小（字节）
    :return: 可读格式的文件大小字符串
    """
    if size_in_bytes < 0:
        logger.error("文件大小不能为负数")
        raise ValueError("文件大小不能为负数")

    try:
        logger.info(f"转换文件大小为可读格式: {size_in_bytes} 字节")
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_in_bytes < 1024:
                readable_size = f"{size_in_bytes:.2f} {unit}"
                logger.info(f"可读大小: {readable_size}")
                return readable_size
            size_in_bytes /= 1024
    except Exception as e:
        logger.error(f"转换文件大小失败: {e}")
        raise RuntimeError(f"转换文件大小失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例时间戳
    timestamp = get_timestamp()
    print(f"生成的时间戳: {timestamp}")

    # 示例日期解析
    date_str = "2025-01-01"
    date_obj = parse_date(date_str)
    print(f"解析后的日期: {date_obj}")

    # 示例生成唯一 ID
    unique_id = generate_unique_id()
    print(f"生成的唯一 ID: {unique_id}")

    # 示例文件大小
    file_path = "example.txt"
    content = "这是一个示例文件内容。"
    save_file(file_path, content)  # 假设 save_file 函数存在
    print(f"文件已保存到: {file_path}")

    size_mb = file_size_in_mb(file_path)
    print(f"文件大小: {size_mb} MB")

    # 示例可读文件大小
    size_in_bytes = 123456789
    readable_size = human_readable_size(size_in_bytes)
    print(f"可读文件大小: {readable_size}")

    # 删除示例文件
    delete_file(file_path)  # 假设 delete_file 函数存在
    print(f"文件已删除: {file_path}")