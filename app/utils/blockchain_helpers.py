import hashlib
import logging
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

def get_blockchain_data(api_url, endpoint, params=None):
    """
    从区块链 API 获取数据。
    :param api_url: API 的基础 URL
    :param endpoint: API 的端点
    :param params: 请求参数（可选）
    :return: API 返回的数据
    """
    import requests

    if not api_url:
        logger.error("API URL 为空")
        raise ValueError("API URL 不能为空")

    if not endpoint:
        logger.error("API 端点为空")
        raise ValueError("API 端点不能为空")

    try:
        logger.info(f"从 {api_url}{endpoint} 获取数据")
        response = requests.get(f"{api_url}{endpoint}", params=params)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        logger.info("数据获取完成")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"获取数据失败: {e}")
        raise RuntimeError(f"获取数据失败: {str(e)}")
    except ValueError as e:
        logger.error(f"解析 JSON 数据失败: {e}")
        raise RuntimeError(f"解析 JSON 数据失败: {str(e)}")

def calculate_sha256_hash(data):
    """
    计算数据的 SHA-256 哈希值。
    :param data: 输入数据（字符串）
    :return: SHA-256 哈希值
    """
    if not data:
        logger.error("输入数据为空")
        raise ValueError("输入数据不能为空")

    try:
        logger.info("计算 SHA-256 哈希值")
        sha256_hash = hashlib.sha256(data.encode()).hexdigest()
        logger.info("哈希值计算完成")
        return sha256_hash
    except Exception as e:
        logger.error(f"计算哈希值失败: {e}")
        raise RuntimeError(f"计算哈希值失败: {str(e)}")

def validate_blockchain_address(address):
    """
    验证区块链地址的有效性。
    :param address: 区块链地址
    :return: 验证结果（True 或 False）
    """
    if not address:
        logger.error("区块链地址为空")
        raise ValueError("区块链地址不能为空")

    try:
        logger.info(f"验证区块链地址: {address}")
        # 这里可以添加具体的地址验证逻辑
        # 例如，检查地址格式是否正确
        if len(address) != 42 or not address.startswith("0x"):
            logger.error(f"地址格式不正确: {address}")
            return False
        logger.info("地址验证完成")
        return True
    except Exception as e:
        logger.error(f"验证区块链地址失败: {e}")
        raise RuntimeError(f"验证区块链地址失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例 API 请求
    api_url = "https://api.blockchain.info"
    endpoint = "/q/getblockcount"
    try:
        data = get_blockchain_data(api_url, endpoint)
        print("获取的数据:", data)
    except Exception as e:
        print(f"示例 API 请求失败: {e}")

    # 示例哈希计算
    data = "这是一个示例数据"
    try:
        hash_value = calculate_sha256_hash(data)
        print("SHA-256 哈希值:", hash_value)
    except Exception as e:
        print(f"示例哈希计算失败: {e}")

    # 示例地址验证
    address = "0x1234567890123456789012345678901234567890"
    try:
        is_valid = validate_blockchain_address(address)
        print(f"地址 {address} 是否有效: {is_valid}")
    except Exception as e:
        print(f"示例地址验证失败: {e}")