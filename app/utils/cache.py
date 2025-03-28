import logging
from cachetools import LRUCache, TTLCache
from app.utils.logger import setup_logger
from threading import Lock

# 设置日志记录器 → # Setup logger
# 定义缓存策略 → # Define caching strategy
LRU_CACHE_SIZE = 1000  # LRU 缓存的最大大小 → # Max size for LRU cache
TTL_CACHE_SIZE = 1000  # TTL 缓存的最大大小 → # Max size for TTL cache
TTL_CACHE_TTL = 3600   # TTL 缓存的过期时间（秒） → # TTL expiration in seconds

# 初始化缓存和锁 → # Initialize cache and locks
# 初始化缓存实例。 → # Initialize cache instance
# 获取 LRU 缓存实例 → # Get LRU cache instance
# :return: LRU 缓存实例 → # :return: LRU cache instance
# 将键值对缓存到 LRU 缓存中 → # Cache key-value pair in LRU
# :param key: 缓存键 → # :param key: Cache key
logger.error("缓存键为空") → logger.error("Cache key is empty")
raise ValueError("缓存键不能为空") → raise ValueError("Cache key cannot be empty")
logger.info(f"将键值对缓存到 LRU 缓存中: {key}") → logger.info(f"Caching to LRU: {key}")
logger.error(f"缓存键值对失败: {e}") → logger.error(f"Cache failed: {e}")

# 从 LRU 缓存中获取值 → # Get value from LRU cache
logger.info(f"从 LRU 缓存中获取值: {key}") → logger.info(f"Getting from LRU: {key}")
logger.info(f"缓存中未找到键: {key}") → logger.info(f"Key not found: {key}")

# 清空 LRU 缓存 → # Clear LRU cache
logger.info("清空 LRU 缓存") → logger.info("Clearing LRU cache")
logger.error(f"清空 LRU 缓存失败: {e}") → logger.error(f"Clear LRU failed: {e}")

# 示例用法 → # Example usage
# 初始化缓存 → # Initialize cache
# 示例数据 → # Sample data
# 缓存数据到 LRU 缓存 → # Cache data to LRU
# 从 LRU 缓存中获取数据 → # Get data from LRU
# 清空缓存 → # Clear cache
if __name__ == "__main__":
    # 初始化缓存
    _initialize_caches()
    
    # 示例数据
    key1 = "example_key1"
    value1 = "example_value1"
    key2 = "example_key2"
    value2 = "example_value2"

    # 缓存数据到 LRU 缓存
    cache_lru(key1, value1)
    cache_lru(key2, value2)

    # 从 LRU 缓存中获取数据
    print(f"从 LRU 缓存中获取 {key1}: {get_from_lru_cache(key1)}")
    print(f"从 LRU 缓存中获取 {key2}: {get_from_lru_cache(key2)}")

    # 缓存数据到 TTL 缓存
    cache_ttl(key1, value1)
    cache_ttl(key2, value2)

    # 从 TTL 缓存中获取数据
    print(f"从 TTL 缓存中获取 {key1}: {get_from_ttl_cache(key1)}")
    print(f"从 TTL 缓存中获取 {key2}: {get_from_ttl_cache(key2)}")

    # 清空缓存
    clear_lru_cache()
    clear_ttl_cache()
    print("缓存已清空")