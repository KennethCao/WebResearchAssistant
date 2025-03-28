import os
from app.utils.logger import setup_logger
from app.config.environment import (
    BASE_DIR,
    UPLOAD_FOLDER,
    VECTOR_DB_PATH,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    MODEL_DIMENSION,
    BLOCKCHAIN_PAPER_PATH,
    BLOCKCHAIN_MODEL_PATH,
    BLOCKCHAIN_INDEX_PATH,
    LOG_FILE,
    MAX_CONTENT_LENGTH,
    SQLITE_DB_PATH
)

# 设置日志记录器
logger = setup_logger(__name__)

def validate_config() -> bool:
    """
    Validate configuration validity
    :return: True if valid, False otherwise
    """
    try:
        # 验证基础目录
        if not os.path.exists(BASE_DIR):
            logger.error("BASE_DIR does not exist")
            return False

        # 验证上传目录
        if not os.path.exists(UPLOAD_FOLDER):
            try:
                os.makedirs(UPLOAD_FOLDER)
                logger.info(f"Created upload directory: {UPLOAD_FOLDER}")
            except Exception as e:
                logger.error(f"创建上传目录失败: {e}")
                return False

        # 验证向量数据库目录
        if not os.path.exists(VECTOR_DB_PATH):
            try:
                os.makedirs(VECTOR_DB_PATH)
                logger.info(f"创建向量数据库目录: {VECTOR_DB_PATH}")
            except Exception as e:
                logger.error(f"创建向量数据库目录失败: {e}")
                return False

        # 验证 FAISS 索引目录
        faiss_dir = os.path.dirname(FAISS_INDEX_PATH)
        if not os.path.exists(faiss_dir):
            try:
                os.makedirs(faiss_dir)
                logger.info(f"创建 FAISS 索引目录: {faiss_dir}")
            except Exception as e:
                logger.error(f"创建 FAISS 索引目录失败: {e}")
                return False

        # 验证模型名称
        if not MODEL_NAME:
            logger.error("MODEL_NAME not configured")
            return False

        # 验证模型维度
        if not isinstance(MODEL_DIMENSION, int) or MODEL_DIMENSION <= 0:
            logger.error("MODEL_DIMENSION 必须是正整数")
            return False

        # 验证最大文件大小
        if not isinstance(MAX_CONTENT_LENGTH, int) or MAX_CONTENT_LENGTH <= 0:
            logger.error("MAX_CONTENT_LENGTH 必须是正整数")
            return False

        # 验证区块链相关目录
        required_dirs = [
            BLOCKCHAIN_PAPER_PATH,
            BLOCKCHAIN_MODEL_PATH,
            BLOCKCHAIN_INDEX_PATH
        ]
        
        for directory in required_dirs:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    logger.info(f"创建目录: {directory}")
                except Exception as e:
                    logger.error(f"创建目录失败 {directory}: {e}")
                    return False

        # 验证日志文件目录
        log_dir = os.path.dirname(LOG_FILE)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                logger.info(f"创建日志目录: {log_dir}")
            except Exception as e:
                logger.error(f"创建日志目录失败: {e}")
                return False

        logger.info("配置验证成功")
        return True

    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        return False

def validate_config_complete():
    """验证配置是否完整"""
    required_paths = {
        'UPLOAD_FOLDER': UPLOAD_FOLDER,
        'BLOCKCHAIN_PAPER_PATH': BLOCKCHAIN_PAPER_PATH,
        'BLOCKCHAIN_MODEL_PATH': BLOCKCHAIN_MODEL_PATH,
        'BLOCKCHAIN_INDEX_PATH': BLOCKCHAIN_INDEX_PATH,
        'SQLITE_DB_PATH': SQLITE_DB_PATH
    }
    
    for name, path in required_paths.items():
        if not os.path.exists(os.path.dirname(path)):
            raise ValueError(f"Required directory for {name} does not exist")

# 示例用法
if __name__ == "__main__":
    if validate_config():
        print("配置验证通过")
    else:
        print("配置验证失败")