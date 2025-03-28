import os
from app.config.environment import MODEL_CACHE_DIR, USE_GPU
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

# 模型存储路径（从环境变量或配置文件中加载）
MODEL_DIR = MODEL_CACHE_DIR

def ensure_model_folder_exists():
    """
    确保模型文件夹存在，如果不存在则创建。
    """
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"模型文件夹已确认存在: {MODEL_DIR}")
    except Exception as e:
        logger.error(f"创建模型文件夹失败: {e}")
        raise RuntimeError(f"创建模型文件夹失败: {str(e)}")

def list_available_models():
    """
    列出当前可用的模型文件。
    :return: 可用模型文件的列表
    """
    try:
        if not os.path.exists(MODEL_DIR):
            logger.warning(f"模型文件夹不存在: {MODEL_DIR}")
            return []

        models = [f for f in os.listdir(MODEL_DIR) if os.path.isfile(os.path.join(MODEL_DIR, f))]
        logger.info(f"可用模型文件: {models}")
        return models
    except Exception as e:
        logger.error(f"列出模型文件失败: {e}")
        return []

def get_model_path(model_name):
    """
    根据模型名称获取完整的模型路径。
    :param model_name: 模型文件名
    :return: 模型文件的完整路径
    """
    if not model_name:
        logger.error("模型名称为空")
        raise ValueError("模型名称不能为空")

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        raise FileNotFoundError(f"模型文件 {model_name} 不存在")
    return model_path

def initialize_model_environment():
    """
    初始化模型环境，根据是否启用 GPU 设置相应的环境变量。
    """
    try:
        if USE_GPU:
            logger.info("已启用 GPU 模式运行。")
        else:
            logger.info("运行在 CPU 模式下。")
        ensure_model_folder_exists()
    except Exception as e:
        logger.error(f"初始化模型环境失败: {e}")
        raise RuntimeError(f"初始化模型环境失败: {str(e)}")