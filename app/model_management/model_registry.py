import os
import json
from app.utils.logger import setup_logger
from app.model_management import ensure_model_folder_exists, USE_GPU

# 设置日志记录
logger = setup_logger()

# 模型注册表文件路径
MODEL_REGISTRY_FILE = "./models/model_registry.json"

def initialize_model_registry():
    """
    初始化模型注册表文件。
    如果注册表文件不存在，则创建一个空的 JSON 文件。
    """
    ensure_model_folder_exists()  # 确保模型目录存在
    if not os.path.exists(MODEL_REGISTRY_FILE):
        with open(MODEL_REGISTRY_FILE, "w") as f:
            json.dump({}, f, indent=4)
        logger.info(f"模型注册表已创建: {MODEL_REGISTRY_FILE}")

def load_model_registry():
    """
    加载模型注册表。
    :return: 注册表字典
    """
    if not os.path.exists(MODEL_REGISTRY_FILE):
        logger.error(f"模型注册表文件不存在: {MODEL_REGISTRY_FILE}")
        raise FileNotFoundError(f"模型注册表文件不存在: {MODEL_REGISTRY_FILE}")

    try:
        with open(MODEL_REGISTRY_FILE, "r") as f:
            registry = json.load(f)
        logger.info("模型注册表加载成功")
        return registry
    except json.JSONDecodeError as e:
        logger.error(f"模型注册表文件格式错误: {e}")
        raise ValueError(f"模型注册表文件格式错误: {str(e)}")
    except Exception as e:
        logger.error(f"加载模型注册表失败: {e}")
        raise RuntimeError(f"加载模型注册表失败: {str(e)}")

def save_model_registry(registry):
    """
    保存模型注册表到文件。
    :param registry: 模型注册表字典
    """
    try:
        with open(MODEL_REGISTRY_FILE, "w") as f:
            json.dump(registry, f, indent=4)
        logger.info("模型注册表已更新")
    except Exception as e:
        logger.error(f"保存模型注册表失败: {e}")
        raise RuntimeError(f"保存模型注册表失败: {str(e)}")

def register_model(name, version, model_path, description=""):
    """
    注册一个新模型到注册表中。
    :param name: 模型名称
    :param version: 模型版本
    :param model_path: 模型文件路径
    :param description: 模型描述
    """
    if not name:
        logger.error("模型名称为空")
        raise ValueError("模型名称不能为空")

    if not version:
        logger.error("模型版本为空")
        raise ValueError("模型版本不能为空")

    if not model_path:
        logger.error("模型路径为空")
        raise ValueError("模型路径不能为空")

    initialize_model_registry()
    registry = load_model_registry()

    if name not in registry:
        registry[name] = {}

    if version in registry[name]:
        logger.warning(f"模型 {name} 版本 {version} 已存在，覆盖现有条目")

    registry[name][version] = {
        "path": model_path,
        "description": description,
        "use_gpu": USE_GPU
    }

    save_model_registry(registry)
    logger.info(f"模型已注册: {name} (版本: {version}, 路径: {model_path})")

def get_model_info(name, version):
    """
    获取指定模型及版本的信息。
    :param name: 模型名称
    :param version: 模型版本
    :return: 模型信息字典
    """
    if not name:
        logger.error("模型名称为空")
        raise ValueError("模型名称不能为空")

    if not version:
        logger.error("模型版本为空")
        raise ValueError("模型版本不能为空")

    registry = load_model_registry()

    if name not in registry:
        logger.error(f"模型 {name} 未在注册表中找到")
        raise ValueError(f"模型 {name} 未在注册表中找到")

    if version not in registry[name]:
        logger.error(f"模型 {name} 版本 {version} 未在注册表中找到")
        raise ValueError(f"模型 {name} 版本 {version} 未在注册表中找到")

    return registry[name][version]

def list_registered_models():
    """
    列出所有已注册的模型及其版本。
    :return: 模型注册表字典
    """
    registry = load_model_registry()
    logger.info("已注册模型列表加载成功")
    return registry