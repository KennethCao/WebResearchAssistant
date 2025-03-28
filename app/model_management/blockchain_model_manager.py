import os
import torch
from app.model_management import MODEL_DIR, USE_GPU
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

class BlockchainModelManager:
    """
    管理区块链相关的模型加载、保存和使用。
    """

    def __init__(self):
        """
        初始化模型管理器
        """
        self.models = {}
        self.initialize_model_environment()

    def initialize_model_environment(self):
        """
        初始化模型环境，确保模型文件夹存在。
        """
        try:
            if USE_GPU:
                logger.info("已启用 GPU 模式运行。")
            else:
                logger.info("运行在 CPU 模式下。")
            self.ensure_model_folder_exists()
        except Exception as e:
            logger.error(f"初始化模型环境失败: {e}")
            raise RuntimeError(f"初始化模型环境失败: {str(e)}")

    def ensure_model_folder_exists(self):
        """
        确保模型文件夹存在，如果不存在则创建。
        """
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            logger.info(f"模型文件夹已确认存在: {MODEL_DIR}")
        except Exception as e:
            logger.error(f"创建模型文件夹失败: {e}")
            raise RuntimeError(f"创建模型文件夹失败: {str(e)}")

    def list_available_models(self):
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

    def get_model_path(self, model_name):
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

    def load_model(self, model_name):
        """
        加载指定的模型。
        :param model_name: 模型文件名
        :return: 加载的模型实例
        """
        if not model_name:
            logger.error("模型名称为空")
            raise ValueError("模型名称不能为空")

        if model_name in self.models:
            logger.info(f"模型已加载: {model_name}")
            return self.models[model_name]

        try:
            model_path = self.get_model_path(model_name)
            model = self._load_model_from_path(model_path)
            self.models[model_name] = model
            logger.info(f"模型加载成功: {model_name}")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def _load_model_from_path(self, model_path):
        """
        从指定路径加载模型。
        :param model_path: 模型文件路径
        :return: 加载的模型实例
        """
        try:
            if USE_GPU:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device("cpu")

            model = torch.load(model_path, map_location=device)
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"从路径加载模型失败: {model_path}, 错误: {e}")
            raise RuntimeError(f"从路径加载模型失败: {model_path}, 错误: {str(e)}")

    def save_model(self, model, model_name):
        """
        保存模型到指定路径。
        :param model: 模型实例
        :param model_name: 模型文件名
        """
        if not model_name:
            logger.error("模型名称为空")
            raise ValueError("模型名称不能为空")

        try:
            model_path = self.get_model_path(model_name)
            torch.save(model.state_dict(), model_path)
            logger.info(f"模型保存成功: {model_name}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            raise RuntimeError(f"保存模型失败: {str(e)}")

    def unload_model(self, model_name):
        """
        卸载指定的模型。
        :param model_name: 模型文件名
        """
        if not model_name:
            logger.error("模型名称为空")
            raise ValueError("模型名称不能为空")

        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"模型已卸载: {model_name}")
        else:
            logger.warning(f"模型未加载: {model_name}")