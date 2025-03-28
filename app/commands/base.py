import click
from abc import ABC, abstractmethod
from app.utils.logger import setup_logger
import torch
import gc
import os

logger = setup_logger(__name__)

class CommandBase(ABC):
    def __init__(self):
        self.common_options = [
            click.option('--data-path', required=True, help='训练数据路径'),
            click.option('--epochs', default=3, help='训练轮数'),
            click.option('--batch-size', default=16, help='批次大小'),
            click.option('--learning-rate', default=2e-5, help='学习率'),
        ]

    @abstractmethod
    def validate_data_path(self, data_path):
        """验证数据路径公共方法"""
        expanded_path = os.path.expanduser(data_path)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"训练数据文件不存在: {expanded_path}")
        return expanded_path

    @abstractmethod
    def setup_training_environment(self):
        """初始化训练环境公共方法"""
        # 假设这里使用一个默认路径，你可以根据实际情况修改
        BLOCKCHAIN_MODEL_PATH = './blockchain_model'
        logger.info("初始化训练环境...")
        os.makedirs(BLOCKCHAIN_MODEL_PATH, exist_ok=True)

    def add_options(self, func):
        for option in reversed(self.common_options):
            func = option(func)
        return func

class TrainingLifecycleMixin:
    def pre_training(self):
        logger.info("执行训练前检查")
        torch.cuda.empty_cache()
        gc.collect()

    def post_training(self):
        logger.info("执行训练后清理")
        torch.cuda.empty_cache()
        gc.collect()

    @abstractmethod
    def training_workflow(self):
        pass