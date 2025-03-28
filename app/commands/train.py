import click
from flask.cli import with_appcontext
from transformers import AutoTokenizer
import os
import gc
import torch
from .base import CommandBase, TrainingLifecycleMixin
from app.data.data_processor import prepare_training_data, BlockchainDataset
from app.config.environment import BLOCKCHAIN_MODEL_PATH
# 确保 app.training.train_model 模块存在且包含 ModelTrainer 类
try:
    from app.training.train_model import ModelTrainer
    if not hasattr(ModelTrainer, '__init__'):
        raise ImportError("ModelTrainer 不是一个有效的类。")
    # 检查导入是否成功
    if ModelTrainer is None:
        raise ImportError("ModelTrainer 导入失败，请检查模块路径和类名是否正确。")
except ImportError as e:
    import sys
    print(f"Error: 无法从 app.training.train_model 导入 ModelTrainer。请检查模块路径和类名是否正确。错误详情: {e}", file=sys.stderr)
    raise
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TrainCommand(CommandBase, TrainingLifecycleMixin):
    def __init__(self):
        super().__init__()
        self.trainer = ModelTrainer()

    @click.command('train-model')
    @with_appcontext
    def command(self):
        """训练命令入口"""
        pass

    def training_workflow(self, data_path, epochs, batch_size, learning_rate):
    """训练模型"""
    try:
        self.pre_training()
        
        if isinstance(self, TrainCommand):
            if 'data_path' in locals() or 'data_path' in globals():
                expanded_path = self.validate_data_path(data_path)
            else:
                raise ValueError("'data_path' is not defined.")
        else:
            raise ValueError("'self' is not an instance of TrainCommand.")

        if not hasattr(self, 'validate_data_path'):
            raise AttributeError("The 'validate_data_path' method is not defined in the class.")
        if hasattr(self, 'setup_training_environment'):
            self.setup_training_environment()
        else:
            logger.error("The 'setup_training_environment' method is not defined in the class.")
        
        logger.info("准备训练数据...")
        texts, labels = prepare_training_data(expanded_path)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        BlockchainDataset(texts, labels, tokenizer)
        
        model_save_path = os.path.join(BLOCKCHAIN_MODEL_PATH, 'fine_tuned')
        
        if 'epochs' in locals() or 'epochs' in globals():
            logger.info(f"开始训练 ({epochs}轮次)")
        else:
            logger.error("未定义 'epochs' 变量，请确保在调用此代码前已经定义了 'epochs' 变量。")
        self.trainer.train(
            data_path=expanded_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate if learning_rate is not None else 1e-4  # 默认学习率设为1e-4
        )
        if hasattr(self, 'post_training'):
            self.post_training()
        else:
            logger.error("The 'post_training' method is not defined in the class.")


        # 保存模型
        model_save_path = os.path.join(BLOCKCHAIN_MODEL_PATH, 'fine_tuned')
        # 确保在调用 save_model 方法时，self 是有效的实例
        # 由于 TrainCommand 已经在代码中定义，这里不需要再次处理未定义的问题
        if isinstance(self, TrainCommand):
            if hasattr(self.trainer, 'save_model'):
                self.trainer.save_model(model_save_path)
            else:
                logger.error("'self.trainer' does not have the 'save_model' method. Cannot save the model.")
        else:
            logger.error("'self' is not an instance of TrainCommand. Cannot save the model.")


        
        logger.info(f"Model successfully saved to {model_save_path}")
        click.echo('Model training completed successfully.')
        
    except Exception as e:
        logger.error(f'Error training model: {e}', exc_info=True)
        click.echo(f'Error training model: {e}')