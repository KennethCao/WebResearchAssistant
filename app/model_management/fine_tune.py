import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from app.utils.logger import setup_logger
from app.model_management import ensure_model_folder_exists, USE_GPU

# 设置日志记录
logger = setup_logger()

# 模型存储路径
MODEL_DIR = "./models"
FINE_TUNED_MODEL_DIR = "./models/fine_tuned_model"

def fine_tune_huggingface_model(model_name, train_data, test_data, num_labels=2, epochs=3, batch_size=8):
    """
    微调 Hugging Face 模型
    :param model_name: 预训练模型名称或本地路径
    :param train_data: 训练数据集 (list of dicts with 'text' and 'label' keys)
    :param test_data: 测试数据集 (list of dicts with 'text' and 'label' keys)
    :param num_labels: 分类任务的标签数量
    :param epochs: 微调的训练轮数
    :param batch_size: 每批训练的数据大小
    :return: 微调后的模型路径
    """
    if not model_name:
        logger.error("模型名称为空")
        raise ValueError("模型名称不能为空")

    if not train_data:
        logger.error("训练数据为空")
        raise ValueError("训练数据不能为空")

    if not test_data:
        logger.error("测试数据为空")
        raise ValueError("测试数据不能为空")

    ensure_model_folder_exists()

    try:
        # 加载分词器和预训练模型
        logger.info(f"加载模型和分词器: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # 转换训练数据和测试数据为 Hugging Face 数据集格式
        def preprocess_data(data):
            return tokenizer(data['text'], padding=True, truncation=True)

        logger.info("准备训练和测试数据集")
        train_dataset = Dataset.from_list(train_data).map(preprocess_data, batched=True)
        test_dataset = Dataset.from_list(test_data).map(preprocess_data, batched=True)

        # 设置训练参数，根据配置选择使用 GPU 或 CPU
        logger.info("设置训练参数")
        training_args = TrainingArguments(
            output_dir=FINE_TUNED_MODEL_DIR,  # 保存微调模型的目录
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",  # 日志目录
            logging_steps=10,
            save_total_limit=2,  # 保存的模型 checkpoint 数量
            load_best_model_at_end=True,
            no_cuda=not USE_GPU,  # 根据配置选择使用 GPU 或 CPU
        )

        # 创建 Trainer 实例
        logger.info("初始化 Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
        )

        # 开始训练
        logger.info("开始微调模型")
        trainer.train()

        # 保存微调后的模型
        logger.info(f"保存微调后的模型到: {FINE_TUNED_MODEL_DIR}")
        trainer.save_model(FINE_TUNED_MODEL_DIR)
        tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)

        return FINE_TUNED_MODEL_DIR

    except Exception as e:
        logger.error(f"微调模型失败: {e}")
        raise RuntimeError(f"微调模型失败: {str(e)}")

def evaluate_model(test_data, model_dir):
    """
    使用微调后的模型进行评估
    :param test_data: 测试数据集 (list of dicts with 'text' and 'label' keys)
    :param model_dir: 微调模型的路径
    :return: 模型的评估结果
    """
    if not test_data:
        logger.error("测试数据为空")
        raise ValueError("测试数据不能为空")

    if not model_dir:
        logger.error("模型路径为空")
        raise ValueError("模型路径不能为空")

    try:
        logger.info(f"加载微调模型进行评估: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # 转换测试数据
        def preprocess_data(data):
            return tokenizer(data['text'], padding=True, truncation=True)

        test_dataset = Dataset.from_list(test_data).map(preprocess_data, batched=True)

        # 创建 Trainer 实例，根据配置选择使用 GPU 或 CPU
        logger.info("初始化 Trainer 进行评估")
        trainer = Trainer(model=model, tokenizer=tokenizer)

        # 评估模型
        logger.info("开始评估模型")
        results = trainer.evaluate(test_dataset)
        logger.info(f"评估结果: {results}")

        return results

    except Exception as e:
        logger.error(f"评估模型失败: {e}")
        raise RuntimeError(f"评估模型失败: {str(e)}")