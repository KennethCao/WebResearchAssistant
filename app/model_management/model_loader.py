import os
import requests
from transformers import AutoModel, AutoTokenizer
from app.utils.logger import setup_logger
from app.model_management import ensure_model_folder_exists

# 设置日志记录
logger = setup_logger()

# 模型存储路径
MODEL_DIR = "./models"

def download_huggingface_model(model_name, save_dir=None):
    """
    从 Hugging Face 下载模型并保存到指定目录，仅使用 CPU。
    :param model_name: Hugging Face 模型名称，例如 'bert-base-uncased'
    :param save_dir: 模型保存目录（默认保存到 ./models）
    :return: 模型和分词器的本地保存路径
    """
    if not model_name:
        logger.error("模型名称为空")
        raise ValueError("模型名称不能为空")

    if save_dir is None:
        save_dir = os.path.join(MODEL_DIR, model_name.replace("/", "_"))

    ensure_model_folder_exists()
    os.makedirs(save_dir, exist_ok=True)

    try:
        # 下载模型和分词器，仅使用 CPU
        logger.info(f"开始下载 Hugging Face 模型（仅使用 CPU）: {model_name}")
        model = AutoModel.from_pretrained(model_name, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 保存到本地
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        logger.info(f"模型已下载并保存到: {save_dir}")
        return save_dir
    except Exception as e:
        logger.error(f"下载 Hugging Face 模型失败: {e}")
        raise RuntimeError(f"下载 Hugging Face 模型失败: {str(e)}")

def download_remote_model(url, save_path):
    """
    从指定的远程 URL 下载模型文件。
    :param url: 远程模型文件的 URL
    :param save_path: 本地保存路径
    :return: 模型本地保存路径
    """
    if not url:
        logger.error("URL 为空")
        raise ValueError("URL 不能为空")

    if not save_path:
        logger.error("保存路径为空")
        raise ValueError("保存路径不能为空")

    ensure_model_folder_exists()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        logger.info(f"开始从远程下载模型: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        # 下载并写入文件
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"远程模型已下载并保存到: {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        logger.error(f"下载远程模型失败: {e}")
        raise RuntimeError(f"下载远程模型失败: {str(e)}")
    except Exception as e:
        logger.error(f"保存远程模型失败: {e}")
        raise RuntimeError(f"保存远程模型失败: {str(e)}")