from transformers import pipeline
from app.utils.logger import setup_logger
from app.nlp import get_tokenizer, get_model
import torch

# 设置日志记录
logger = setup_logger()

def summarize_text(text, max_length=150, min_length=40, length_penalty=2.0, num_beams=4):
    """
    使用预训练模型生成文本摘要。
    :param text: 要生成摘要的文本
    :param max_length: 摘要的最大长度
    :param min_length: 摘要的最小长度
    :param length_penalty: 长度惩罚系数（值越大生成的摘要越简短）
    :param num_beams: Beam Search 的束宽
    :return: 文本摘要
    """
    if not text:
        logger.error("输入文本为空")
        raise ValueError("输入文本不能为空")

    if max_length <= min_length:
        logger.error("最大长度必须大于最小长度")
        raise ValueError("最大长度必须大于最小长度")

    try:
        logger.info("加载摘要生成模型和分词器")
        model = get_model()
        tokenizer = get_tokenizer()

        # 使用 Hugging Face 的 pipeline 简化摘要生成
        logger.info("初始化摘要生成管道")
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)  # device=-1 强制使用 CPU

        logger.info("开始生成摘要")
        with torch.no_grad():  # 确保禁用梯度计算
            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=num_beams,
                do_sample=False
            )

        logger.info("摘要生成完成")
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"生成摘要失败: {e}")
        raise RuntimeError(f"生成摘要失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    text = "这是一个示例文本，用于测试摘要生成功能。摘要生成器应该能够从这个文本中提取出关键信息，并生成一个简洁的摘要。"
    summary = summarize_text(text)
    print("生成的摘要:", summary)