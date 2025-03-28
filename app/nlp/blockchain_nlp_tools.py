import os
import torch
from transformers import AutoTokenizer, AutoModel
from app.utils.logger import setup_logger
from app.nlp import get_tokenizer, get_model

# Setup logging
# 假设 setup_logger 函数需要 name 参数，这里设置一个合适的名称
logger = setup_logger(name="BlockchainNLPTools")

class BlockchainNLPTools:
    """
    Provides blockchain-related NLP tools and functionalities.
    """

    def __init__(self):
        """
        初始化 BlockchainNLPTools 类，加载默认的 NLP 模型。
        """
        self.tokenizer = get_tokenizer()
        self.model = get_model()

    def encode_text(self, text):
        """
        对文本进行编码。
        :param text: Input text
        :return: Encoded input tensors
        """
        if not text:
            logger.error("Input text is empty")
            raise ValueError("Input text cannot be empty")

        try:
            logger.info("Encoding text")
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            return inputs
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise RuntimeError(f"Text encoding failed: {str(e)}")

    def get_embeddings(self, text):
        """
        获取文本的嵌入向量。
        :param text: Input text
        :return: Text embeddings
        """
        if not text:
            logger.error("Input text is empty")
            raise ValueError("Input text cannot be empty")

        try:
            logger.info("Generating text embeddings")
            inputs = self.encode_text(text)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise RuntimeError(f"Failed to get embeddings: {str(e)}")

    def summarize_text(self, text, max_length=150, min_length=30, length_penalty=2.0):
        """
        Generate text summary.
        :param text: Input text
        :param max_length: Maximum summary length
        :param min_length: Minimum summary length
        :param length_penalty: Length penalty coefficient
        :return: Summary text
        """
        if not text:
            logger.error("Input text is empty")
            raise ValueError("Input text cannot be empty")

        try:
            logger.info("Generating text summary")
            inputs = self.encode_text(text)
            # 假设 inputs 是一个字典，需要解包传递给 generate 方法
            # 解包 inputs 字典传递给 generate 方法
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=4,
                early_stopping=True
            )
                **inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=4,
                early_stopping=True
            )
# 删除未使用的表达式
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=4,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            raise RuntimeError(f"Text summarization failed: {str(e)}")

    def classify_text(self, text, labels):
        """
        Classify text.
        :param text: Input text
        :param labels: List of labels
        :return: Classification result
        """
        if not text:
            logger.error("Input text is empty")
            raise ValueError("Input text cannot be empty")

        if not labels:
            logger.error("Label list is empty")
            raise ValueError("Label list cannot be empty")

        try:
            logger.info("Classifying text")
            inputs = self.encode_text(text)
            # 假设模型的输入需要使用 input_ids 和 attention_mask
            # 修改此处，直接将 inputs 字典解包传入模型
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_label_index = torch.argmax(probabilities, dim=-1).item()
            predicted_label = labels[predicted_label_index]
            return predicted_label
        except Exception as e:
            logger.error(f"Text classification failed: {e}")
            raise RuntimeError(f"Text classification failed: {str(e)}")

# 示例用法
if __name__ == "__main__":
    nlp_tools = BlockchainNLPTools()
    text = "这是一个示例文本，用于测试 NLP 工具的功能。"
    print("编码文本:", nlp_tools.encode_text(text))
    print("嵌入向量:", nlp_tools.get_embeddings(text))
    print("摘要文本:", nlp_tools.summarize_text(text))
    print("分类结果:", nlp_tools.classify_text(text, labels=["类别1", "类别2"]))