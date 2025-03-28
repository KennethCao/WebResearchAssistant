from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.utils.logger import setup_logger

# 设置日志记录
logger = setup_logger()

# 全局变量
generation_model = None
tokenizer = None
GENERATION_MODEL_NAME = "facebook/bart-large-cnn"  # 默认生成模型

def initialize_generator():
    """
    初始化生成模型和分词器。
    """
    global generation_model, tokenizer

    try:
        logger.info(f"加载生成模型: {GENERATION_MODEL_NAME}")
        # 强制使用 CPU 加载模型
        generation_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME, device_map=None)
        tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
        logger.info("生成模型初始化完成，运行于 CPU")
    except Exception as e:
        logger.error(f"初始化生成模型失败: {e}")
        raise RuntimeError(f"初始化生成模型失败: {str(e)}")

def generate_text(query, context_list, max_length=200, num_beams=4):
    """
    基于查询和上下文生成文本。
    :param query: 用户的查询文本
    :param context_list: 检索到的相关文档列表
    :param max_length: 生成文本的最大长度
    :param num_beams: Beam Search 的宽度
    :return: 生成的文本
    """
    global generation_model, tokenizer

    if not query:
        logger.error("查询文本为空")
        raise ValueError("查询文本不能为空")

    if not context_list:
        logger.error("上下文列表为空")
        raise ValueError("上下文列表不能为空")

    if not all(isinstance(doc, str) for doc in context_list):
        logger.error("上下文列表中的每个元素必须是字符串")
        raise TypeError("上下文列表中的每个元素必须是字符串")

    if max_length <= 0:
        logger.error("生成文本的最大长度必须大于 0")
        raise ValueError("生成文本的最大长度必须大于 0")

    if num_beams <= 0:
        logger.error("Beam Search 的宽度必须大于 0")
        raise ValueError("Beam Search 的宽度必须大于 0")

    if generation_model is None or tokenizer is None:
        logger.warning("生成模型未初始化，正在初始化...")
        initialize_generator()

    try:
        # 组合上下文信息
        logger.info("组合上下文信息")
        context = "\n".join(context_list)
        input_text = f"上下文: {context}\n问题: {query}"

        # 分词并生成
        logger.info("开始生成文本")
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
        outputs = generation_model.generate(
            inputs.to("cpu"),  # 确保在 CPU 上运行
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

        # 解码生成的文本
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("文本生成完成")
        return response
    except Exception as e:
        logger.error(f"生成文本失败: {e}")
        raise RuntimeError(f"生成文本失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 初始化生成模型和分词器
    initialize_generator()

    # 示例查询和上下文
    query = "区块链技术的发展趋势"
    context_list = [
        "区块链技术在金融领域的应用越来越广泛。",
        "智能合约是区块链技术的核心组成部分。",
        "区块链技术提供了去中心化的解决方案。",
        "区块链技术的安全性和透明性是其主要优势。"
    ]

    # 生成文本
    response = generate_text(query, context_list, max_length=200, num_beams=4)
    print("生成的文本:", response)