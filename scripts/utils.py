import logging
import os
from pathlib import Path

def setup_script_logger(name):
    """为脚本设置简单的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger 