import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name):
    """设置日志记录器"""
    # 检查是否已经存在同名logger
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f'{name}.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 避免日志重复
    logger.propagate = False
    
    return logger

# 示例用法
if __name__ == "__main__":
    # 获取根日志记录器
    root_logger = setup_logger(__name__)
    root_logger.info("这是根日志记录器的测试消息")
    
    # 获取命名日志记录器
    app_logger = setup_logger("app")
    app_logger.info("这是应用日志记录器的测试消息")
    
    # 使用自定义日志文件
    custom_logger = setup_logger("custom")
    custom_logger.info("这是自定义日志文件的测试消息")