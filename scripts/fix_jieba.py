import os
import shutil
from pathlib import Path
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def fix_jieba():
    """修复 jieba 配置"""
    try:
        # 获取 conda 环境路径
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if not conda_prefix:
            raise ValueError("CONDA_PREFIX not found")
            
        # jieba 字典路径
        jieba_dict = os.path.join(conda_prefix, 'Lib', 'site-packages', 'jieba', 'dict.txt')
        
        if os.path.exists(jieba_dict):
            # 备份原始文件
            backup_path = jieba_dict + '.backup'
            shutil.copy2(jieba_dict, backup_path)
            logger.info(f"Backed up original dict to {backup_path}")
            
            # 使用小词典模式
            os.environ['JIEBA_DICT'] = 'small'
            
            logger.info("Jieba fixed to use small dictionary mode")
            return True
    except Exception as e:
        logger.error(f"Failed to fix jieba: {e}")
        return False

if __name__ == "__main__":
    fix_jieba() 