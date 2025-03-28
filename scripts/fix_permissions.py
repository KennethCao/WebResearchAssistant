import os
import sys
from pathlib import Path
from scripts.utils import setup_script_logger

logger = setup_script_logger(__name__)

def fix_permissions():
    """修复目录权限"""
    try:
        # 获取项目根目录
        root_dir = Path(__file__).parent.parent
        
        # 需要修复权限的目录
        directories = [
            'cache',
            'cache/faiss',
            'cache/models',
            'cache/huggingface',
            'uploads',
            'instance',
            'logs',
            'data/knowledge_base',
            'static/pdfs'
        ]
        
        for dir_name in directories:
            dir_path = root_dir / dir_name
            # 创建目录（如果不存在）
            dir_path.mkdir(parents=True, exist_ok=True)
            
            try:
                # 设置目录权限为完全访问
                if sys.platform == 'win32':
                    os.system(f'icacls "{dir_path}" /grant Everyone:F /T')
                else:
                    os.chmod(dir_path, 0o777)
                logger.info(f"Fixed permissions for: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to set permissions for {dir_path}: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to fix permissions: {e}")
        return False

if __name__ == "__main__":
    fix_permissions() 