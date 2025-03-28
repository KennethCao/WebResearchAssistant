from datetime import datetime
from pathlib import Path
import json
import hashlib
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class VersionControlSystem:
    def __init__(self):
        self.repo_path = Path('data/versions')
        self.metadata_file = self.repo_path / 'versions.json'
        self._initialize()

    def _initialize(self):
        self.repo_path.mkdir(parents=True, exist_ok=True)
        
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({'current_version': None, 'versions': []}, f)

    def commit_change(self, description):
        """提交模型变更并创建新版本"""
        try:
            # 生成版本哈希
            model_files = list(Path('models').glob('*.bin'))
            kb_files = list(Path('data/knowledge_base').glob('*'))
            
            hasher = hashlib.sha256()
            for file in model_files + kb_files:
                hasher.update(file.read_bytes())
            version_hash = hasher.hexdigest()[:8]

            # 创建版本元数据
            new_version = {
                'hash': version_hash,
                'timestamp': datetime.now().isoformat(),
                'description': description,
                'model_files': [str(p) for p in model_files],
                'kb_files': [str(p) for p in kb_files]
            }

            # 更新版本记录
            with open(self.metadata_file, 'r+') as f:
                data = json.load(f)
                data['versions'].append(new_version)
                data['current_version'] = version_hash
                f.seek(0)
                json.dump(data, f, indent=2)

            logger.info(f"成功提交新版本: {version_hash} - {description}")
            return True

        except Exception as e:
            logger.error(f"版本提交失败: {e}")
            return False

    def rollback_version(self, target_hash):
        """回滚到指定版本"""
        try:
            with open(self.metadata_file) as f:
                data = json.load(f)

            target_version = next((v for v in data['versions'] if v['hash'] == target_hash), None)
            
            if not target_version:
                raise ValueError(f"未找到版本: {target_hash}")

            # 恢复模型文件
            for path in target_version['model_files']:
                dest = Path(path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                (self.repo_path / target_hash / dest.name).replace(dest)

            # 恢复知识库文件
            for path in target_version['kb_files']:
                dest = Path(path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                (self.repo_path / target_hash / dest.name).replace(dest)

            data['current_version'] = target_hash
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"成功回滚到版本: {target_hash}")
            return True

        except Exception as e:
            logger.error(f"版本回滚失败: {e}")
            return False

    def get_version_history(self):
        """获取版本历史记录"""
        try:
            with open(self.metadata_file) as f:
                return json.load(f)['versions']
        except Exception as e:
            logger.error(f"获取版本历史失败: {e}")
            return []

def init_version_system():
    vcs = VersionControlSystem()
    if not list(vcs.repo_path.glob('*')):
        vcs.commit_change('初始版本')
    logger.info("版本控制系统初始化完成")