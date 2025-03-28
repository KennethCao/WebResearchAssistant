try:
    from flask_migrate import Migrate
    from flask_sqlalchemy import SQLAlchemy
except ImportError:
    print("无法导入所需模块，请确保你已经安装了 flask_migrate 和 flask_sqlalchemy。")
    print("你可以使用以下命令来安装它们：")
    print("pip install flask_migrate flask_sqlalchemy")
try:
    from flask_sqlalchemy import SQLAlchemy
except ImportError:
    print("无法导入 flask_sqlalchemy 模块，请确保你已经安装了它。")
    print("你可以使用以下命令来安装：")
    print("pip install flask_sqlalchemy")
    # 这里可以添加尝试自动安装的代码
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask_sqlalchemy"])
        from flask_sqlalchemy import SQLAlchemy
        print("flask_sqlalchemy 安装成功并已导入。")
    except subprocess.CalledProcessError:
        print("自动安装 flask_sqlalchemy 失败，请手动安装。")
from app import create_app
from app.config import Config

class MigrationManager:
    def __init__(self):
        self.app = create_app(Config())
        self.db = SQLAlchemy(self.app)
        self.migrate = Migrate(self.app, self.db)

    def init_migrations(self):
        """初始化迁移仓库"""
        with self.app.app_context():
            self.db.create_all()

    def create_migration(self, message):
        """创建新迁移版本"""
        with self.app.app_context():
            self.db.session.commit()
            self.migrate.migrate( message=message)

    def apply_migrations(self):
        """应用所有未执行的迁移"""
        with self.app.app_context():
            self.migrate.upgrade()

# 集成到Flask CLI
migration_manager = MigrationManager()