# 删除这个文件的内容，因为我们已经在 web.py 中实现了上传功能
from flask import Blueprint

# 保留蓝图定义
upload_bp = Blueprint('upload', __name__) 