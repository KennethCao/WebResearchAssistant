# app/database/models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from . import Base

class UploadedFile(Base):
    """
    上传文件表：存储上传的 PDF 文件的元信息
    """
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 文件唯一 ID
    filename = Column(String(255), nullable=False, index=True)  # 文件名
    filepath = Column(String(255), nullable=False)  # 文件存储路径
    upload_time = Column(DateTime, default=datetime.utcnow)  # 上传时间
    metadata = Column(Text, nullable=True)  # 提取的元信息（JSON 格式）

    # 关系 - 文件与分析结果
    analysis_results = relationship("AnalysisResult", back_populates="uploaded_file")

    def __repr__(self):
        return f"<UploadedFile(id={self.id}, filename={self.filename})>"

class AnalysisResult(Base):
    """
    分析结果表：存储 PDF 文件的分析结果
    """
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 分析结果唯一 ID
    file_id = Column(Integer, ForeignKey("uploaded_files.id"), nullable=False, index=True)  # 关联的文件 ID
    summary = Column(Text, nullable=True)  # 摘要
    keywords = Column(Text, nullable=True)  # 关键词（以逗号分隔）
    created_at = Column(DateTime, default=datetime.utcnow)  # 结果生成时间

    # 关系 - 关联文件
    uploaded_file = relationship("UploadedFile", back_populates="analysis_results")

    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, file_id={self.file_id})>"

class User(Base):
    """
    用户表：存储用户信息（可选）
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)  # 用户唯一 ID
    username = Column(String(255), unique=True, nullable=False, index=True)  # 用户名
    email = Column(String(255), unique=True, nullable=False, index=True)  # 邮箱
    password_hash = Column(String(255), nullable=False)  # 密码哈希值
    created_at = Column(DateTime, default=datetime.utcnow)  # 用户创建时间

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"

# 添加索引
Index('idx_uploaded_files_filename', UploadedFile.filename)
Index('idx_analysis_results_file_id', AnalysisResult.file_id)
Index('idx_users_username', User.username)
Index('idx_users_email', User.email)