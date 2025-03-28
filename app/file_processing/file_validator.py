import os
from typing import Tuple, Optional
import mimetypes
from app.utils.logger import setup_logger
try:
    import pdfplumber
except ImportError:
    print("The pdfplumber library is not installed. Please install it using 'pip install pdfplumber'.")
from app.config.environment import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH
)

# 设置日志记录器
logger = setup_logger(__name__)

def allowed_file(filename: str) -> bool:
    """Check if the file type is allowed"""
    return filename.lower().endswith('.pdf')

def validate_file_type(file) -> bool:
    """
    验证文件类型。
    
    Args:
        file: 文件对象
        
    Returns:
        bool: 如果文件类型有效则返回 True，否则返回 False
    """
    try:
        # 获取文件的 MIME 类型
        mime = mimetypes.guess_type(file.filename)[0]
        
        # 检查 MIME 类型
        valid_mimes = {
            'application/pdf': ['pdf'],
            'text/plain': ['txt'],
            'application/msword': ['doc'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['docx']
        }
        
        for mime_type, extensions in valid_mimes.items():
            if mime == mime_type and any(ext in ALLOWED_EXTENSIONS for ext in extensions):
                logger.info(f"Valid file type: {mime}")
                return True

        logger.error(f"Unsupported MIME type: {mime}")
        return False
        
    except Exception as e:
        logger.error(f"Failed to validate file type: {e}")
        return False

def validate_file_size(file) -> bool:
    """
    Validate file size compliance
    
    Args:
        file: File object
        
    Returns:
        bool: True if within size limit, False otherwise
    """
    try:
        # 获取文件大小
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)  # 重置文件指针
        
        # 检查文件大小
        if size > MAX_CONTENT_LENGTH:
            logger.error(f"File too large: {size} bytes")
            return False
            
        logger.info(f"Valid file size: {size} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate file size: {e}")
        return False

def validate_file_name(filename: str) -> str:
    """
    验证并清理文件名。
    
    Args:
        filename (str): 原始文件名
        
    Returns:
        str: 清理后的安全文件名
    """
    try:
        # 使用 werkzeug 的 secure_filename 清理文件名
        safe_filename = mimetypes.guess_filename(filename)
        
        if not safe_filename:
            logger.error("Empty filename provided")
        raise ValueError("Filename cannot be empty")
            
        logger.info(f"Sanitized filename successfully: {safe_filename}")
        return safe_filename
        
    except Exception as e:
        logger.error(f"Failed to validate filename: {e}")
        raise

def scan_file_content(file) -> bool:
    """
    扫描文件内容是否安全。
    
    Args:
        file: 文件对象
        
    Returns:
        bool: 如果文件内容安全则返回 True，否则返回 False
    """
    try:
        # 读取文件内容
        content = file.read()
        file.seek(0)  # 重置文件指针
        
        # 检查是否包含可疑内容
        suspicious_patterns = [
            b'<script',
            b'<?php',
            b'#!/',
            b'eval(',
            b'exec('
        ]
        
        for pattern in suspicious_patterns:
            if pattern in content.lower():
                logger.error(f"Suspicious content found: {pattern}")
                return False
        
        logger.info("File content appears safe")
        return True
        
    except Exception as e:
        logger.error(f"Failed to scan file content: {e}")
        return False

def validate_pdf(file_path):
    # Validate PDF format and blockchain content
    BLOCKCHAIN_KEYWORDS = {'blockchain', 'distributed ledger', 'smart contract', 'consensus algorithm'}
    
    # First check file format
    if not file_path.lower().endswith('.pdf'):
        return False, 'invalid_format', 'Only PDF files are allowed'
    
    # Then check content
    try:
        # 尝试导入 pdfplumber 库，如果未安装会抛出 ImportError
    except ImportError:
        return False, 'missing_library', 'The pdfplumber library is not installed.'
    
    try:
        # 引入 pdfplumber 库用于提取 PDF 文本
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
        if not any(kw in text.lower() for kw in BLOCKCHAIN_KEYWORDS):
            return False, 'non_blockchain', 'Document must contain blockchain-related content'
        return True, None, None
    except Exception as e:
        return False, 'corrupted_file', f'Invalid PDF file: {str(e)}'


def validate_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded file

    Args:
        file_path: File path

    Returns:
        Tuple containing validation status and error message
    """
    try:
        # Standardize path handling
        file_path = os.path.expanduser(file_path)
        file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return False, "File does not exist"

        # Get file size in bytes
        file_size = os.path.getsize(file_path)

        # Check file size limit (50MB)
        if file_size > 50 * 1024 * 1024:
            return False, "File too large (exceeds 50MB)"

        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Check allowed extensions
        allowed_extensions = {'.pdf'}
        if ext not in allowed_extensions:
            return False, "Only PDF files are allowed"

        # Verify MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            return False, "Unrecognized file type"

        allowed_mimes = {
            'application/pdf',
            'text/plain',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }

        if mime_type not in allowed_mimes:
            return False, f"Unsupported file type: {mime_type}"

        return True, None

    except Exception as e:
        logger.error(f"File validation error: {e}")
        return False, f"File validation failed: {str(e)}"

# 示例用法
if __name__ == "__main__":
    # 测试 PDF 文件
    test_pdf_path = "path/to/test.pdf"
    if os.path.exists(test_pdf_path):
        is_valid, error = validate_file(test_pdf_path)
        print(f"PDF file is valid: {is_valid}")
        if error:
            print(f"Error: {error}")

# 确保导出函数
__all__ = ['allowed_file', 'validate_file']