import os
from typing import Tuple, Dict, Any
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from app.utils.logger import setup_logger
from app.config.environment import PDF_CHUNK_SIZE, PDF_OVERLAP

# 设置日志记录器
logger = setup_logger(__name__)

def extract_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    提取 PDF 文件的元数据。
    
    Args:
        pdf_path (str): PDF 文件路径
        
    Returns:
        Dict[str, Any]: PDF 文件的元数据
    """
    try:
        # 打开 PDF 文件
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            
            # 获取元数据
            metadata = reader.metadata
            if metadata:
                # 清理元数据
                cleaned_metadata = {
                    'title': metadata.get('/Title', ''),
                    'author': metadata.get('/Author', ''),
                    'subject': metadata.get('/Subject', ''),
                    'keywords': metadata.get('/Keywords', ''),
                    'creator': metadata.get('/Creator', ''),
                    'producer': metadata.get('/Producer', ''),
                    'creation_date': metadata.get('/CreationDate', ''),
                    'modification_date': metadata.get('/ModDate', ''),
                    'page_count': len(reader.pages)
                }
            else:
                cleaned_metadata = {
                    'page_count': len(reader.pages)
                }
            
            logger.info(f"Extracted metadata from {os.path.basename(pdf_path)}")
            return cleaned_metadata
            
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        raise

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 文件中提取文本。
    
    Args:
        pdf_path (str): PDF 文件路径
        
    Returns:
        str: 提取的文本内容
    """
    try:
        # 使用 pdfminer 提取文本
        text = extract_text(pdf_path)
        
        if not text.strip():
            logger.warning(f"No text content found in {os.path.basename(pdf_path)}")
            return ""
        
        logger.info(f"Extracted text from {os.path.basename(pdf_path)}")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Failed to extract text: {e}")
        raise

def split_text_into_chunks(text: str, chunk_size: int = PDF_CHUNK_SIZE, overlap: int = PDF_OVERLAP) -> list[str]:
    """
    将文本分割成重叠的块。
    
    Args:
        text (str): 要分割的文本
        chunk_size (int): 每个块的最大大小
        overlap (int): 块之间的重叠大小
        
    Returns:
        list[str]: 文本块列表
    """
    try:
        # 如果文本为空，返回空列表
        if not text.strip():
            return []
        
        # 按句子分割文本
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # 确保句子以句号结束
            sentence = sentence.strip() + '.'
            sentence_size = len(sentence)
            
            # 如果当前块加上新句子超过块大小，保存当前块并开始新块
            if current_size + sentence_size > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # 保留最后几个句子作为重叠部分
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_chunk.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_chunk
                current_size = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Failed to split text into chunks: {e}")
        raise

def parse_pdf(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    解析 PDF 文件，提取文本内容和元数据。
    
    Args:
        pdf_path (str): PDF 文件路径
        
    Returns:
        Tuple[str, Dict[str, Any]]: 提取的文本内容和元数据
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # 提取文本和元数据
        text = extract_text_from_pdf(pdf_path)
        metadata = extract_metadata(pdf_path)
        
        logger.info(f"Successfully parsed PDF: {os.path.basename(pdf_path)}")
        return text, metadata
        
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise

# 示例用法
if __name__ == "__main__":
    # 测试 PDF 文件路径
    test_pdf = "test.pdf"
    
    try:
        # 解析 PDF
        content, meta = parse_pdf(test_pdf)
        
        # 打印元数据
        print("\nMetadata:")
        for key, value in meta.items():
            print(f"{key}: {value}")
        
        # 打印文本内容的一部分
        print("\nContent preview:")
        print(content[:500] + "..." if len(content) > 500 else content)
        
        # 分割文本
        chunks = split_text_into_chunks(content)
        print(f"\nNumber of chunks: {len(chunks)}")
        print("\nFirst chunk preview:")
        print(chunks[0][:200] + "..." if chunks else "No chunks created")
        
    except Exception as e:
        print(f"Error: {e}")