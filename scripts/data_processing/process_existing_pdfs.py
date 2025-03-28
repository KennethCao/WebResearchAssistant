import os
import sys
from pathlib import Path
import gc
import torch
from tqdm import tqdm

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.utils.logger import setup_logger
from app.services.paper_service import PaperService
from app.nlp.blockchain_qa import BlockchainQA

logger = setup_logger(__name__)

def process_existing_pdfs():
    """Process existing PDF files"""
    try:
        # 获取PDF目录
        pdf_dir = os.path.join(ROOT_DIR, 'static', 'pdfs')
        logger.info(f"Created PDF directory: {pdf_dir}")
        os.makedirs(pdf_dir)
        
        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        if not pdf_files:
            logger.info("No PDF files found, please upload files to static/pdfs directory")
            return
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 初始化服务
        paper_service = PaperService()
        qa_system = BlockchainQA()
        
        # 处理每个PDF文件
        success_count = 0
        for pdf_file in tqdm(pdf_files, desc="处理PDF文件"):
            try:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                logger.info(f"处理文件: {pdf_file}")
                
                # 解析PDF
                paper_data = paper_service.parse_pdf(pdf_path)
                if not paper_data:
                    logger.warning(f"无法解析文件: {pdf_file}")
                    continue
                
                # 处理PDF文本
                if qa_system.process_pdf(paper_data.get('full_text', '')):
                    success_count += 1
                    logger.info(f"成功处理文件: {pdf_file}")
                
                # 清理内存
                gc.collect()
                
            except Exception as e:
                logger.error(f"处理文件 {pdf_file} 时出错: {e}")
                continue
        
        logger.info(f"Process completed: successfully processed {success_count} files")
        
    except Exception as e:
        logger.error(f"处理PDF文件时出错: {e}")
        raise

if __name__ == '__main__':
    try:
        logger.info("开始处理已存在的PDF文件...")
        process_existing_pdfs()
        logger.info("PDF处理完成")
    except KeyboardInterrupt:
        logger.info("User interrupted processing")
    except Exception as e:
        logger.error(f"Program error: {e}")
        sys.exit(1)