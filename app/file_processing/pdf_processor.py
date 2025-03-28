from PyPDF2 import PdfReader
from typing import Dict
import os
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def extract_text_from_pdf(file_path: str, max_pages: int = None) -> Dict:
    """Extract text from PDF file"""
    try:
        logger.info(f"Starting PDF extraction: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        text = ""
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            # Process all pages if max_pages is None
            pages_to_process = len(pdf.pages) if max_pages is None else min(max_pages, len(pdf.pages))
            
            for i in range(pages_to_process):
                try:
                    page = pdf.pages[i]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    logger.info(f"Processed page {i+1}/{pages_to_process}")
                except Exception as e:
                    logger.error(f"Error on page {i}: {e}")
                    continue
                    
        if not text:
            raise ValueError("No text extracted from PDF")
            
        logger.info(f"Successfully extracted {len(text)} characters")
        return {
            "full_text": text.strip(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise 