import PyPDF2
import re
from typing import Dict, List, Optional
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextExtractor:
    def __init__(self):
        self.title_patterns = [
            r'^Title:?\s*(.+)$',
            r'^\s*(.+?)\s*\n\s*Abstract\s*$'
        ]
        
        self.abstract_patterns = [
            r'Abstract\s*\n+\s*(.+?)(?=\n*(?:Introduction|Keywords|1\.|$))',
            r'ABSTRACT\s*\n+\s*(.+?)(?=\n*(?:Introduction|Keywords|1\.|$))'
        ]
        
        self.keywords_patterns = [
            r'Keywords?:?\s*(.+?)(?=\n\n|\n[A-Z])',
            r'Index Terms:?\s*(.+?)(?=\n\n|\n[A-Z])'
        ]

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """Extract structured text from PDF"""
        try:
            # Extract raw text
            text = self._read_pdf(pdf_path)
            
            # Extract components
            title = self._extract_title(text)
            abstract = self._extract_abstract(text)
            keywords = self._extract_keywords(text)
            sections = self._extract_sections(text)
            
            return {
                'title': title,
                'abstract': abstract,
                'keywords': keywords,
                'sections': sections,
                'full_text': text
            }
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise

    def _read_pdf(self, pdf_path: str) -> str:
        """Read text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
            
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract paper title"""
        for pattern in self.title_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Extract paper abstract"""
        for pattern in self.abstract_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    def _extract_keywords(self, text: str) -> Optional[str]:
        """Extract paper keywords"""
        for pattern in self.keywords_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        """Extract paper sections"""
        sections = []
        
        # Match section headers and content
        section_pattern = r'(?:^|\n)(\d+\.?\s+[A-Z][^\n]+)\n+((?:(?!\n\d+\.?\s+[A-Z]).+\n*)+)'
        matches = re.finditer(section_pattern, text, re.MULTILINE)
        
        for match in matches:
            sections.append({
                'title': match.group(1).strip(),
                'content': match.group(2).strip()
            })
            
        return sections

    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove headers/footers
        text = re.sub(r'(?m)^\s*.+?\s*\n\s*$', '', text)
        
        return text.strip() 