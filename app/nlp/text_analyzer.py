from typing import Dict, List, Any
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from app.utils.logger import setup_logger
from app.config.environment import (
    MODEL_NAME,
    MODEL_CACHE_DIR,
    DEVICE,
    BLOCKCHAIN_EMBEDDING_DIMENSION
)
import os

logger = setup_logger(__name__)

class TextAnalyzer:
    """文本分析器类"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TextAnalyzer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                self.model = self._load_model()
                self._initialized = True
                logger.info("Text analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize text analyzer: {e}")
                self.model = self._create_simple_model()
                self._initialized = True

    def _load_model(self):
        """加载模型"""
        try:
            # 尝试加载本地模型
            local_model_path = os.path.join(MODEL_CACHE_DIR, MODEL_NAME)
            if os.path.exists(local_model_path):
                try:
                    # 延迟导入以避免早期导入错误
                    from transformers import AutoModel, AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                    model = AutoModel.from_pretrained(local_model_path)
                    model.to(DEVICE)
                    return {"model": model, "tokenizer": tokenizer}
                except Exception as e:
                    logger.error(f"Error loading transformer model: {e}")
                    return self._create_simple_model()
            return self._create_simple_model()
        except Exception as e:
            logger.error(f"Error in _load_model: {e}")
            return self._create_simple_model()

    def _create_simple_model(self):
        """创建简单的平均池化模型"""
        from torch import nn
        
        class SimpleModel(nn.Module):
            def __init__(self, embedding_dim=BLOCKCHAIN_EMBEDDING_DIMENSION):
                super().__init__()
                self.embedding_dim = embedding_dim
                
            def encode(self, texts: List[str], **kwargs) -> torch.Tensor:
                if isinstance(texts, str):
                    texts = [texts]
                return torch.zeros((len(texts), self.embedding_dim))

        return {"model": SimpleModel(), "tokenizer": None}

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """分析文本"""
        try:
            # 生成文本嵌入
            embedding = self.encode_text(text)
            
            # 返回分析结果
            return {
                "embedding": embedding,
                "length": len(text),
                "success": True
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "embedding": torch.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION),
                "length": 0,
                "success": False,
                "error": str(e)
            }

    def encode_text(self, text: str) -> torch.Tensor:
        """生成文本嵌入向量"""
        try:
            if isinstance(self.model["model"], torch.nn.Module):
                if self.model["tokenizer"]:
                    inputs = self.model["tokenizer"](text, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = self.model["model"](**inputs)
                    return torch.mean(outputs.last_hidden_state, dim=1)
                return torch.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)
            return torch.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return torch.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)

    def batch_encode(self, texts: List[str]) -> torch.Tensor:
        """批量生成文本嵌入向量"""
        try:
            return torch.stack([self.encode_text(text) for text in texts])
        except Exception as e:
            # 使用更小的模型并优化内存使用
            self.device = torch.device('cpu')
            logger.info(f"Using device: {self.device}")
            
            # 暂时跳过模型加载，只提供基本功能
            self.summarizer = None
            self.classifier = None
            
            # 设置模型参数
            self.model_kwargs = {
                'low_cpu_mem_usage': True,
                'torch_dtype': torch.float32
            }
            
            logger.info("TextAnalyzer initialized with basic functionality")
            
        except Exception as e:
            logger.error(f"Failed to initialize TextAnalyzer: {e}")
            raise

    def load_models(self):
        """延迟加载模型"""
        try:
            if self.summarizer is None:
                from transformers import pipeline
                self.summarizer = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-6-6",  # 使用更小的模型
                    device=self.device,
                    **self.model_kwargs
                )
            
            if self.classifier is None:
                from transformers import pipeline
                self.classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased",  # 使用更小的模型
                    device=self.device,
                    **self.model_kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def analyze_paper(self, text_data: Dict[str, str]) -> Dict[str, Any]:
        """Analyze paper content"""
        try:
            results = {
                'title': text_data.get('title'),
                'abstract': text_data.get('abstract'),
                'keywords': text_data.get('keywords')
            }
            
            # Generate summary
            if text_data.get('full_text'):
                results['summary'] = self._generate_summary(text_data['full_text'])
            
            # Extract key findings
            results['key_findings'] = self._extract_key_findings(text_data)
            
            # Extract methodology
            results['methodology'] = self._extract_methodology(text_data)
            
            # Extract innovations
            results['innovations'] = self._extract_innovations(text_data)
            
            # Generate embeddings
            results['embedding'] = self._generate_embedding(text_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze paper: {e}")
            raise

    def _generate_summary(self, text: str, max_length: int = 150) -> str:
        """Generate paper summary"""
        try:
            # Split text into chunks if too long
            chunks = self._split_text(text)
            summaries = []
            
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            
            return " ".join(summaries)
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return ""

    def _extract_key_findings(self, text_data: Dict[str, str]) -> List[str]:
        """Extract key findings from paper"""
        try:
            findings = []
            
            # Look for findings in conclusion section
            sections = text_data.get('sections', [])
            conclusion = next(
                (s for s in sections if 'conclusion' in s['title'].lower()),
                None
            )
            
            if conclusion:
                # Use zero-shot classification
                sentences = conclusion['content'].split('.')
                for sentence in sentences:
                    if len(sentence.strip()) < 10:
                        continue
                        
                    result = self.zero_shot(
                        sentence,
                        candidate_labels=['finding', 'result', 'contribution', 'other']
                    )
                    
                    if result['labels'][0] in ['finding', 'result', 'contribution']:
                        findings.append(sentence.strip())
            
            return findings[:5]  # Return top 5 findings
            
        except Exception as e:
            logger.error(f"Failed to extract findings: {e}")
            return []

    def _extract_methodology(self, text_data: Dict[str, str]) -> str:
        """Extract methodology description"""
        try:
            # Look for methodology section
            sections = text_data.get('sections', [])
            method = next(
                (s for s in sections if any(x in s['title'].lower() 
                    for x in ['method', 'approach', 'implementation'])),
                None
            )
            
            if method:
                # Generate concise summary
                return self._generate_summary(method['content'], max_length=100)
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to extract methodology: {e}")
            return ""

    def _extract_innovations(self, text_data: Dict[str, str]) -> List[str]:
        """Extract innovative aspects"""
        try:
            innovations = []
            
            # Look in introduction and proposed method sections
            sections = text_data.get('sections', [])
            relevant_sections = [
                s for s in sections if any(x in s['title'].lower() 
                    for x in ['introduction', 'proposed', 'novel'])
            ]
            
            for section in relevant_sections:
                sentences = section['content'].split('.')
                for sentence in sentences:
                    if len(sentence.strip()) < 10:
                        continue
                        
                    result = self.zero_shot(
                        sentence,
                        candidate_labels=['innovation', 'contribution', 'novelty', 'other']
                    )
                    
                    if result['labels'][0] != 'other':
                        innovations.append(sentence.strip())
            
            return innovations[:3]  # Return top 3 innovations
            
        except Exception as e:
            logger.error(f"Failed to extract innovations: {e}")
            return []

    def _generate_embedding(self, text_data: Dict[str, str]) -> np.ndarray:
        """Generate paper embedding"""
        try:
            # Combine important sections
            text = " ".join([
                text_data.get('title', ''),
                text_data.get('abstract', ''),
                text_data.get('keywords', '')
            ])
            
            # 使用 transformers 生成嵌入
            if isinstance(self.model["model"], torch.nn.Module) and self.model["tokenizer"]:
                inputs = self.model["tokenizer"](
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model["model"](**inputs)
                    # 使用 [CLS] token 的输出作为嵌入
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    return embedding.squeeze()
            
            # 如果模型不可用，返回零向量
            return np.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(BLOCKCHAIN_EMBEDDING_DIMENSION)  # Return zero vector on error

    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks 

    def analyze_text(self, text: str) -> dict:
        """Analyze text content"""
        try:
            results = {}
            
            # 生成摘要
            if self.summarizer:
                try:
                    summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
                    results['summary'] = summary[0]['summary_text']
                except Exception as e:
                    logger.error(f"Summarization failed: {e}")
                    results['summary'] = text[:200] + "..."  # Fallback to text truncation
            else:
                results['summary'] = text[:200] + "..."
            
            # 分类
            if self.classifier:
                try:
                    classification = self.classifier(text[:512])  # 限制输入长度
                    results['classification'] = classification[0]
                except Exception as e:
                    logger.error(f"Classification failed: {e}")
                    results['classification'] = {'label': 'unknown', 'score': 0.0}
            else:
                results['classification'] = {'label': 'unknown', 'score': 0.0}
            
            return results
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                'summary': text[:200] + "...",
                'classification': {'label': 'unknown', 'score': 0.0}
            }
    
    def extract_keywords(self, text: str) -> list:
        """提取关键词"""
        try:
            # 简单的关键词提取实现
            words = text.lower().split()
            # 过滤掉常见词和短词
            keywords = [w for w in words if len(w) > 3]
            return list(set(keywords[:10]))  # 返回前10个唯一关键词
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []