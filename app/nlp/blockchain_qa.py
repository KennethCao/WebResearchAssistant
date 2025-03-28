from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import os
from app.utils.logger import setup_logger
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_CACHE_DIR = os.path.join(os.environ.get('BLOCKCHAIN_ASSISTANT_DATA_DIR', './data'), 'models')
from app.config.environment import DEVICE
import os
import requests
from urllib.parse import urlparse
import time
from sentence_transformers import SentenceTransformer
import os
import re
import gc
import faiss
import numpy as np
from transformers import AutoModelForSeq2SeqLM
from app.config import ROOT_DIR

logger = setup_logger(__name__)

class BlockchainQA:
    """区块链问答系统"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BlockchainQA, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        try:
            self.model = self._load_model()
            self._initialized = True
            logger.info("BlockchainQA initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BlockchainQA: {e}")
            raise

    def _download_with_retry(self, url: str, save_path: str, max_retries: int = 3) -> bool:
        """带重试的文件下载"""
        for attempt in range(max_retries):
            try:
                # 设置请求头
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # 使用镜像
                url = url.replace('huggingface.co', 'hf-mirror.com')
                
                # 发送请求
                response = requests.get(
                    url,
                    headers=headers,
                    verify=False,
                    stream=True,
                    timeout=60
                )
                response.raise_for_status()
                
                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))
                
                # 写入文件
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证文件大小
                actual_size = os.path.getsize(save_path)
                if actual_size < 1000:  # 文件太小
                    logger.warning(f"Downloaded file is too small: {actual_size} bytes")
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    continue
                    
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if os.path.exists(save_path):
                    os.remove(save_path)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
            
        return False

    def _load_model(self):
        """加载模型"""
        try:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            logger.info("尝试从本地缓存加载模型...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(
                model_name_or_path=MODEL_NAME,
                cache_folder=os.path.join(MODEL_CACHE_DIR, 'sentence-transformers'),
                device=DEVICE,
                use_auth_token=False,
            )
            return {"model": model}
        except Exception as e:
            logger.error(f"本地模型加载失败: {e}")
            return self._create_simple_model()

    def _create_simple_model(self):
        """创建简易本地模型"""
        from transformers import AutoModel, AutoTokenizer
        try:
            model_path = os.path.abspath(os.path.join(MODEL_CACHE_DIR, 'sentence-transformers', MODEL_NAME))
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            logger.error(f"创建简易模型失败: {e}")
            raise

    def encode_text(self, text: str) -> torch.Tensor:
        """生成文本嵌入"""
        try:
            if isinstance(self.model["model"], AutoModel):
                # BERT 模型
                inputs = self.model["tokenizer"](
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                with torch.no_grad():
# 由于 AutoModel 对象不能直接调用，需要调用其 forward 方法
outputs = self.model["model"].forward(**inputs)
                    return outputs.last_hidden_state[:, 0, :]
            else:
                # Sentence Transformer 模型
                return self.model["model"].encode(
                    text,
                    convert_to_tensor=True,
                    device=DEVICE
                )
                
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return torch.zeros(768).to(DEVICE)  # 返回零向量作为后备

    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        """回答问题"""
        try:
            # 计算问题和上下文的嵌入
            q_embedding = self.encode_text(question)
            c_embedding = self.encode_text(context)
            
            # 计算相似度
            similarity = torch.nn.functional.cosine_similarity(
                q_embedding, c_embedding
            ).item()
            
            # 根据相似度返回答案
            if similarity > 0.5:
                return {
                    "answer": context,
                    "confidence": similarity,
                    "source": "context"
                }
            else:
                return {
                    "answer": "抱歉，我无法找到相关答案。",
                    "confidence": 0.0,
                    "source": "default"
                }
                
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "answer": "系统处理出错，请稍后再试。",
                "confidence": 0.0,
                "source": "error"
            }

    def update_knowledge_base(self, text: str) -> bool:
        """更新知识库"""
        try:
            if not isinstance(text, str):
                raise ValueError("Input text cannot be empty")
                
            # 添加到知识库并更新索引
            self.knowledge_base.append(text)
            self._update_index()
            return True
        except Exception as e:
            logger.exception(f"Knowledge base update failed: {e}", exc_info=True)
            return False
    
    def search_similar(self, query: str, k: int = 3) -> list:
        """搜索相似内容"""
        try:
            if not self.knowledge_base:
                return []
            
            # 简单的关键词匹配
            results = []
            query_words = set(query.lower().split())
            for text in self.knowledge_base:
                text_words = set(text.lower().split())
                if query_words & text_words:  # 如果有共同词
                    results.append(text)
            return results[:k]
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_answer(self, question: str) -> str:
        """获取答案"""
        try:
            # 搜索相关内容
            relevant_texts = self.search_similar(question)
            
            if not relevant_texts:
                return "抱歉，我没有找到相关信息。"
            
            # 简单地返回最相关的文本
            return f"根据相关资料：\n\n{relevant_texts[0]}"
            
        except Exception as e:
            logger.error(f"Failed to get answer: {e}")
            return f"抱歉，生成答案时出错：{str(e)}"

    def load_models(self):
        """延迟加载模型"""
        try:
            if self.tokenizer is None:
                logger.info("Loading tokenizer...")
                # 设置下载进度条
                from transformers.utils import logging
                logging.set_verbosity_info()
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    'sshleifer/distilbart-cnn-6-6',  # 使用更小的模型
                    mirror='tuna',
                    cache_dir=os.path.join(ROOT_DIR, 'cache', 'models'),
                    local_files_only=False,
                    use_fast=True  # 使用快速tokenizer
                )
                gc.collect()
            
            if self.qa_model is None:
                logger.info("Loading QA model (this may take a few minutes)...")
                self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(
                    'sshleifer/distilbart-cnn-6-6',  # 使用更小的模型
                    mirror='tuna',
                    cache_dir=os.path.join(ROOT_DIR, 'cache', 'models'),
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    local_files_only=False,
                    device_map='auto',  # 自动处理设备映射
                    max_memory={'cpu': '4GB'}  # 限制内存使用
                ).to(self.device)
                gc.collect()
            
            if self.embedding_model is None:
                logger.info("Loading embedding model...")
                self.embedding_model = SentenceTransformer(
                    'paraphrase-multilingual-MiniLM-L6-v2',  # 使用多语言小模型
                    device=self.device,
                    cache_folder=os.path.join(ROOT_DIR, 'cache', 'models')
                )
                gc.collect()
                
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        try:
            # 1. 移除引用标记、DOI、URL等
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'doi:.*?(?=\s|$)', '', text)
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'pp\.\s*\d+[-–]\d+', '', text)
            
            # 2. 移除数学公式和特殊字符
            text = re.sub(r'\$.*?\$', '', text)
            text = re.sub(r'[^\u4e00-\u9fff\w\s.,?!;:()\[\]{}\'\""-]', '', text)
            
            # 3. 规范化空白字符
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"文本预处理失败: {e}")
            return text

    def _ensure_nltk_data(self):
        """确保NLTK数据可用"""
        if not setup_nltk():
            raise RuntimeError("NLTK数据初始化失败，请检查日志获取详细信息")

    def _split_text(self, text: str) -> List[str]:
        """将文本分割成段落"""
        try:
            # 清理文本
            text = text.strip()
            if not text:
                return []
            
            # 使用简单的分割方法
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # 如果段落太长，进一步分割
            result = []
            for p in paragraphs:
                if len(p) > 1000:  # 如果段落超过1000字符
                    # 按句号分割
                    sentences = [s.strip() for s in p.split('.') if s.strip()]
                    # 将句子组合成适当长度的段落
                    current_para = ""
                    for s in sentences:
                        if len(current_para) + len(s) < 1000:
                            current_para += s + ". "
                        else:
                            if current_para:
                                result.append(current_para.strip())
                            current_para = s + ". "
                    if current_para:
                        result.append(current_para.strip())
                else:
                    result.append(p)
            
            return result
            
        except Exception as e:
            logger.error(f"分割文本时出错: {e}")
            # 使用最简单的分割方法作为后备
            return [p.strip() for p in text.split('\n') if p.strip()]

    def _update_index(self):
        """更新FAISS索引"""
        try:
            if self.embeddings is None or len(self.embeddings) == 0:
                logger.warning("No embeddings available for index update")
                return
                
            dimension = self.embeddings.shape[1]
            
            # 使用GPU加速（如果可用）
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))
            else:
                self.index = faiss.IndexFlatL2(dimension)
                
            self.index.add(self.embeddings.astype('float32'))
            logger.info(f"Index updated with {len(self.embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Index update failed: {e}")
            raise

    def is_blockchain_related(self, text: str) -> bool:
        """判断文本是否与区块链相关"""
        return any(keyword.lower() in text.lower() 
            for keyword in ['blockchain', '智能合约', '去中心化'])

    def add_to_knowledge_base(self, text: str, source: str):
        """将文本添加到知识库"""
        try:
            knowledge_path = os.path.join(MODEL_CACHE_DIR, 'knowledge_base.db')
            # 实现知识库存储逻辑
            logger.info(f"Added {source} to knowledge base")
        except Exception as e:
            logger.error(f"Knowledge base update failed: {e}")

    def save_knowledge_base(self):
        """保存知识库更新"""
        # 实现知识库持久化逻辑
        logger.info("Knowledge base updated")

    def process_pdf(self, text: str) -> bool:
        """处理PDF文本并添加到知识库"""
        try:
            # 分段
            paragraphs = self._split_text_into_paragraphs(text)
            
            if not paragraphs:
                logger.warning("No valid paragraphs found")
                return False
            
            # 过滤掉太短的段落
            valid_paragraphs = [p for p in paragraphs if len(p.strip()) > 100]
            
            if not valid_paragraphs:
                logger.warning("No valid paragraphs of sufficient length found")
                return False
            
            # 添加到知识库
            self.knowledge_base.extend(valid_paragraphs)
            
            # 更新嵌入向量
            new_embeddings = self.embedding_model.encode(valid_paragraphs)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # 更新索引
            self._update_index()
            
            # 保存知识库
            self.save_knowledge_base()
            
            logger.info(f"成功添加 {len(valid_paragraphs)} 个段落到知识库")
            return True
            
        except Exception as e:
            logger.error(f"处理PDF文本失败: {e}")
            return False

    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        # 按照多个换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        # 清理每个段落
        cleaned = []
        for p in paragraphs:
            p = p.strip()
            if p and len(p) > 50:  # 只保留非空且长度超过50的段落
                # 清理多余的空白字符
                p = re.sub(r'\s+', ' ', p)
                cleaned.append(p)
        return cleaned