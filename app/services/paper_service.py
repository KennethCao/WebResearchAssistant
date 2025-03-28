from app.extensions import db
from app.models.paper import Paper
from app.nlp.text_analyzer import TextAnalyzer
from app.utils.logger import setup_logger
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from app.nlp.blockchain_qa import BlockchainQA
from pathlib import Path
import fitz  # PyMuPDF
import re
from typing import List, Tuple, Dict, Any, Optional
from app.nlp.analyzer import analyze_paper
from flask import current_app
from datetime import datetime
from threading import Thread

logger = setup_logger(__name__)
ROOT_DIR = Path(__file__).parent.parent.parent

class PaperService:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        self.qa_system = BlockchainQA()
        self._jieba = None
        
        # 延迟初始化 jieba
        try:
            import jieba
            import jieba.analyse
            # 修复词典问题
            dict_path = os.path.join(os.path.dirname(jieba.__file__), 'dict.txt')
            if os.path.exists(dict_path):
                # 创建一个临时的清理后的词典
                temp_dict = os.path.join(ROOT_DIR, 'data', 'jieba_dict.txt')
                with open(dict_path, 'r', encoding='utf-8') as f_in:
                    with open(temp_dict, 'w', encoding='utf-8') as f_out:
                        for line in f_in:
                            # 跳过有问题的行
                            if '曾凡军' not in line:
                                f_out.write(line)
                
                # 使用清理后的词典
                jieba.set_dictionary(temp_dict)
                
            self._jieba = jieba
            logger.info("Jieba initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize jieba: {e}")
            self._jieba = None
    
    def process_paper(self, paper_id: int) -> bool:
        """处理论文文件"""
        try:
            paper = Paper.query.get(paper_id)
            if not paper:
                logger.error(f"Paper not found: {paper_id}")
                return False
            
            # 更新状态为处理中
            paper.analysis_status = 'processing'
            db.session.commit()
            
            # 读取文件内容
            file_path = os.path.join(
                os.getenv('BLOCKCHAIN_ASSISTANT_DATA_DIR', 'data'),
                'uploads',
                paper.filename
            )
            
            try:
                # 提取文本内容
                content = self.extract_pdf_content(file_path)
                
                # 更新论文内容
                paper.content = content
                
                # 生成摘要
                paper.summary = self.generate_summary(content)
                
                # 提取关键词
                keywords = self.extract_keywords(content)
                if keywords:
                    paper.keywords = ','.join(keywords)
                
                # 提取主题
                topics = self.analyze_topics(content)
                if topics:
                    paper.topics = ','.join(topics)
                
                # 更新状态
                paper.analysis_status = 'completed'
                paper.error_message = None
                
                # 如果是区块链相关的论文，添加到知识库
                if self.is_blockchain_related(content):
                    self.qa_system.update_knowledge_base(content)
                    paper.added_to_kb = True
                
                # 保存更改
                db.session.commit()
                logger.info(f"Successfully processed paper {paper_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error processing paper content: {e}")
                paper.analysis_status = 'failed'
                paper.error_message = str(e)
                db.session.commit()
                return False
                
        except Exception as e:
            logger.error(f"Failed to process paper {paper_id}: {e}")
            return False
    
    def extract_pdf_content(self, file_path: str) -> str:
        """提取PDF文件内容"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract PDF content: {e}")
            raise
    
    def generate_summary(self, text: str) -> str:
        """生成文本摘要"""
        try:
            # 使用 TextRank 算法提取摘要
            sentences = re.split(r'[。！？]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not sentences:
                return "无法生成摘要"
            
            # 取前3个句子作为摘要
            summary = "。".join(sentences[:3]) + "。"
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "生成摘要时发生错误"
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """提取关键词"""
        try:
            if self._jieba and hasattr(self._jieba, 'analyse'):
                # 使用 jieba 提取关键词
                keywords = self._jieba.analyse.extract_tags(text, topK=top_k)
                return keywords
            else:
                # 使用简单的分词方法作为后备
                words = text.lower().split()
                # 过滤掉短词
                words = [w for w in words if len(w) > 3]
                # 返回最常见的词
                from collections import Counter
                return [w for w, _ in Counter(words).most_common(top_k)]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def analyze_topics(self, text: str) -> List[str]:
        """分析主题"""
        try:
            if not self._jieba:
                return []
            
            # 使用 jieba 提取主题关键词
            topics = self._jieba.analyse.extract_tags(text, topK=5)
            return topics
            
        except Exception as e:
            logger.error(f"Failed to analyze topics: {e}")
            return []
    
    def is_blockchain_related(self, text: str) -> bool:
        """检查文本是否与区块链相关"""
        try:
            # 区块链相关关键词
            blockchain_keywords = {
                "blockchain", "smart contract", "decentralized", "consensus mechanism", "cryptocurrency",
                "bitcoin", "ethereum", "distributed ledger", "digital currency", "mining",
                "public chain", "private chain", "consortium chain", "cross-chain", "sidechain",
                "token", "wallet", "hash", "node", "transaction", "zero-knowledge proof", "oracle",
                "dapp", "defi", "nft", "staking", "governance", "sharding", "layer2", "pow", "pos"
            }
            
            # 使用精确匹配并设置最小阈值
            matched_keywords = [kw for kw in keywords if kw.lower() in blockchain_keywords]
            
            return matched_keywords if len(matched_keywords) >= 5 else []
            
        except Exception as e:
            logger.error(f"Failed to check blockchain relevance: {e}")
            return False

    def save_paper(self, file) -> dict:
        """保存论文文件并创建数据库记录
        
        Args:
            file: 上传的文件对象
            
        Returns:
            dict: 包含处理结果的字典
        """
        try:
            # 确保文件名安全
            filename = secure_filename(file.filename)
            
            # 创建上传目录
            upload_dir = os.path.join(ROOT_DIR, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            
            # 保存文件
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # 提取文本内容
            with open(file_path, 'rb') as pdf_file:
                pdf = PdfReader(pdf_file)
                content = ""
                for page in pdf.pages:
                    content += page.extract_text()
                
                # 即时内容验证
                if not self.validate_blockchain_content(content):
                    os.remove(file_path)
                    return {
                        'success': False,
                        'error': '文献内容与区块链无关，已拒绝上传'
                    }
            
            # 创建论文记录
            paper = Paper(
                filename=filename,
                content=content,
                analysis_status='pending'
            )
            
            db.session.add(paper)
            db.session.commit()
            
            # 启动异步分析任务
            self._start_analysis_task(paper.id)
            
            return {
                'success': True,
                'paper_id': paper.id,
                'message': 'File uploaded and analysis started'
            }
            
        except Exception as e:
            logger.error(f"Failed to save paper: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _start_analysis_task(self, paper_id: int):
        """启动异步分析任务"""
        try:
            def analyze_task():
                with current_app.app_context():
                    try:
                        # 获取论文记录
                        paper = Paper.query.get(paper_id)
                        if not paper:
                            logger.error(f"Paper not found: {paper_id}")
                            return
                        
                        # 更新状态
                        paper.analysis_status = 'processing'
                        db.session.commit()
                        
                        # 分析文本
                        summary = self.analyzer.generate_summary(paper.content)
                        keywords = self.analyzer.extract_keywords(paper.content)
                        topics = self.analyzer.extract_topics(paper.content)
                        
                        # 更新论文记录
                        paper.summary = summary
                        paper.keywords = ','.join(keywords)
                        paper.topics = ','.join(topics)
                        paper.analysis_status = 'completed'
                        
                        # 添加到知识库
                        if self.is_blockchain_related(paper.content):
                            self.qa_system.add_to_knowledge_base(paper.content)
                            paper.added_to_kb = True
                        
                        db.session.commit()
                        if paper.added_to_kb:
                            self.qa_system.reload_knowledge_base()
                        logger.info(f"Analysis completed for paper {paper_id}")
                        
                    except Exception as e:
                        logger.error(f"Analysis failed for paper {paper_id}: {e}")
                        paper.analysis_status = 'error'
                        paper.error_message = str(e)
                        db.session.commit()
            
            # 启动分析线程
            thread = Thread(target=analyze_task)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start analysis task: {e}")

    def validate_blockchain_content(self, text: str) -> List[str]:
        """验证并提取区块链相关关键词"""
        try:
            # 使用延迟加载的 jieba
            keywords = self._jieba.analyse.extract_tags(text, topK=20)
            
            # 区块链相关关键词列表
            blockchain_keywords = {
                "区块链", "智能合约", "去中心化", "共识机制", "加密货币",
                "比特币", "以太坊", "分布式账本", "数字货币", "挖矿",
                "公链", "私链", "联盟链", "跨链", "侧链",
                "代币", "钱包", "哈希", "节点", "交易"
            }
            
            # 过滤出区块链相关关键词
            blockchain_related = [
                kw for kw in keywords 
                if any(bk in kw or kw in bk for bk in blockchain_keywords)
            ]
            
            return blockchain_related[:10]  # 返回前10个关键词
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []

    def process_pdf(self, file_path: str) -> str:
        """处理PDF文件"""
        try:
            # 使用 PyMuPDF 提取文本
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            if not text.strip():
                raise ValueError("PDF文件中没有提取到文本")
                
            # 确保文本是有意义的
            if len(text.strip()) < 100:  # 如果文本太短
                raise ValueError("提取的文本内容太少")
                
            logger.info(f"成功从PDF提取文本，长度: {len(text)}")
            
            # 初始化知识库（如果需要）
            if not hasattr(self.qa_system, 'knowledge_base'):
                self.qa_system.initialize_knowledge_base()
            
            return text
            
        except Exception as e:
            logger.error(f"处理PDF文件失败: {e}")
            raise

    def analyze_paper(self, file_path: str, paper_id: str = None) -> dict:
        """分析论文内容"""
        try:
            # 提取文本
            text = self.process_pdf(file_path)
            if not text:
                raise ValueError("无法从PDF中提取文本")
            
            logger.info(f"成功提取文本，长度: {len(text)}")
            
            # 添加到知识库
            added_to_kb = self.qa_system.process_pdf(text)
            if not added_to_kb:
                logger.warning("文献未能成功添加到知识库")
            else:
                logger.info("文献已成功添加到知识库")
            
            # 提取关键词
            keywords = self.validate_blockchain_content(text)
            
            # 生成摘要
            summary = self.generate_summary(text)
            
            # 分析主题
            topics = self.analyze_topics(text)
            
            # 保存到数据库
            if paper_id:
                paper = Paper.query.get(paper_id)
                if paper:
                    paper.keywords = ','.join(keywords)
                    paper.summary = summary
                    paper.topics = ','.join(topics)
                    paper.analysis_status = 'completed'
                    paper.added_to_kb = added_to_kb
                    db.session.commit()
                    logger.info(f"分析结果已保存到数据库: {paper_id}")
            
            return {
                'status': 'completed',
                'keywords': keywords,
                'summary': summary,
                'topics': topics,
                'added_to_kb': added_to_kb
            }
            
        except Exception as e:
            logger.error(f"论文分析失败: {e}")
            if paper_id:
                paper = Paper.query.get(paper_id)
                if paper:
                    paper.analysis_status = 'failed'
                    paper.error_message = str(e)
                    db.session.commit()
            raise
    
    def get_paper(self, paper_id: str):
        """获取论文信息"""
        try:
            paper = Paper.query.get(paper_id)
            if not paper:
                logger.warning(f"未找到论文: {paper_id}")
                return None
            
            logger.info(f"成功获取论文: {paper_id}")
            return paper
            
        except Exception as e:
            logger.error(f"获取论文失败: {e}")
            raise
    
    def get_recent_papers(self, limit=10):
        """Get recent papers"""
        try:
            papers = Paper.query.order_by(
                Paper.uploaded_at.desc()
            ).limit(limit).all()
            return papers
            
        except Exception as e:
            logger.error(f"Failed to get recent papers: {e}")
            raise 

    def parse_pdf(self, pdf_path: str) -> dict:
        """解析PDF文件"""
        try:
            logger.info(f"开始解析PDF文件: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF文件不存在: {pdf_path}")
                return None
                
            # 打开PDF文件
            doc = fitz.open(pdf_path)
            
            # 提取文本
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            # 清理文本
            full_text = self._clean_text(full_text)
            
            # 提取标题（假设第一页的第一段是标题）
            title = doc[0].get_text().strip().split('\n')[0]
            
            # 尝试提取摘要
            abstract = self._extract_abstract(full_text)
            
            # 尝试提取关键词
            keywords = self._extract_keywords(full_text)
            
            doc.close()
            
            result = {
                'title': title,
                'abstract': abstract,
                'keywords': keywords,
                'full_text': full_text,
                'file_path': pdf_path
            }
            
            logger.info(f"成功解析PDF文件: {os.path.basename(pdf_path)}")
            logger.info(f"提取的文本长度: {len(full_text)} 字符")
            
            return result
            
        except Exception as e:
            logger.error(f"解析PDF文件时出错: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,?!;:()\[\]{}\'\""-]', '', text)
        return text.strip()
    
    def _extract_abstract(self, text: str) -> str:
        """提取摘要"""
        try:
            # 尝试找到摘要部分
            abstract_patterns = [
                r'Abstract[:\s]+(.*?)(?=\n\n|\d+\.?\s+Introduction)',
                r'摘要[：:\s]+(.*?)(?=\n\n|关键词|引言)',
            ]
            
            for pattern in abstract_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # 如果找不到明确的摘要，返回前500个字符
            return text[:500].strip()
            
        except Exception as e:
            logger.error(f"提取摘要时出错: {e}")
            return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        try:
            # 尝试从关键词部分提取
            keyword_patterns = [
                r'Keywords?[:\s]+(.*?)(?=\n\n|\d\.)',
                r'关键词[：:\s]+(.*?)(?=\n\n|\d\.)',
                r'Index Terms[:\s]+(.*?)(?=\n\n|\d\.)'
            ]
            
            for pattern in keyword_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    keywords = match.group(1).strip()
                    # 分割关键词
                    return [k.strip() for k in re.split(r'[,;，；]', keywords) if k.strip()]
            
            # 如果没有找到关键词部分，使用文本分析提取
            from rake_nltk import Rake
            r = Rake()
            r.extract_keywords_from_text(text)
            
            # 获取前10个关键词
            keywords = []
            for rating, keyword in r.get_ranked_phrases_with_scores()[:10]:
                if rating > 5.0:  # 只保留评分较高的关键词
                    keywords.append(keyword)
                
            return keywords
            
        except Exception as e:
            logger.error(f"Error updating paper status: {str(e)}")
            return []

    def get_trend_data(self):
        """获取研究趋势数据"""
        try:
            # 获取所有已分析完成的论文
            papers = Paper.query.filter_by(analysis_status='completed').all()
            
            # 初始化返回数据
            trend_data = {
                'topics': {},
                'recent_papers': [],
                'total_papers': len(papers)
            }
            
            if not papers:
                return trend_data
            
            # 处理主题数据
            all_topics = []
            for paper in papers:
                if paper.topics:
                    topics = [t.strip() for t in paper.topics.split(',') if t.strip()]
                    all_topics.extend(topics)
            
            # 计算主题频率
            from collections import Counter
            topic_counter = Counter(all_topics)
            
            # 计算权重 (确保最小字体为1em，最大字体为3em)
            if topic_counter:
                max_count = max(topic_counter.values())
                trend_data['topics'] = {
                    topic: 1 + (count / max_count) * 2
                    for topic, count in topic_counter.items()
                }
            
            # 准备最近论文数据（按上传时间排序）
            sorted_papers = sorted(papers, key=lambda x: x.uploaded_at, reverse=True)
            for paper in sorted_papers[:10]:  # 最近10篇
                paper_data = {
                    'id': paper.id,
                    'title': paper.title or paper.filename,
                    'keywords': paper.keywords,
                    'uploaded_at': paper.uploaded_at,
                    'summary': paper.summary[:200] + '...' if paper.summary else None
                }
                trend_data['recent_papers'].append(paper_data)
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Failed to get trend data: {e}")
            return {
                'topics': {},
                'recent_papers': [],
                'total_papers': 0
            }

    def _extract_title(self, text: str) -> str:
        """提取PDF文档标题"""
        try:
            # 尝试从文本开始部分提取标题
            lines = text.split('\n')
            for line in lines[:10]:  # 只检查前10行
                line = line.strip()
                if len(line) > 10 and len(line) < 200:  # 标题长度通常在这个范围内
                    # 排除一些常见的非标题行
                    if not any(x in line.lower() for x in ['abstract', 'introduction', 'contents']):
                        return line
            return ""
        except Exception as e:
            logger.error(f"Error extracting title: {e}")
            return ""

    def _extract_topics_and_technologies(self, text: str) -> Tuple[List[str], List[str]]:
        """提取主题和技术"""
        try:
            # 区块链相关主题
            blockchain_topics = [
                "Blockchain Technology", "Smart Contracts", "Cryptocurrency",
                "Consensus Mechanisms", "DeFi", "NFTs", "Web3",
                "Distributed Systems", "Security", "Privacy",
                "区块链技术", "智能合约", "加密货币", "共识机制",
                "去中心化金融", "非同质化代币", "分布式系统", "安全性", "隐私保护"
            ]
            
            # 区块链相关技术
            blockchain_technologies = [
                "Ethereum", "Bitcoin", "Hyperledger", "Solidity", "IPFS",
                "Zero-Knowledge Proof", "PoW", "PoS", "DPoS", "BFT",
                "Layer 2", "Rollups", "Sidechains", "State Channels",
                "以太坊", "比特币", "超级账本", "零知识证明", "工作量证明",
                "权益证明", "拜占庭容错", "侧链", "状态通道"
            ]
            
            # 在文本中查找主题和技术
            found_topics = []
            found_technologies = []
            
            text_lower = text.lower()
            
            for topic in blockchain_topics:
                if topic.lower() in text_lower:
                    found_topics.append(topic)
                
            for tech in blockchain_technologies:
                if tech.lower() in text_lower:
                    found_technologies.append(tech)
            
            return (
                found_topics if found_topics else ["Blockchain"],
                found_technologies
            )
            
        except Exception as e:
            logger.error(f"Error extracting topics and technologies: {e}")
            return (["Blockchain"], [])

    def _extract_topics(self, text: str) -> List[str]:
        """提取主题"""
        try:
            # 区块链相关主题列表
            blockchain_topics = [
                "Blockchain Technology", "Smart Contracts", "Cryptocurrency",
                "Consensus Mechanisms", "DeFi", "NFTs", "Web3",
                "Distributed Systems", "Security", "Privacy",
                "区块链技术", "智能合约", "加密货币", "共识机制",
                "去中心化金融", "非同质化代币", "分布式系统", "安全性", "隐私保护"
            ]
            
            # 在文本中查找主题
            found_topics = []
            for topic in blockchain_topics:
                if topic.lower() in text.lower():
                    found_topics.append(topic)
            
            return found_topics if found_topics else ["Blockchain"]
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return ["Blockchain"]