from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from app.utils.logger import setup_logger
from app.nlp.summarizer import TextSummarizer
from app.utils.analysis import get_analysis_status
import jieba
import re
from app.qa.blockchain_terms import BlockchainTerms
import jieba.analyse
from app.nlp.embedding_generator import EmbeddingGenerator
from app.analysis.document_comparison import DocumentComparison

logger = setup_logger(__name__)

class BlockchainQA:
    """区块链文献问答系统"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BlockchainQA, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        try:
            # 初始化问答模型
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if torch.cuda.is_available() else -1
            )
            self._initialized = True
            logger.info("QA system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QA system: {e}")
            raise

    def answer_question(self, question: str, file_id: str, session_id: str = None) -> dict:
        """回答问题"""
        try:
            # 获取文档分析结果
            analysis_data = get_analysis_status(file_id)
            if not analysis_data:
                return {"error": "Document not found"}
                
            # 确保分析已完成
            if analysis_data.get("status") != "completed":
                return {"error": "Document analysis not complete"}
                
            # 获取文档内容
            context = analysis_data.get("summary", "")
            if not context:
                return {"error": "No document content available"}

            # 使用问答模型生成答案
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_length=100
            )
            
            return {
                "answer": result["answer"],
                "confidence": float(result["score"]),
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {"error": str(e)}

    def _split_into_paragraphs(self, text: str) -> list:
        """Split text into paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs

    def _find_relevant_paragraphs(self, question: str, paragraphs: List[str]) -> List[Tuple[str, float]]:
        """Find paragraphs most relevant to the question"""
        try:
            # Generate question embedding
            question_embedding = self.embedding_generator.generate_embedding(question)
            
            # Generate embeddings for all paragraphs and calculate similarity
            similarities: List[Tuple[str, float]] = []
            for para in paragraphs:
                para_embedding = self.embedding_generator.generate_embedding(para)
                similarity = torch.cosine_similarity(
                    question_embedding.unsqueeze(0),
                    para_embedding.unsqueeze(0)
                ).item()
                similarities.append((para, similarity))
            
            # Sort by similarity
            return sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
            
        except Exception as e:
            logger.error(f"Error finding relevant paragraphs: {e}")
            return []

    def _extract_answer(self, question: str, context: str) -> str:
        """Extract answer from context"""
        try:
            # Prepare input
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get model output
            outputs = self.model(**inputs)
            
            # Decode answer
            answer_start = torch.argmax(outputs.start_logits).item()
            answer_end = torch.argmax(outputs.end_logits).item()
            
            # Safely access input IDs
            try:
                input_ids = inputs.get("input_ids")
                if input_ids is None or not isinstance(input_ids, torch.Tensor):
                    return "I cannot process the input properly."
                    
                # Ensure we can safely access the first element
                if input_ids.shape[0] == 0:
                    return "Input is empty."
                    
                # Convert to Python list
                input_ids_list = input_ids[0].tolist()
                
                # Extract answer tokens
                if 0 <= answer_start <= answer_end < len(input_ids_list):
                    answer_tokens = input_ids_list[answer_start:answer_end + 1]
                    
                    # Decode answer
                    answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                    
                    # Ensure return non-empty string
                    if answer and answer.strip():
                        return answer.strip()
                        
                return "I cannot find a specific answer in the document."
                
            except (IndexError, AttributeError, TypeError) as e:
                logger.error(f"Error accessing input IDs: {e}")
                return "I encountered an error while processing the input."
                
        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return "I encountered an error while trying to find the answer."

    def clear_history(self, session_id: str):
        """Clear conversation history"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]

    def _init_knowledge_base(self):
        """Initialize knowledge base"""
        self.knowledge_base = {
            # Basic concepts
            "blockchain_basic": {
                "keywords": ["blockchain", "what is", "how it works"],
                "answer": "Blockchain is a distributed ledger technology that ensures data cannot be tampered with. Its core features include decentralization, transparency, and security."
            },
            # Smart contracts
            "smart_contract": {
                "keywords": ["smart contract", "contract", "automatic execution"],
                "answer": "Smart contracts are programs that automatically execute on the blockchain, which can automatically execute preset rules and conditions without third-party intervention."
            },
            # Consensus mechanism
            "consensus": {
                "keywords": ["consensus", "consensus mechanism", "pow", "pos"],
                "answer": "The consensus mechanism is a method for all nodes in the blockchain network to reach agreement, including work-based proof (PoW) and equity-based proof (PoS)."
            },
            # Decentralization
            "decentralization": {
                "keywords": ["decentralization", "centralization", "distributed"],
                "answer": "Decentralization is the core feature of blockchain, meaning that there is no central authority in the network, all nodes participate equally, and the system's security and reliability are improved."
            },
            # Cryptography
            "cryptography": {
                "keywords": ["cryptography", "encryption", "hashing", "signing"],
                "answer": "Blockchain uses cryptographic technology to ensure data security, including hash functions, digital signatures, and asymmetric encryption, to ensure transaction security and tamper-proofing."
            },
            # Distributed storage
            "distributed_storage": {
                "keywords": ["distributed storage", "storage", "IPFS"],
                "answer": "Blockchain uses distributed storage, each node saves a complete ledger copy, which improves data availability and attack resistance."
            },
            # Web3
            "web3": {
                "keywords": ["web3", "web3.0", "decentralized application"],
                "answer": "Web3 is the next-generation internet based on blockchain, characterized by decentralized, user data sovereignty, and credit economy, supporting the development and operation of decentralized applications (DApp)."
            }
        }
        
        # Add knowledge graph relationships
        self.knowledge_graph = {
            "blockchain": {
                "components": ["block", "transaction", "consensus mechanism", "cryptography"],
                "features": ["decentralized", "tamper-proof", "transparent", "secure"],
                "applications": ["cryptocurrency", "supply chain", "digital identity", "smart contract"]
            },
            "smart contract": {
                "functions": ["automatic execution", "condition triggering", "state storage"],
                "platforms": ["Ethereum", "Solana", "Polkadot"],
                "languages": ["Solidity", "Rust", "WebAssembly"]
            },
            "consensus mechanism": {
                "types": ["PoW", "PoS", "DPoS", "PBFT"],
                "goals": ["consistency", "fault tolerance", "security"],
                "applications": ["Bitcoin", "Ethereum", "Cardano"]
            }
        }
    
    def _understand_with_context(self, question: str, history: List[Dict]) -> str:
        """Understand question based on context"""
        try:
            if not history:
                return question
                
            # Handle pronouns
            pronouns = ["it", "this", "that", "these", "those"]
            if any(p in question.lower() for p in pronouns):
                # Get topic words from previous conversation
                last_keywords = jieba.analyse.extract_tags(
                    history[-1]["question"] + " " + history[-1]["answer"],
                    topK=3
                )
                # Replace pronouns with specific topics
                for pronoun in pronouns:
                    if pronoun in question.lower():
                        keywords_str = " ".join([str(kw) for kw in last_keywords])
                        question = question.replace(pronoun, keywords_str)
                        break
            
            return question
            
        except Exception as e:
            logger.error(f"Failed to understand question context: {e}")
            return question
    
    def _answer_from_document(self, question: str, file_id: str) -> Optional[Dict]:
        """Find answer from document"""
        try:
            # Get document analysis result
            analysis = get_analysis_status(file_id)
            if not analysis or analysis.get("status") != "completed":
                return None
            
            # Extract document content
            paragraphs = analysis.get("paragraphs", [])
            keywords = analysis.get("keywords", [])
            
            if not paragraphs:
                return None
            
            # Split question into words
            question_words = jieba.lcut(question)
            
            # Calculate relevance for each paragraph
            paragraph_scores = []
            for i, para in enumerate(paragraphs):
                para_words = jieba.lcut(para)
                score = self._calculate_relevance(question_words, para_words)
                paragraph_scores.append((score, i, para))
            
            # Sort by relevance
            paragraph_scores.sort(reverse=True)
            
            # If the score of the most relevant paragraph exceeds the threshold
            if paragraph_scores[0][0] > 0.2:
                top_paragraphs = paragraph_scores[:2]  # Take the top two most relevant paragraphs
                answer_text = "\n\n".join([p[2] for p in top_paragraphs])
                
                return {
                    "answer": answer_text,
                    "confidence": top_paragraphs[0][0],
                    "source": "document",
                    "keywords": keywords,
                    "paragraph_index": [p[1] for p in top_paragraphs]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find answer from document: {e}")
            return None
    
    def _answer_from_knowledge_base(self, question: str) -> Dict:
        """Find answer from knowledge base"""
        try:
            words = jieba.lcut(question)
            best_match = None
            max_score = 0
            
            for topic, content in self.knowledge_base.items():
                score = self._calculate_relevance(words, content["keywords"])
                if score > max_score:
                    max_score = score
                    best_match = content["answer"]
            
            if max_score > 0.3:
                return {
                    "answer": best_match,
                    "confidence": max_score,
                    "source": "knowledge_base"
                }
            
            return {
                "answer": "I apologize, but I cannot answer this question at the moment. Please try rephrasing your question or check our documentation.",
                "confidence": 0,
                "source": "default"
            }
            
        except Exception as e:
            logger.error(f"Failed to find answer from knowledge base: {e}")
            return {
                "answer": "An error occurred while processing your question. Please try again later.",
                "confidence": 0,
                "source": "error"
            }
    
    def _calculate_relevance(self, words1: List[str], words2: List[str]) -> float:
        """Calculate relevance between two groups of words"""
        try:
            # Create word frequency dictionary
            freq1: Dict[str, int] = {}
            freq2: Dict[str, int] = {}
            
            for word in words1:
                if len(word) > 1:  # Ignore single-character words
                    freq1[word] = freq1.get(word, 0) + 1
            
            for word in words2:
                if len(word) > 1:
                    freq2[word] = freq2.get(word, 0) + 1
            
            # Calculate the number of common words
            common_words = set(freq1.keys()) & set(freq2.keys())
            if not common_words:
                return 0.0
            
            # Calculate relevance score
            score = sum(min(freq1[word], freq2[word]) for word in common_words)
            # Normalize
            score = score / (len(words1) + len(words2))
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance: {e}")
            return 0.0

    def compare_documents(self, doc1_id: str, doc2_id: str) -> Dict:
        """Compare two documents"""
        try:
            # Get document contents
            doc1_analysis = get_analysis_status(doc1_id)
            doc2_analysis = get_analysis_status(doc2_id)
            
            if not doc1_analysis or not doc2_analysis:
                raise ValueError("One or both documents not found")
            
            doc1_text = doc1_analysis.get("full_text", "")
            doc2_text = doc2_analysis.get("full_text", "")
            
            if not doc1_text or not doc2_text:
                raise ValueError("One or both documents have no content")
            
            # Perform comparison
            comparison_result = self.document_comparison.compare_documents(
                doc1_text, doc2_text
            )
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            return {
                "error": str(e),
                "status": "error"
            } 