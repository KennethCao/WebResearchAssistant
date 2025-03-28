from typing import Optional, Dict, Any
from app.models.paper import Paper
from app.nlp.text_analyzer import TextAnalyzer
from app.utils.logger import setup_logger
import numpy as np
from sentence_transformers import util

logger = setup_logger(__name__)

class QAService:
    def __init__(self):
        self.analyzer = TextAnalyzer()
        
    def get_answer(self, question: str, paper_id: Optional[int] = None) -> str:
        """Get answer for a question"""
        try:
            # Get context from paper if paper_id provided
            context = ""
            if paper_id:
                paper = Paper.query.get(paper_id)
                if paper:
                    context = paper.content
            
            # Generate question embedding
            question_embedding = self.analyzer.embedding_model.encode(question)
            
            # If paper_id provided, use paper-specific context
            if context:
                # Split context into chunks
                chunks = self._split_text(context)
                chunk_embeddings = self.analyzer.embedding_model.encode(chunks)
                
                # Find most relevant chunks
                scores = util.dot_score(question_embedding, chunk_embeddings)[0]
                top_k = min(3, len(chunks))
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                
                # Combine relevant chunks
                relevant_context = " ".join([chunks[i] for i in top_indices])
                
                # Generate answer using relevant context
                answer = self._generate_answer(question, relevant_context)
                
            else:
                # Use general blockchain knowledge
                answer = self._generate_general_answer(question)
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to get answer: {e}")
            return "Sorry, I couldn't process your question. Please try again."
    
    def _split_text(self, text: str, chunk_size: int = 512) -> list:
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
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using context"""
        try:
            # Combine question and context
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Get answer from model
            answer = self.analyzer.qa_model(prompt)[0]['generated_text']
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "Sorry, I couldn't generate an answer. Please try again."
    
    def _generate_general_answer(self, question: str) -> str:
        """Generate answer using general knowledge"""
        try:
            # Add blockchain-specific prompt
            prompt = f"As a blockchain expert, please answer this question: {question}"
            
            # Get answer from model
            answer = self.analyzer.qa_model(prompt)[0]['generated_text']
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate general answer: {e}")
            return "Sorry, I couldn't generate an answer. Please try again."

    def add_to_knowledge_base(self, text: str) -> bool:
        """添加文本到知识库"""
        try:
            self.analyzer.embedding_model.add_documents([text])
            return True
        except Exception as e:
            logger.error(f"添加到知识库失败: {e}")
            return False 