from typing import Dict, Any, List
from app.services.qa_service import QAService
from app.nlp.blockchain_analyzer import BlockchainAnalyzer
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class BlockchainQAService(QAService):
    def __init__(self):
        super().__init__()
        self.blockchain_analyzer = BlockchainAnalyzer()
        
        # 预定义问题模板
        self.question_templates = {
            'consensus': 'What consensus mechanism is used in this paper?',
            'innovation': 'What are the main innovations of this paper?',
            'challenge': 'What are the main challenges discussed in this paper?',
            'performance': 'What are the performance metrics mentioned in this paper?',
            'application': 'What are the potential applications discussed in this paper?'
        }
        
    def get_blockchain_answer(self, question: str, paper_id: int = None) -> Dict[str, Any]:
        """处理区块链相关问题"""
        try:
            # 识别问题类型
            question_type = self._classify_question(question)
            
            # 获取基础答案
            base_answer = self.get_answer(question, paper_id)
            
            # 根据问题类型增强答案
            enhanced_answer = self._enhance_answer(
                base_answer,
                question_type,
                paper_id
            )
            
            return {
                'answer': enhanced_answer,
                'type': question_type
            }
            
        except Exception as e:
            logger.error(f"Failed to get blockchain answer: {e}")
            return {
                'answer': "Sorry, I couldn't process your question about blockchain.",
                'type': 'unknown'
            }
            
    def _classify_question(self, question: str) -> str:
        """分类区块链相关问题"""
        try:
            # 预定义问题类型
            question_types = [
                'consensus',
                'smart_contract',
                'privacy',
                'scalability',
                'security',
                'interoperability',
                'governance',
                'tokenomics',
                'application'
            ]
            
            # 使用零样本分类
            result = self.blockchain_analyzer.zero_shot(
                question,
                candidate_labels=question_types
            )
            
            return result['labels'][0]
            
        except Exception as e:
            logger.error(f"Failed to classify question: {e}")
            return 'general'
            
    def _enhance_answer(self, base_answer: str, question_type: str, paper_id: int) -> str:
        """增强答案内容"""
        try:
            if not paper_id:
                return base_answer
                
            # 获取论文分析结果
            paper = self.paper_service.get_paper(paper_id)
            if not paper or not paper.analysis_results:
                return base_answer
                
            analysis = paper.analysis_results
            
            # 根据问题类型添加相关信息
            if question_type == 'consensus':
                if 'technical_details' in analysis and 'consensus_mechanism' in analysis['technical_details']:
                    base_answer += f"\n\nThe paper uses {analysis['technical_details']['consensus_mechanism']} as its consensus mechanism."
                    
            elif question_type == 'innovation':
                if 'innovations' in analysis and analysis['innovations']:
                    base_answer += "\n\nKey innovations include:\n" + "\n".join(f"- {i}" for i in analysis['innovations'])
                    
            elif question_type == 'challenge':
                if 'challenges' in analysis and analysis['challenges']:
                    base_answer += "\n\nMain challenges include:\n" + "\n".join(f"- {c}" for c in analysis['challenges'])
                    
            elif question_type == 'performance':
                if 'technical_details' in analysis and 'performance' in analysis['technical_details']:
                    metrics = analysis['technical_details']['performance']
                    base_answer += "\n\nPerformance metrics:\n" + "\n".join(f"- {k}: {v}" for k, v in metrics.items())
                    
            return base_answer
            
        except Exception as e:
            logger.error(f"Failed to enhance answer: {e}")
            return base_answer 