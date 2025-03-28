from typing import Dict, List, Any
import torch
from transformers import pipeline
from app.utils.logger import setup_logger
from app.nlp.text_analyzer import TextAnalyzer

logger = setup_logger(__name__)

class BlockchainAnalyzer(TextAnalyzer):
    def __init__(self):
        super().__init__()
        
        # 区块链特定的分类标签
        self.blockchain_aspects = [
            'consensus mechanism',
            'smart contract',
            'privacy protection',
            'scalability solution',
            'security mechanism',
            'cross-chain technology',
            'governance model',
            'token economics',
            'use case'
        ]
        
        # 加载微调后的模型
        try:
            self.blockchain_classifier = pipeline(
                'text-classification',
                model='models/finetuned',
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load finetuned model: {e}")
            self.blockchain_classifier = None

    def analyze_blockchain_paper(self, text_data: Dict[str, str]) -> Dict[str, Any]:
        """分析区块链论文"""
        try:
            # 基础分析
            results = super().analyze_paper(text_data)
            
            # 区块链特定分析
            results.update({
                'blockchain_aspects': self._analyze_blockchain_aspects(text_data),
                'technical_details': self._extract_technical_details(text_data),
                'innovations': self._extract_blockchain_innovations(text_data),
                'challenges': self._extract_challenges(text_data)
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze blockchain paper: {e}")
            raise

    def _analyze_blockchain_aspects(self, text_data: Dict[str, str]) -> List[Dict[str, float]]:
        """分析区块链相关方面"""
        try:
            text = text_data.get('abstract', '') + ' ' + text_data.get('full_text', '')
            
            results = self.zero_shot(
                text,
                candidate_labels=self.blockchain_aspects,
                multi_label=True
            )
            
            # 过滤低置信度的结果
            return [
                {'aspect': label, 'score': score}
                for label, score in zip(results['labels'], results['scores'])
                if score > 0.3
            ]
            
        except Exception as e:
            logger.error(f"Failed to analyze blockchain aspects: {e}")
            return []

    def _extract_technical_details(self, text_data: Dict[str, str]) -> Dict[str, str]:
        """提取技术细节"""
        try:
            details = {}
            
            # 查找共识机制
            consensus = self._find_consensus_mechanism(text_data)
            if consensus:
                details['consensus_mechanism'] = consensus
            
            # 查找智能合约平台
            contract_platform = self._find_contract_platform(text_data)
            if contract_platform:
                details['contract_platform'] = contract_platform
            
            # 查找性能指标
            performance = self._extract_performance_metrics(text_data)
            if performance:
                details['performance'] = performance
            
            return details
            
        except Exception as e:
            logger.error(f"Failed to extract technical details: {e}")
            return {}

    def _extract_blockchain_innovations(self, text_data: Dict[str, str]) -> List[str]:
        """提取区块链创新点"""
        try:
            innovations = []
            
            # 在引言和方法部分查找创新点
            sections = text_data.get('sections', [])
            relevant_sections = [
                s for s in sections if any(x in s['title'].lower() 
                    for x in ['introduction', 'proposed', 'novel', 'contribution'])
            ]
            
            for section in relevant_sections:
                sentences = section['content'].split('.')
                for sentence in sentences:
                    if len(sentence.strip()) < 10:
                        continue
                        
                    # 使用零样本分类判断是否是创新点
                    result = self.zero_shot(
                        sentence,
                        candidate_labels=['innovation', 'existing work'],
                        hypothesis_template='This describes {}'
                    )
                    
                    if result['labels'][0] == 'innovation' and result['scores'][0] > 0.8:
                        innovations.append(sentence.strip())
            
            return innovations[:5]  # 返回最多5个创新点
            
        except Exception as e:
            logger.error(f"Failed to extract blockchain innovations: {e}")
            return []

    def _extract_challenges(self, text_data: Dict[str, str]) -> List[str]:
        """提取研究挑战"""
        try:
            challenges = []
            
            # 在讨论和结论部分查找挑战
            sections = text_data.get('sections', [])
            relevant_sections = [
                s for s in sections if any(x in s['title'].lower() 
                    for x in ['discussion', 'conclusion', 'future', 'challenge'])
            ]
            
            for section in relevant_sections:
                sentences = section['content'].split('.')
                for sentence in sentences:
                    if len(sentence.strip()) < 10:
                        continue
                        
                    # 使用零样本分类判断是否描述了挑战
                    result = self.zero_shot(
                        sentence,
                        candidate_labels=['challenge', 'achievement'],
                        hypothesis_template='This describes a {}'
                    )
                    
                    if result['labels'][0] == 'challenge' and result['scores'][0] > 0.7:
                        challenges.append(sentence.strip())
            
            return challenges[:3]  # 返回最多3个主要挑战
            
        except Exception as e:
            logger.error(f"Failed to extract challenges: {e}")
            return []

    def _find_consensus_mechanism(self, text_data: Dict[str, str]) -> str:
        """识别共识机制"""
        consensus_mechanisms = [
            'Proof of Work',
            'Proof of Stake',
            'Delegated Proof of Stake',
            'Practical Byzantine Fault Tolerance',
            'Proof of Authority',
            'Proof of Space',
            'Proof of Capacity'
        ]
        
        text = text_data.get('full_text', '')
        
        result = self.zero_shot(
            text,
            candidate_labels=consensus_mechanisms,
            multi_label=True
        )
        
        # 返回置信度最高的共识机制
        if result['scores'][0] > 0.5:
            return result['labels'][0]
        return None

    def _find_contract_platform(self, text_data: Dict[str, str]) -> str:
        """识别智能合约平台"""
        platforms = [
            'Ethereum',
            'Binance Smart Chain',
            'Solana',
            'Polkadot',
            'Cardano',
            'Hyperledger Fabric'
        ]
        
        text = text_data.get('full_text', '')
        
        result = self.zero_shot(
            text,
            candidate_labels=platforms,
            multi_label=True
        )
        
        if result['scores'][0] > 0.5:
            return result['labels'][0]
        return None

    def _extract_performance_metrics(self, text_data: Dict[str, str]) -> Dict[str, str]:
        """提取性能指标"""
        metrics = {}
        
        # 在结果部分查找性能指标
        sections = text_data.get('sections', [])
        results_sections = [
            s for s in sections if any(x in s['title'].lower() 
                for x in ['result', 'evaluation', 'experiment', 'performance'])
        ]
        
        if not results_sections:
            return metrics
            
        text = results_sections[0]['content']
        
        # 使用规则提取常见指标
        import re
        
        # TPS (Transactions Per Second)
        tps_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:TPS|transactions per second)', text, re.I)
        if tps_match:
            metrics['tps'] = tps_match.group(1)
            
        # Latency
        latency_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?|seconds?)\s*latency', text, re.I)
        if latency_match:
            metrics['latency'] = latency_match.group(1)
            
        # Block time
        block_time_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:s|seconds?)\s*block[- ]time', text, re.I)
        if block_time_match:
            metrics['block_time'] = block_time_match.group(1)
            
        return metrics 