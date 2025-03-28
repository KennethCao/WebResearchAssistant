from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class BlockchainDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer_name: str, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # 区块链领域的标签
        self.labels = [
            'consensus',           # 共识机制
            'smart_contract',      # 智能合约
            'privacy',            # 隐私保护
            'scalability',        # 可扩展性
            'security',           # 安全性
            'interoperability',   # 互操作性
            'governance',         # 治理机制
            'tokenomics',         # 代币经济
            'application'         # 应用场景
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)
        
        # 加载数据
        self.examples = self._load_data()
        
    def _load_data(self):
        """加载区块链论文数据"""
        examples = []
        
        # 遍历所有JSON文件
        for json_file in self.data_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    
                # 提取文本和标签
                text = f"{paper['title']} {paper['abstract']}"
                labels = paper.get('topics', [])
                
                # 将标签转换为one-hot编码
                label_ids = torch.zeros(self.num_labels)
                for label in labels:
                    if label in self.label2id:
                        label_ids[self.label2id[label]] = 1
                
                examples.append({
                    'text': text,
                    'labels': label_ids
                })
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                continue
                
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            example['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': example['labels']
        }

    def split_train_eval(self, eval_ratio=0.1):
        """Split dataset into train and eval sets"""
        eval_size = int(len(self) * eval_ratio)
        train_size = len(self) - eval_size
        
        return torch.utils.data.random_split(
            self, 
            [train_size, eval_size]
        ) 