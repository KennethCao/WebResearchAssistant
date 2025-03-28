import pandas as pd
from torch.utils.data import Dataset
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class BlockchainDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def prepare_training_data(data_path):
    """
    准备训练数据
    """
    try:
        # 读取数据
        df = pd.read_csv(data_path)
        
        # 数据预处理
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        logger.info(f"Prepared {len(texts)} training samples")
        return texts, labels
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise 