from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import torch
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model(train_data, model_save_path, epochs=3, batch_size=16, learning_rate=2e-5):
    """
    训练模型
    
    Args:
        train_data: 训练数据集
        model_save_path: 模型保存路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    try:
        logger.info("Loading pre-trained model...")
        # 加载预训练模型和分词器
        model = AutoModelForSequenceClassification.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 准备训练数据
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 设置优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 训练循环
        logger.info("Starting training...")
        model.train()
        total_steps = len(train_loader) * epochs
        
        for epoch in range(epochs):
            epoch_loss = 0
            for step, batch in enumerate(train_loader):
                # 将数据移到设备
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs.loss
                epoch_loss += loss.item()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 打印进度
                if step % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}')
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # 保存模型
        logger.info("Saving model...")
        model.save_pretrained(model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        
        return {
            'epochs_completed': epochs,
            'final_loss': avg_epoch_loss,
            'save_path': model_save_path
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise 