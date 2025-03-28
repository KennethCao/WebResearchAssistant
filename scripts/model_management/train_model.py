import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import gc
import time

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.config.environment import BLOCKCHAIN_MODEL_PATH
from app.utils.logger import setup_logger

# 设置日志
logger = setup_logger(__name__)

class BlockchainDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        try:
            self.encodings = tokenizer(
                texts, 
                truncation=True, 
                padding=True, 
                max_length=max_length,
                return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise

    def __getitem__(self, idx):
        try:
            item = {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.labels[idx]
            }
            return item
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            raise

    def __len__(self):
        return len(self.labels)

def load_data_in_chunks(data_path, chunk_size=1000):
    """分批加载数据"""
    try:
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            texts = chunk['text'].tolist()
            labels = chunk['label'].tolist()
            yield texts, labels
    except Exception as e:
        logger.error(f"Error loading data chunk: {e}")
        raise

def train_model(data_path, epochs=3, batch_size=4, learning_rate=2e-5, checkpoint_dir=None):
    try:
        # Set device
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2
        )
        model = model.to('cpu')
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Create checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(BLOCKCHAIN_MODEL_PATH, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training loop
        logger.info("Starting training...")
        model.train()
        best_loss = float('inf')
        global_step = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            batch_count = 0
            
            # Load data in batches
            for chunk_idx, (texts, labels) in enumerate(load_data_in_chunks(data_path)):
                # 创建数据集
                try:
                    dataset = BlockchainDataset(texts, labels, tokenizer)
                    train_loader = DataLoader(
                        dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=0
                    )
                    
                    # Release memory
                    del texts, labels
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    progress_bar = tqdm(
                        train_loader,
                        desc=f'Epoch {epoch + 1}/{epochs} Chunk {chunk_idx+1}',
                        ncols=100
                    )
                    
                    for batch_idx, batch in enumerate(progress_bar):
                        try:
                            # Move data to device
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)
                            
                            # Clear gradients
                            optimizer.zero_grad()
                            
                            # Forward pass
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            
                            loss = outputs.loss
                            total_loss += loss.item()
                            batch_count += 1
                            
                            # Backward propagation
                            loss.backward()
                            optimizer.step()
                            
                            # Release memory
                            del outputs, loss
                            gc.collect()
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                            
                            # Update progress bar
                            progress_bar.set_postfix({
                                'loss': f'{loss.item():.4f}',
                                'avg_loss': f'{total_loss/batch_count:.4f}',
                                'memory': f'{torch.cuda.memory_allocated()/1024**2:.1f}MB' if torch.cuda.is_available() else 'CPU'
                            })
                            
                            global_step += 1
                            
                            # Save checkpoint every 50 batches
                            if global_step % 50 == 0:
                                checkpoint_path = os.path.join(
                                    checkpoint_dir, 
                                    f'checkpoint_epoch_{epoch+1}_step_{global_step}.pt'
                                )
                                torch.save({
                                    'epoch': epoch,
                                    'global_step': global_step,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss.item(),
                                }, checkpoint_path)
                                logger.info(f"Saved checkpoint to {checkpoint_path}")
                                
                        except Exception as e:
                            logger.error(f"Error in batch {batch_idx}: {e}")
                            continue
                            
                    # Release data loader
                    del train_loader, dataset
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx}: {e}")
                    continue

            # Calculate average loss
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s")
            logger.info(f"Average loss: {avg_loss:.4f}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                output_dir = os.path.join(BLOCKCHAIN_MODEL_PATH, 'fine_tuned')
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Saved best model with loss {best_loss:.4f}")

        logger.info("Training completed successfully")
        return True

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    try:
        data_path = os.path.join(project_root, "data", "blockchain_papers.csv")
        checkpoint_dir = os.path.join(BLOCKCHAIN_MODEL_PATH, 'checkpoints')
        
        # Use smaller batch size
        success = train_model(
            data_path=data_path,
            epochs=3,
            batch_size=4,  # Reduced batch size
            learning_rate=2e-5,
            checkpoint_dir=checkpoint_dir
        )
        
        if success:
            print("Training completed successfully!")
        else:
            print("Training failed. Check the logs for details.")
            
    except Exception as e:
        print(f"Error running training script: {e}")