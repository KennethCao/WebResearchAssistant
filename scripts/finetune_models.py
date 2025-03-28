import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from app.config.environment import MODEL_CACHE_DIR
from app.utils.logger import setup_logger
from app.data.dataset import BlockchainDataset

logger = setup_logger(__name__)

def finetune_models():
    """Finetune models on blockchain dataset"""
    try:
        # Load custom dataset
        dataset = BlockchainDataset(
            data_dir='data/blockchain',
            tokenizer_name='bert-base-uncased'
        )
        
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            cache_dir=MODEL_CACHE_DIR,
            num_labels=dataset.num_labels
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='models/finetuned',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='logs/finetune',
            logging_steps=100,
            evaluation_strategy="epoch"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.eval_dataset
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = Path('models/finetuned')
        model_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_path))
        
        logger.info("Model finetuning completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to finetune models: {e}")
        raise

if __name__ == '__main__':
    finetune_models() 