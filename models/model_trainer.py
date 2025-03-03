"""
Model training module for financial translation system.
Handles training, fine-tuning, and evaluation of translation models.
"""

import os
import logging
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import yaml
from tqdm import tqdm
import wandb
from transformers import (
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer for financial translation models."""
    
    def __init__(self, 
                 model,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config_path: str = "../config.yaml",
                 output_dir: str = "models/checkpoints"):
        """
        Initialize the model trainer.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']['fine_tuning']
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.learning_rate = self.config.get('learning_rate', 2e-5)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.epochs = self.config.get('epochs', 10)
        self.warmup_steps = self.config.get('warmup_steps', 500)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Enable gradient checkpointing if available to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        self._setup_optimizer_scheduler()
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project="financial-translation-poc", config=self.config)
    
    def _setup_optimizer_scheduler(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        """
        logger.info(f"Starting training on {self.device}")
        
        history = {'train_loss': [], 'val_loss': []}
        scaler = torch.cuda.amp.GradScaler()  # Initialize AMP scaler
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        decoder_attention_mask=batch.get('decoder_attention_mask')
                    )
                scaler.scale(outputs.loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                
                train_loss += outputs.loss.item()
                train_steps += 1
                self.global_step += 1
                progress_bar.set_postfix({'loss': outputs.loss.item()})
                
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train_loss': outputs.loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
            
            avg_train_loss = train_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            
            val_loss = self.evaluate()
            history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': avg_train_loss,
                    'val_loss_epoch': val_loss
                })
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"epoch_{epoch+1}_val_loss_{val_loss:.4f}")
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        logger.info("Training completed")
        return history
    
    def evaluate(self) -> float:
        """Evaluate the model on validation data."""
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    decoder_attention_mask=batch.get('decoder_attention_mask')
                )
                val_loss += outputs.loss.item()
                val_steps += 1
        avg_val_loss = val_loss / val_steps
        return avg_val_loss
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load model checkpoint."""
        model_class = type(self.model)
        self.model = model_class.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.optimizer.load_state_dict(training_state['optimizer'])
            self.scheduler.load_state_dict(training_state['scheduler'])
            self.global_step = training_state['global_step']
            self.best_val_loss = training_state['best_val_loss']
            logger.info(f"Resumed training from step {self.global_step}")
        else:
            logger.warning(f"No training state found at {training_state_path}")
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from transformers import MBartForConditionalGeneration
    from torch.utils.data import DataLoader

    # Dummy dataset for demonstration
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (50,)),
                'attention_mask': torch.ones(50),
                'labels': torch.randint(0, 1000, (50,)),
                'decoder_attention_mask': torch.ones(50)
            }
    
    train_dataloader = DataLoader(DummyDataset(), batch_size=4)
    val_dataloader = DataLoader(DummyDataset(), batch_size=4)
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    trainer = ModelTrainer(model, train_dataloader, val_dataloader)
    trainer.epochs = 1
    history = trainer.train()
    print(f"Training history: {history}")
    trainer.save_checkpoint("final_model")
"""
Model training module for financial translation system.
Handles training, fine-tuning, and evaluation of translation models.
"""

import os
import logging
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List
import yaml
from tqdm import tqdm
import wandb
from transformers import (
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Trainer for financial translation models."""
    
    def __init__(self, 
                 model,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 config_path: str = "../config.yaml",
                 output_dir: str = "models/checkpoints"):
        """
        Initialize the model trainer.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']['fine_tuning']
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.learning_rate = self.config.get('learning_rate', 2e-5)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.epochs = self.config.get('epochs', 10)
        self.warmup_steps = self.config.get('warmup_steps', 500)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Enable gradient checkpointing if available to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        self._setup_optimizer_scheduler()
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project="financial-translation-poc", config=self.config)
    
    def _setup_optimizer_scheduler(self) -> None:
        """Set up optimizer and learning rate scheduler."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        """
        logger.info(f"Starting training on {self.device}")
        
        history = {'train_loss': [], 'val_loss': []}
        scaler = torch.cuda.amp.GradScaler()  # Initialize AMP scaler
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1}/{self.epochs}")
            
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        decoder_attention_mask=batch.get('decoder_attention_mask')
                    )
                scaler.scale(outputs.loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.scheduler.step()
                
                train_loss += outputs.loss.item()
                train_steps += 1
                self.global_step += 1
                progress_bar.set_postfix({'loss': outputs.loss.item()})
                
                if self.use_wandb and self.global_step % 10 == 0:
                    wandb.log({
                        'train_loss': outputs.loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
            
            avg_train_loss = train_loss / train_steps
            history['train_loss'].append(avg_train_loss)
            
            val_loss = self.evaluate()
            history['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': avg_train_loss,
                    'val_loss_epoch': val_loss
                })
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"epoch_{epoch+1}_val_loss_{val_loss:.4f}")
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        logger.info("Training completed")
        return history
    
    def evaluate(self) -> float:
        """Evaluate the model on validation data."""
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    decoder_attention_mask=batch.get('decoder_attention_mask')
                )
                val_loss += outputs.loss.item()
                val_steps += 1
        avg_val_loss = val_loss / val_steps
        return avg_val_loss
    
    def save_checkpoint(self, checkpoint_name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }, os.path.join(checkpoint_dir, "training_state.pt"))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load model checkpoint."""
        model_class = type(self.model)
        self.model = model_class.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=self.device)
            self.optimizer.load_state_dict(training_state['optimizer'])
            self.scheduler.load_state_dict(training_state['scheduler'])
            self.global_step = training_state['global_step']
            self.best_val_loss = training_state['best_val_loss']
            logger.info(f"Resumed training from step {self.global_step}")
        else:
            logger.warning(f"No training state found at {training_state_path}")
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from transformers import MBartForConditionalGeneration
    from torch.utils.data import DataLoader

    # Dummy dataset for demonstration
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 1000, (50,)),
                'attention_mask': torch.ones(50),
                'labels': torch.randint(0, 1000, (50,)),
                'decoder_attention_mask': torch.ones(50)
            }
    
    train_dataloader = DataLoader(DummyDataset(), batch_size=4)
    val_dataloader = DataLoader(DummyDataset(), batch_size=4)
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    trainer = ModelTrainer(model, train_dataloader, val_dataloader)
    trainer.epochs = 1
    history = trainer.train()
    print(f"Training history: {history}")
    trainer.save_checkpoint("final_model")
