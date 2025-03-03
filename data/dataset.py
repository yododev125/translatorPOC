"""
Dataset module for financial translation system.
Handles data loading, batching, and tokenization for model training.
"""

import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import yaml
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class FinancialTranslationDataset(Dataset):
    """Dataset for financial document translation."""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer_name: str,
                 max_length: int = 128,
                 src_lang: str = 'en',
                 tgt_lang: str = 'ar',
                 split: str = 'train'):
        """
        Initialize the dataset.
        """
        self.data_path = data_path
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.split = split
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = self._load_data()
        logger.info(f"Loaded {len(self.data)} examples for {split} split")
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        """
        file_path = os.path.join(self.data_path, f"{self.split}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = pd.read_csv(file_path)
        src_col = 'english' if self.src_lang == 'en' else 'arabic'
        tgt_col = 'arabic' if self.tgt_lang == 'ar' else 'english'
        if src_col not in df.columns or tgt_col not in df.columns:
            raise ValueError(f"Data must contain '{src_col}' and '{tgt_col}' columns")
        df = df.dropna(subset=[src_col, tgt_col])
        return df
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        src_col = 'english' if self.src_lang == 'en' else 'arabic'
        tgt_col = 'arabic' if self.tgt_lang == 'ar' else 'english'
        src_text = row[src_col]
        tgt_text = row[tgt_col]
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_encoding = {k: v.squeeze(0) for k, v in src_encoding.items()}
        tgt_encoding = {k: v.squeeze(0) for k, v in tgt_encoding.items()}
        inputs = {
            'input_ids': src_encoding['input_ids'],
            'attention_mask': src_encoding['attention_mask'],
            'labels': tgt_encoding['input_ids'],
            'decoder_attention_mask': tgt_encoding['attention_mask']
        }
        if 'document_id' in row:
            inputs['document_id'] = row['document_id']
        if 'source' in row:
            inputs['source'] = row['source']
        return inputs

def create_dataloaders(config_path: str = "../config.yaml") -> Dict[str, DataLoader]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_config = config['data']['preprocessing']
    model_config = config['model']
    tokenizer_name = model_config.get('base_model', 'facebook/mbart-large-50')
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = FinancialTranslationDataset(
            data_path='data/processed',
            tokenizer_name=tokenizer_name,
            max_length=data_config['max_sequence_length'],
            split=split
        )
    dataloaders = {}
    batch_size = model_config['fine_tuning']['batch_size']
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    dataloaders['test'] = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    logger.info(f"Created dataloaders with batch size {batch_size}")
    return dataloaders

# --- New Enhancement: Dataset Checksum ---

import hashlib

def hash_dataset(df: pd.DataFrame) -> str:
    """
    Generate a SHA256 checksum of the dataset.
    
    Args:
        df: Pandas DataFrame of the dataset.
        
    Returns:
        A hexadecimal checksum string.
    """
    hashed_series = pd.util.hash_pandas_object(df, index=True)
    checksum = hashlib.sha256(hashed_series.values.tobytes()).hexdigest()
    return checksum
