"""
Transformer-based Neural Machine Translation model.
Implements advanced translation models using transformer architectures.
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
import yaml
from transformers import (
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

logger = logging.getLogger(__name__)

class TransformerNMT:
    """Neural Machine Translation model using transformer architecture."""
    
    def __init__(self, 
                 model_name: str = "facebook/mbart-large-50",
                 tokenizer_name: Optional[str] = None,
                 config_path: str = "../config.yaml"):
        """
        Initialize the transformer NMT model.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        
        if "mbart" in model_name.lower():
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self.model_type = "mbart"
        elif "mt5" in model_name.lower():
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)
            self.model_type = "mt5"
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_type = "seq2seq"
        
        self.model.to(self.device)
        logger.info(f"Loaded {self.model_type} model {model_name} on {self.device}")
        
        if self.model_type == "mbart":
            self.en_code = "en_XX"
            self.ar_code = "ar_AR"
    
    def translate(self, 
                  text: str, 
                  source_lang: str = 'en', 
                  target_lang: str = 'ar',
                  max_length: int = 128,
                  num_beams: int = 4) -> str:
        """
        Translate text from source language to target language with glossary constraints.
        """
        if not text:
            return ""
        
        if self.model_type == "mbart":
            src_lang_code = self.en_code if source_lang == 'en' else self.ar_code
            tgt_lang_code = self.ar_code if target_lang == 'ar' else self.en_code
            self.tokenizer.src_lang = src_lang_code
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_code]
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            forced_bos_token_id = None
        
        # Obtain glossary constraints using the unified TerminologyService
        try:
            from models.terminology_service import TerminologyService
            term_service = TerminologyService()
            constraints = term_service.get_constraints(text, self.tokenizer, source_lang)
        except Exception as e:
            logger.error(f"Error obtaining glossary constraints: {str(e)}")
            constraints = []
        
        try:
            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "early_stopping": True,
                    "forced_bos_token_id": forced_bos_token_id
                }
                if constraints:
                    generate_kwargs["constraints"] = constraints
                outputs = self.model.generate(**generate_kwargs)
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            return ""
    
    def batch_translate(self, 
                        texts: List[str], 
                        source_lang: str = 'en', 
                        target_lang: str = 'ar',
                        max_length: int = 128,
                        num_beams: int = 4,
                        batch_size: int = 8) -> List[str]:
        """
        Translate a batch of texts.
        """
        if not texts:
            return []
        all_translations = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if self.model_type == "mbart":
                src_lang_code = self.en_code if source_lang == 'en' else self.ar_code
                tgt_lang_code = self.ar_code if target_lang == 'ar' else self.en_code
                self.tokenizer.src_lang = src_lang_code
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang_code]
            else:
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                forced_bos_token_id = None
            
            # For each batch, try to get constraints for each text (if available)
            batch_constraints = []
            try:
                from models.terminology_service import TerminologyService
                term_service = TerminologyService()
                for text in batch_texts:
                    constraints = term_service.get_constraints(text, self.tokenizer, source_lang)
                    batch_constraints.append(constraints)
            except Exception as e:
                logger.error(f"Error obtaining constraints for batch: {str(e)}")
                batch_constraints = [None] * len(batch_texts)
            
            try:
                with torch.no_grad():
                    # Note: Currently, transformers' generate API applies same constraints for entire batch.
                    # For simplicity, if constraints exist for the first sample, apply them to the batch.
                    constraints_to_apply = batch_constraints[0] if batch_constraints and batch_constraints[0] else None
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=num_beams,
                        early_stopping=True,
                        forced_bos_token_id=forced_bos_token_id,
                        constraints=constraints_to_apply
                    )
                batch_translations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                all_translations.extend(batch_translations)
            except Exception as e:
                logger.error(f"Error during batch translation: {str(e)}")
                all_translations.extend([""] * len(batch_texts))
        return all_translations
    
    def save_model(self, output_dir: str) -> None:
        """
        Save the model and tokenizer.
        """
        os.makedirs(output_dir, exist_ok=True)
        try:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, model_dir: str) -> None:
        """
        Load the model and tokenizer from a directory.
        """
        try:
            if self.model_type == "mbart":
                from transformers import MBartForConditionalGeneration
                self.model = MBartForConditionalGeneration.from_pretrained(model_dir)
            elif self.model_type == "mt5":
                from transformers import MT5ForConditionalGeneration
                self.model = MT5ForConditionalGeneration.from_pretrained(model_dir)
            else:
                from transformers import AutoModelForSeq2SeqLM
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
            
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model.to(self.device)
            logger.info(f"Model loaded from {model_dir}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
