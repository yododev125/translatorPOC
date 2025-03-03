"""
Glossary integration module for translation models.
Implements methods to incorporate domain-specific terminology into translations.
"""

import os
import re
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
import yaml

logger = logging.getLogger(__name__)

class GlossaryIntegrator:
    """Integrates domain-specific glossary into translation process."""
    
    def __init__(self, 
                 glossary_path: str,
                 tokenizer,
                 config_path: str = "../config.yaml"):
        """
        Initialize the glossary integrator.
        
        Args:
            glossary_path: Path to glossary file
            tokenizer: Tokenizer used by the translation model
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['model']['glossary_integration']
        
        self.glossary_path = glossary_path
        self.tokenizer = tokenizer
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)
        
        # Load glossary
        self.en_to_ar = {}
        self.ar_to_en = {}
        self._load_glossary()
        
        # Prepare term patterns for efficient matching
        self._prepare_term_patterns()
    
    def _load_glossary(self) -> None:
        """Load glossary from CSV file."""
        try:
            df = pd.read_csv(self.glossary_path)
            
            if 'english_term' in df.columns and 'arabic_term' in df.columns:
                self.en_to_ar = dict(zip(df['english_term'], df['arabic_term']))
                self.ar_to_en = dict(zip(df['arabic_term'], df['english_term']))
                logger.info(f"Loaded {len(self.en_to_ar)} terms from glossary")
            else:
                logger.error("Glossary file does not have required columns")
        
        except Exception as e:
            logger.error(f"Error loading glossary: {str(e)}")
    
    def _prepare_term_patterns(self) -> None:
        """Prepare regex patterns for term matching."""
        if not self.en_to_ar:
            self.en_pattern = None
            return
        
        # Sort terms by length (longest first) to ensure proper matching
        sorted_en_terms = sorted(self.en_to_ar.keys(), key=len, reverse=True)
        
        # Escape special regex characters and join with OR
        escaped_terms = [re.escape(term) for term in sorted_en_terms]
        pattern_str = r'\b(' + '|'.join(escaped_terms) + r')\b'
        
        try:
            self.en_pattern = re.compile(pattern_str, re.IGNORECASE)
            logger.debug("Created English term matching pattern")
        except re.error:
            logger.error("Failed to create English term matching pattern")
            self.en_pattern = None
        
        # Do the same for Arabic terms
        if self.ar_to_en:
            sorted_ar_terms = sorted(self.ar_to_en.keys(), key=len, reverse=True)
            escaped_terms = [re.escape(term) for term in sorted_ar_terms]
            pattern_str = '(' + '|'.join(escaped_terms) + ')'
            
            try:
                self.ar_pattern = re.compile(pattern_str)
                logger.debug("Created Arabic term matching pattern")
            except re.error:
                logger.error("Failed to create Arabic term matching pattern")
                self.ar_pattern = None
    
    def identify_terms(self, text: str, lang: str = 'en') -> List[Tuple[str, int, int]]:
        """
        Identify glossary terms in a text.
        
        Args:
            text: Text to search for terms
            lang: Language code ('en' or 'ar')
            
        Returns:
            List of (term, start_pos, end_pos) tuples
        """
        if not text:
            return []
        
        pattern = self.en_pattern if lang == 'en' else self.ar_pattern
        if not pattern:
            return []
        
        matches = []
        for match in pattern.finditer(text):
            term = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            matches.append((term, start_pos, end_pos))
        
        return matches
    
    def get_term_translation(self, term: str, source_lang: str = 'en') -> Optional[str]:
        """
        Get translation for a term from the glossary.
        
        Args:
            term: Term to translate
            source_lang: Source language code
            
        Returns:
            Translated term or None if not found
        """
        if source_lang == 'en':
            return self.en_to_ar.get(term.lower(), None)
        else:
            return self.ar_to_en.get(term, None)
    
    def constrained_decoding(self, 
                             model, 
                             input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor,
                             source_text: str,
                             source_lang: str = 'en',
                             max_length: int = 128,
                             num_beams: int = 4) -> torch.Tensor:
        """
        Perform constrained decoding to enforce terminology.
        
        Args:
            model: Translation model
            input_ids: Input token IDs
            attention_mask: Attention mask
            source_text: Original source text
            source_lang: Source language code
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search
            
        Returns:
            Generated token IDs with terminology constraints
        """
        # Identify terms in the source text
        term_matches = self.identify_terms(source_text, source_lang)
        
        if not term_matches:
            # No terms found, perform regular decoding
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Prepare constraints for each identified term
        constraints = []
        for term, _, _ in term_matches:
            translation = self.get_term_translation(term, source_lang)
            if translation:
                # Tokenize the translated term
                term_tokens = self.tokenizer.encode(
                    translation, 
                    add_special_tokens=False, 
                    return_tensors="pt"
                )[0]
                
                constraints.append(term_tokens)
        
        # If using HuggingFace's generate with constraints
        try:
            from transformers.generation_logits_process import (
                LogitsProcessorList,
                ConstrainedBeamSearchScorer
            )
            
            # Set up constrained beam search
            constraint_list = []
            for constraint in constraints:
                constraint_list.append(constraint.tolist())
            
            # Generate with constraints
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                constraints=constraint_list
            )
            
            return outputs
            
        except (ImportError, AttributeError):
            # Fall back to regular generation if constrained generation not available
            logger.warning("Constrained decoding not available, falling back to regular decoding")
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
    
    def post_process_translation(self, 
                                source_text: str, 
                                translation: str,
                                source_lang: str = 'en') -> str:
        """
        Post-process translation to ensure terminology consistency.
        
        Args:
            source_text: Original source text
            translation: Generated translation
            source_lang: Source language code
            
        Returns:
            Post-processed translation with correct terminology
        """
        # Identify terms in the source text
        term_matches = self.identify_terms(source_text, source_lang)
        
        if not term_matches:
            return translation
        
        # For each term, ensure it's correctly translated
        for term, _, _ in term_matches:
            correct_translation = self.get_term_translation(term, source_lang)
            
            if not correct_translation:
                continue
            
            # Tokenize the term and its translation
            term_tokens = self.tokenizer.tokenize(term)
            trans_tokens = self.tokenizer.tokenize(correct_translation)
            
            # Simple replacement - in a real system, use more sophisticated methods
            # such as alignment information or attention weights
            if source_lang == 'en':
                # Try to find incorrect translations of the term
                # This is a simplified approach and might not work well in all cases
                term_translation = self.tokenizer.decode(
                    self.tokenizer.encode(term, add_special_tokens=False)
                )
                
                if term_translation in translation:
                    translation = translation.replace(term_translation, correct_translation)
            
        return translation
    
    def attention_biasing(self, 
                          model, 
                          input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor,
                          source_text: str,
                          source_lang: str = 'en',
                          max_length: int = 128,
                          num_beams: int = 4) -> torch.Tensor:
        """
        Perform attention biasing to encourage correct terminology.
        
        Args:
            model: Translation model
            input_ids: Input token IDs
            attention_mask: Attention mask
            source_text: Original source text
            source_lang: Source language code
            max_length: Maximum length of generated translation
            num_beams: Number of beams for beam search
            
        Returns:
            Generated token IDs with biased attention
        """
        # This is a placeholder for attention biasing implementation
        # In a real system, you would modify the attention weights during generation
        # to bias towards correct terminology translations
        
        logger.warning("Attention biasing not fully implemented, using regular generation")
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage (requires a tokenizer)
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
    integrator = GlossaryIntegrator("data/glossary/financial_terms.csv", tokenizer)
    
    # Test term identification
    text = "The central bank increased the interest rate to combat inflation."
    terms = integrator.identify_terms(text)
    print(f"Identified terms: {terms}")
    
    # Test post-processing
    translation = "قام البنك المركزي بزيادة معدل الفائدة لمكافحة التضخم."
    processed = integrator.post_process_translation(text, translation)
    print(f"Post-processed: {processed}") 