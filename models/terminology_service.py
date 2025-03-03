"""
Terminology Service for Financial Translation System.
This module merges glossary management and integration functionalities
to provide a unified interface for domain-specific terminology handling.
"""

import os
import re
import logging
import pandas as pd
from typing import List, Tuple, Optional
import yaml

logger = logging.getLogger(__name__)

class TerminologyService:
    """Unified service for managing and integrating domain-specific terminology."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the Terminology Service.
        
        Args:
            config_path: Path to configuration file.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config['data'].get('glossary', {})
        self.integration_config = config['model'].get('glossary_integration', {})
        self.glossary_path = self.config.get('path', "data/glossary/financial_terms.csv")
        self.confidence_threshold = self.integration_config.get('confidence_threshold', 0.85)
        
        # Dictionaries for terminology
        self.en_to_ar = {}
        self.ar_to_en = {}
        
        # Load glossary and prepare regex patterns
        self.load_glossary()
        self.prepare_term_patterns()
    
    def load_glossary(self) -> None:
        """Load glossary from CSV file."""
        if not os.path.exists(self.glossary_path):
            logger.warning(f"Glossary file not found at {self.glossary_path}. Creating empty glossary.")
            os.makedirs(os.path.dirname(self.glossary_path), exist_ok=True)
            self.create_empty_glossary()
        
        try:
            df = pd.read_csv(self.glossary_path)
            required_columns = ['english_term', 'arabic_term']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Glossary must contain columns: {required_columns}")
            
            self.en_to_ar = dict(zip(df['english_term'], df['arabic_term']))
            self.ar_to_en = dict(zip(df['arabic_term'], df['english_term']))
            logger.info(f"Loaded {len(self.en_to_ar)} terms from glossary")
        except Exception as e:
            logger.error(f"Error loading glossary: {str(e)}")
            self.en_to_ar = {}
            self.ar_to_en = {}
    
    def create_empty_glossary(self) -> None:
        """Create an empty glossary file with required columns."""
        import csv
        with open(self.glossary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['english_term', 'arabic_term', 'domain', 'notes'])
        logger.info(f"Created empty glossary at {self.glossary_path}")
    
    def prepare_term_patterns(self) -> None:
        """Prepare unified regex patterns for English and Arabic term matching."""
        # English pattern
        if self.en_to_ar:
            sorted_en_terms = sorted(self.en_to_ar.keys(), key=len, reverse=True)
            escaped_en_terms = [re.escape(term) for term in sorted_en_terms]
            pattern_str_en = r'\b(' + '|'.join(escaped_en_terms) + r')\b'
            try:
                self.en_pattern = re.compile(pattern_str_en, re.IGNORECASE)
                logger.debug("Created English term matching pattern")
            except re.error:
                logger.error("Failed to create English term matching pattern")
                self.en_pattern = None
        else:
            self.en_pattern = None
        
        # Arabic pattern
        if self.ar_to_en:
            sorted_ar_terms = sorted(self.ar_to_en.keys(), key=len, reverse=True)
            escaped_ar_terms = [re.escape(term) for term in sorted_ar_terms]
            pattern_str_ar = r'(' + '|'.join(escaped_ar_terms) + r')'
            try:
                self.ar_pattern = re.compile(pattern_str_ar)
                logger.debug("Created Arabic term matching pattern")
            except re.error:
                logger.error("Failed to create Arabic term matching pattern")
                self.ar_pattern = None
        else:
            self.ar_pattern = None
    
    def identify_terms(self, text: str, lang: str = 'en') -> List[Tuple[str, int, int]]:
        """
        Identify glossary terms in a text.
        
        Args:
            text: Text to search for terms.
            lang: Language code ('en' or 'ar').
            
        Returns:
            List of (term, start_pos, end_pos) tuples.
        """
        if not text:
            return []
        pattern = self.en_pattern if lang == 'en' else self.ar_pattern
        if not pattern:
            return []
        matches = []
        for match in pattern.finditer(text):
            matches.append((match.group(0), match.start(), match.end()))
        return matches
    
    def get_term_translation(self, term: str, source_lang: str = 'en') -> Optional[str]:
        """
        Get translation for a term from the glossary.
        
        Args:
            term: Term to translate.
            source_lang: Source language code.
            
        Returns:
            Translated term or None if not found.
        """
        if source_lang == 'en':
            return self.en_to_ar.get(term, None)
        else:
            return self.ar_to_en.get(term, None)
    
    def get_constraints(self, text: str, tokenizer, source_lang: str = 'en') -> List[List[int]]:
        """
        Get token constraints for glossary terms in the text.
        
        Args:
            text: Source text.
            tokenizer: Tokenizer to encode constraint tokens.
            source_lang: Source language code.
            
        Returns:
            List of tokenized constraints (each constraint is a list of token IDs).
        """
        constraints = []
        for term, _, _ in self.identify_terms(text, lang=source_lang):
            translation = self.get_term_translation(term, source_lang)
            if translation:
                term_tokens = tokenizer.encode(translation, add_special_tokens=False)
                if term_tokens:
                    constraints.append(term_tokens)
        return constraints
