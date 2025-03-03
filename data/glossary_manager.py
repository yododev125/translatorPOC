"""
Glossary management module for financial translation system.
Handles domain-specific terminology and ensures consistent translations.
"""

import os
import csv
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
import yaml
import re

logger = logging.getLogger(__name__)

class GlossaryManager:
    """Manages domain-specific terminology for financial translation."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the glossary manager.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data']['glossary']
        
        self.glossary_path = self.config['path']
        self.terms = {}  # {english_term: arabic_term}
        self.load_glossary()
    
    def load_glossary(self) -> None:
        """Load glossary from CSV file."""
        if not os.path.exists(self.glossary_path):
            logger.warning(f"Glossary file not found at {self.glossary_path}. Creating empty glossary.")
            os.makedirs(os.path.dirname(self.glossary_path), exist_ok=True)
            self._create_empty_glossary()
        
        try:
            df = pd.read_csv(self.glossary_path)
            required_columns = ['english_term', 'arabic_term']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Glossary must contain columns: {required_columns}")
            
            # Load terms into dictionary
            self.terms = dict(zip(df['english_term'], df['arabic_term']))
            logger.info(f"Loaded {len(self.terms)} terms from glossary")
            
            # Create term pattern for efficient matching
            self._create_term_pattern()
        except Exception as e:
            logger.error(f"Error loading glossary: {str(e)}")
            self.terms = {}
    
    def _create_empty_glossary(self) -> None:
        """Create an empty glossary file with required columns."""
        with open(self.glossary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['english_term', 'arabic_term', 'domain', 'notes'])
        logger.info(f"Created empty glossary at {self.glossary_path}")
    
    def _create_term_pattern(self) -> None:
        """Create regex pattern for efficient term matching."""
        if not self.terms:
            self.term_pattern = None
            return
        
        # Sort terms by length (longest first) to ensure proper matching
        sorted_terms = sorted(self.terms.keys(), key=len, reverse=True)
        
        # Escape special regex characters and join with OR
        escaped_terms = [re.escape(term) for term in sorted_terms]
        pattern_str = r'\b(' + '|'.join(escaped_terms) + r')\b'
        
        try:
            self.term_pattern = re.compile(pattern_str, re.IGNORECASE)
            logger.debug("Created term matching pattern")
        except re.error:
            logger.error("Failed to create term matching pattern")
            self.term_pattern = None
    
    def add_term(self, english_term: str, arabic_term: str, domain: str = "finance", notes: str = "") -> bool:
        """
        Add a new term to the glossary.
        
        Args:
            english_term: Term in English
            arabic_term: Term in Arabic
            domain: Domain category (e.g., banking, investment)
            notes: Additional notes about the term
            
        Returns:
            Success status
        """
        if not english_term or not arabic_term:
            logger.warning("Cannot add empty term to glossary")
            return False
        
        # Add to in-memory dictionary
        self.terms[english_term] = arabic_term
        
        # Update the CSV file
        try:
            df = pd.read_csv(self.glossary_path)
            
            # Check if term already exists
            if english_term in df['english_term'].values:
                df.loc[df['english_term'] == english_term, 'arabic_term'] = arabic_term
                df.loc[df['english_term'] == english_term, 'domain'] = domain
                df.loc[df['english_term'] == english_term, 'notes'] = notes
            else:
                new_row = pd.DataFrame({
                    'english_term': [english_term],
                    'arabic_term': [arabic_term],
                    'domain': [domain],
                    'notes': [notes]
                })
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.to_csv(self.glossary_path, index=False)
            
            # Recreate the term pattern
            self._create_term_pattern()
            
            logger.info(f"Added term '{english_term}' to glossary")
            return True
        except Exception as e:
            logger.error(f"Error adding term to glossary: {str(e)}")
            return False
    
    def get_translation(self, english_term: str) -> Optional[str]:
        """
        Get the Arabic translation for an English term.
        
        Args:
            english_term: Term to translate
            
        Returns:
            Arabic translation or None if not found
        """
        return self.terms.get(english_term.lower(), None)
    
    def identify_terms(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Identify glossary terms in a text.
        
        Args:
            text: Text to search for terms
            
        Returns:
            List of (term, start_pos, end_pos) tuples
        """
        if not text or not self.term_pattern:
            return []
        
        matches = []
        for match in self.term_pattern.finditer(text):
            term = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            matches.append((term, start_pos, end_pos))
        
        return matches
    
    def replace_terms(self, text: str, lang: str = 'en') -> str:
        """
        Replace terms in text with their translations.
        
        Args:
            text: Text to process
            lang: Source language ('en' or 'ar')
            
        Returns:
            Text with terms replaced
        """
        if not text or not self.term_pattern:
            return text
        
        if lang == 'en':
            # Replace English terms with Arabic translations
            def replace_func(match):
                term = match.group(0)
                translation = self.get_translation(term)
                return translation if translation else term
            
            return self.term_pattern.sub(replace_func, text)
        else:
            # Not implemented for Arabic to English
            logger.warning("Term replacement from Arabic to English not implemented")
            return text
    
    def export_glossary(self, output_format: str = 'csv') -> str:
        """
        Export the glossary in various formats.
        
        Args:
            output_format: Format to export ('csv', 'json', 'txt')
            
        Returns:
            Path to exported file
        """
        base_path = os.path.splitext(self.glossary_path)[0]
        
        if output_format == 'csv':
            # Already in CSV format
            return self.glossary_path
        
        elif output_format == 'json':
            output_path = f"{base_path}.json"
            try:
                df = pd.read_csv(self.glossary_path)
                df.to_json(output_path, orient='records', force_ascii=False)
                logger.info(f"Exported glossary to JSON: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error exporting to JSON: {str(e)}")
                return ""
        
        elif output_format == 'txt':
            output_path = f"{base_path}.txt"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for en_term, ar_term in self.terms.items():
                        f.write(f"{en_term} | {ar_term}\n")
                logger.info(f"Exported glossary to TXT: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Error exporting to TXT: {str(e)}")
                return ""
        
        else:
            logger.error(f"Unsupported export format: {output_format}")
            return ""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = GlossaryManager()
    
    # Example usage
    manager.add_term("inflation", "تضخم", "economics")
    manager.add_term("interest rate", "سعر الفائدة", "banking")
    
    # Test term identification
    text = "The central bank increased the interest rate to combat inflation."
    terms = manager.identify_terms(text)
    print(f"Found terms: {terms}")
    
    # Test term replacement
    replaced = manager.replace_terms(text)
    print(f"With replacements: {replaced}") 