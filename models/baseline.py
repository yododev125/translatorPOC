"""
Baseline translation model module.
Implements simple baseline models and cloud service integrations for comparison.
"""

import os
import logging
import requests
import json
import time
from typing import Dict, List, Optional, Union
import yaml
import pandas as pd
from sacrebleu import corpus_bleu

logger = logging.getLogger(__name__)

class BaselineTranslator:
    """Base class for baseline translation models."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the baseline translator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'ar') -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        raise NotImplementedError("Subclasses must implement translate method")
    
    def batch_translate(self, texts: List[str], source_lang: str = 'en', target_lang: str = 'ar') -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        translations = []
        for text in texts:
            try:
                translation = self.translate(text, source_lang, target_lang)
                translations.append(translation)
            except Exception as e:
                logger.error(f"Error translating text: {str(e)}")
                translations.append("")
        
        return translations
    
    def evaluate(self, test_data: pd.DataFrame, source_col: str = 'english', target_col: str = 'arabic') -> Dict[str, float]:
        """
        Evaluate the baseline model on test data.
        
        Args:
            test_data: DataFrame with test data
            source_col: Column name for source texts
            target_col: Column name for reference translations
            
        Returns:
            Dictionary with evaluation metrics
        """
        source_texts = test_data[source_col].tolist()
        reference_texts = test_data[target_col].tolist()
        
        # Translate source texts
        source_lang = 'en' if source_col == 'english' else 'ar'
        target_lang = 'ar' if target_col == 'arabic' else 'en'
        
        translated_texts = self.batch_translate(source_texts, source_lang, target_lang)
        
        # Calculate BLEU score
        bleu = corpus_bleu(translated_texts, [reference_texts])
        
        # Calculate simple accuracy (exact match)
        exact_matches = sum(1 for hyp, ref in zip(translated_texts, reference_texts) if hyp == ref)
        accuracy = exact_matches / len(reference_texts) if reference_texts else 0
        
        results = {
            'bleu': bleu.score,
            'exact_match_accuracy': accuracy,
            'num_samples': len(source_texts)
        }
        
        logger.info(f"Evaluation results: {results}")
        return results


class AzureTranslator(BaselineTranslator):
    """Baseline translator using Azure Translator API."""
    
    def __init__(self, config_path: str = "../config.yaml", api_key: Optional[str] = None):
        """
        Initialize the Azure translator.
        
        Args:
            config_path: Path to configuration file
            api_key: Azure Translator API key (optional, can be in config)
        """
        super().__init__(config_path)
        
        # Get API key from config or parameter
        self.api_key = api_key or os.environ.get('AZURE_TRANSLATOR_KEY')
        if not self.api_key:
            logger.warning("Azure Translator API key not provided. Set AZURE_TRANSLATOR_KEY environment variable.")
        
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        self.location = "global"  # Change to your Azure service location if different
    
    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'ar') -> str:
        """
        Translate text using Azure Translator API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if not self.api_key:
            logger.error("Azure Translator API key not available")
            return ""
        
        if not text:
            return ""
        
        # Prepare request
        path = '/translate'
        constructed_url = self.endpoint + path
        
        params = {
            'api-version': '3.0',
            'from': source_lang,
            'to': target_lang
        }
        
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Ocp-Apim-Subscription-Region': self.location,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        body = [{
            'text': text
        }]
        
        # Make request
        try:
            response = requests.post(constructed_url, params=params, headers=headers, json=body)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            result = response.json()
            
            if result and len(result) > 0 and 'translations' in result[0]:
                return result[0]['translations'][0]['text']
            else:
                logger.warning(f"Unexpected response format: {result}")
                return ""
        
        except Exception as e:
            logger.error(f"Error calling Azure Translator API: {str(e)}")
            return ""
    
    def batch_translate(self, texts: List[str], source_lang: str = 'en', target_lang: str = 'ar') -> List[str]:
        """
        Translate a batch of texts using Azure Translator API.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not self.api_key:
            logger.error("Azure Translator API key not available")
            return [""] * len(texts)
        
        if not texts:
            return []
        
        # Azure has a limit on batch size, so we process in chunks
        batch_size = 25  # Azure's limit is 100, but we use a smaller value
        all_translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Prepare request
            path = '/translate'
            constructed_url = self.endpoint + path
            
            params = {
                'api-version': '3.0',
                'from': source_lang,
                'to': target_lang
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.location,
                'Content-type': 'application/json'
            }
            
            body = [{'text': text} for text in batch]
            
            # Make request
            try:
                response = requests.post(constructed_url, params=params, headers=headers, json=body)
                response.raise_for_status()
                
                result = response.json()
                
                batch_translations = []
                for item in result:
                    if 'translations' in item and len(item['translations']) > 0:
                        batch_translations.append(item['translations'][0]['text'])
                    else:
                        batch_translations.append("")
                
                all_translations.extend(batch_translations)
                
                # Respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in batch translation: {str(e)}")
                # Return empty strings for this batch
                all_translations.extend([""] * len(batch))
        
        return all_translations


class DictionaryBasedTranslator(BaselineTranslator):
    """Simple dictionary-based translator using the domain glossary."""
    
    def __init__(self, config_path: str = "../config.yaml", glossary_path: Optional[str] = None):
        """
        Initialize the dictionary-based translator.
        
        Args:
            config_path: Path to configuration file
            glossary_path: Path to glossary file (optional, can be in config)
        """
        super().__init__(config_path)
        
        # Get glossary path from config or parameter
        self.glossary_path = glossary_path or self.config['data']['glossary']['path']
        
        # Load glossary
        self.en_to_ar = {}
        self.ar_to_en = {}
        self._load_glossary()
    
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
    
    def translate(self, text: str, source_lang: str = 'en', target_lang: str = 'ar') -> str:
        """
        Translate text using dictionary substitution.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text with known terms replaced
        """
        if not text:
            return ""
        
        # This is a very simplistic approach that just replaces known terms
        # In a real system, you would need proper tokenization, word order handling, etc.
        
        if source_lang == 'en' and target_lang == 'ar':
            # English to Arabic
            for en_term, ar_term in self.en_to_ar.items():
                # Simple case-insensitive replacement
                pattern = re.compile(re.escape(en_term), re.IGNORECASE)
                text = pattern.sub(ar_term, text)
            
        elif source_lang == 'ar' and target_lang == 'en':
            # Arabic to English
            for ar_term, en_term in self.ar_to_en.items():
                text = text.replace(ar_term, en_term)
        
        return text

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    translator = DictionaryBasedTranslator()
    
    # Test translation
    en_text = "The central bank increased the interest rate to combat inflation."
    ar_translation = translator.translate(en_text, 'en', 'ar')
    print(f"English: {en_text}")
    print(f"Arabic: {ar_translation}") 