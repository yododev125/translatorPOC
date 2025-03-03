"""
Preprocessing module for financial translation data.
Handles cleaning, normalization, and alignment of parallel texts.
"""

import re
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import yaml
from sacremoses import MosesPunctNormalizer, MosesTokenizer
import spacy
from camel_tools.tokenizers.word import simple_word_tokenize as arabic_tokenize

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocesses text data for translation model training."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data']['preprocessing']
        
        self.max_seq_len = self.config['max_sequence_length']
        self.min_sent_len = self.config['min_sentence_length']
        
        # Initialize tokenizers and normalizers
        self.en_normalizer = MosesPunctNormalizer(lang='en')
        self.en_tokenizer = MosesTokenizer(lang='en')
        
        # Load spaCy for sentence segmentation
        try:
            self.en_nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
            self.en_nlp.add_pipe('sentencizer')
        except OSError:
            logger.warning("Downloading spaCy English model")
            spacy.cli.download('en_core_web_sm')
            self.en_nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
            self.en_nlp.add_pipe('sentencizer')
    
    def clean_english(self, text: str) -> str:
        """
        Clean and normalize English text.
        
        Args:
            text: Raw English text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation
        text = self.en_normalizer.normalize(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,;:!?\'\"()\[\]{}]', '', text)
        
        return text.strip()
    
    def clean_arabic(self, text: str) -> str:
        """
        Clean and normalize Arabic text.
        
        Args:
            text: Raw Arabic text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize Arabic-specific characters
        text = re.sub(r'[إأآا]', 'ا', text)  # Normalize alif
        text = re.sub(r'[يى]', 'ي', text)    # Normalize ya
        text = re.sub(r'[ؤئ]', 'ء', text)    # Normalize hamza
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text.strip()
    
    def segment_sentences(self, text: str, lang: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to segment
            lang: Language code ('en' or 'ar')
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        if lang == 'en':
            doc = self.en_nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        elif lang == 'ar':
            # Simple rule-based sentence splitting for Arabic
            # In a production system, use a proper Arabic sentence segmenter
            sentences = re.split(r'[.!?؟،]', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > self.min_sent_len]
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
        return sentences
    
    def align_sentences(self, en_sentences: List[str], ar_sentences: List[str]) -> List[Tuple[str, str]]:
        """
        Align English and Arabic sentences.
        This is a simplified alignment - in a real system, use more sophisticated methods.
        
        Args:
            en_sentences: List of English sentences
            ar_sentences: List of Arabic sentences
            
        Returns:
            List of (English, Arabic) sentence pairs
        """
        # This is a very simplified alignment assuming 1:1 correspondence
        # In a real system, use length-based or more sophisticated alignment techniques
        
        # Take the minimum length to avoid index errors
        min_len = min(len(en_sentences), len(ar_sentences))
        aligned_pairs = [(en_sentences[i], ar_sentences[i]) for i in range(min_len)]
        
        # Filter out pairs where either sentence is too short
        aligned_pairs = [(en, ar) for en, ar in aligned_pairs 
                         if len(en.split()) >= self.min_sent_len 
                         and len(ar_tokenize(ar)) >= self.min_sent_len]
        
        return aligned_pairs
    
    def preprocess_document_pair(self, en_doc: str, ar_doc: str) -> List[Tuple[str, str]]:
        """
        Preprocess a pair of documents and align their sentences.
        
        Args:
            en_doc: English document
            ar_doc: Arabic document
            
        Returns:
            List of aligned sentence pairs
        """
        # Clean the documents
        en_clean = self.clean_english(en_doc)
        ar_clean = self.clean_arabic(ar_doc)
        
        # Segment into sentences
        en_sentences = self.segment_sentences(en_clean, 'en')
        ar_sentences = self.segment_sentences(ar_clean, 'ar')
        
        # Align sentences
        aligned_pairs = self.align_sentences(en_sentences, ar_sentences)
        
        return aligned_pairs
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a dataframe containing parallel documents.
        
        Args:
            df: DataFrame with 'english' and 'arabic' columns
            
        Returns:
            DataFrame with aligned sentence pairs
        """
        aligned_pairs = []
        document_ids = []
        sources = []
        
        for idx, row in df.iterrows():
            try:
                pairs = self.preprocess_document_pair(row['english'], row['arabic'])
                aligned_pairs.extend(pairs)
                document_ids.extend([row['document_id']] * len(pairs))
                sources.extend([row['source']] * len(pairs))
            except Exception as e:
                logger.error(f"Error processing document {row.get('document_id', idx)}: {str(e)}")
        
        processed_df = pd.DataFrame({
            'english': [pair[0] for pair in aligned_pairs],
            'arabic': [pair[1] for pair in aligned_pairs],
            'document_id': document_ids,
            'source': sources
        })
        
        logger.info(f"Preprocessed {len(df)} documents into {len(processed_df)} sentence pairs")
        return processed_df
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            df: DataFrame with preprocessed data
            
        Returns:
            Dictionary with train, val, and test DataFrames
        """
        # Group by document_id to keep sentences from the same document together
        doc_ids = df['document_id'].unique()
        np.random.shuffle(doc_ids)
        
        n_docs = len(doc_ids)
        train_size = int(n_docs * self.config['train_split'])
        val_size = int(n_docs * self.config['val_split'])
        
        train_ids = doc_ids[:train_size]
        val_ids = doc_ids[train_size:train_size+val_size]
        test_ids = doc_ids[train_size+val_size:]
        
        train_df = df[df['document_id'].isin(train_ids)]
        val_df = df[df['document_id'].isin(val_ids)]
        test_df = df[df['document_id'].isin(test_ids)]
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = TextPreprocessor()
    
    # Example usage
    sample_df = pd.DataFrame({
        'english': ['This is a sample financial document. It contains important information.'],
        'arabic': ['هذه وثيقة مالية نموذجية. تحتوي على معلومات مهمة.'],
        'document_id': ['sample001'],
        'source': ['Sample']
    })
    
    processed = preprocessor.preprocess_dataframe(sample_df)
    print(processed) 