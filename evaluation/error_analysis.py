"""
Error analysis module for translation quality assessment.
Implements tools for analyzing and categorizing translation errors.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Set
import yaml
import re
from collections import Counter

logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    """Analyzes and categorizes translation errors."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the error analyzer.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Define error categories
        self.error_categories = {
            'terminology': 'Incorrect translation of domain-specific terms',
            'omission': 'Missing content from source text',
            'addition': 'Added content not in source text',
            'grammar': 'Grammatical errors in translation',
            'style': 'Inappropriate style or register',
            'word_order': 'Incorrect word order affecting meaning',
            'punctuation': 'Incorrect punctuation',
            'number': 'Incorrect translation of numbers or dates'
        }
        
        # Load glossary if available
        self.glossary = {}
        glossary_path = self.config['data']['glossary'].get('path')
        if glossary_path and os.path.exists(glossary_path):
            self._load_glossary(glossary_path)
    
    def _load_glossary(self, glossary_path: str) -> None:
        """
        Load domain glossary for terminology error detection.
        
        Args:
            glossary_path: Path to glossary file
        """
        try:
            df = pd.read_csv(glossary_path)
            
            if 'english_term' in df.columns and 'arabic_term' in df.columns:
                self.glossary = dict(zip(df['english_term'], df['arabic_term']))
                logger.info(f"Loaded {len(self.glossary)} terms from glossary")
            else:
                logger.error("Glossary file does not have required columns")
        
        except Exception as e:
            logger.error(f"Error loading glossary: {str(e)}")
    
    def detect_terminology_errors(self, 
                                 source_text: str, 
                                 translation: str,
                                 source_lang: str = 'en',
                                 target_lang: str = 'ar') -> List[Dict[str, str]]:
        """
        Detect terminology errors in translation.
        
        Args:
            source_text: Source text
            translation: Translation to analyze
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of detected terminology errors
        """
        if not self.glossary:
            logger.warning("No glossary available for terminology error detection")
            return []
        
        errors = []
        
        if source_lang == 'en' and target_lang == 'ar':
            # Check for English terms in the source that have glossary entries
            for en_term, ar_term in self.glossary.items():
                if en_term.lower() in source_text.lower():
                    # Check if the Arabic term is in the translation
                    if ar_term not in translation:
                        errors.append({
                            'category': 'terminology',
                            'source_term': en_term,
                            'expected_translation': ar_term,
                            'severity': 'high'
                        })
        
        return errors
    
    def detect_number_errors(self, 
                            source_text: str, 
                            translation: str) -> List[Dict[str, str]]:
        """
        Detect errors in number translation.
        
        Args:
            source_text: Source text
            translation: Translation to analyze
            
        Returns:
            List of detected number errors
        """
        errors = []
        
        # Extract numbers from source text
        source_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', source_text)
        
        # For Arabic, we need to handle both Arabic and Latin numerals
        # This is a simplified approach - in a real system, use more sophisticated methods
        arabic_digits = '٠١٢٣٤٥٦٧٨٩'
        latin_digits = '0123456789'
        digit_map = str.maketrans(latin_digits, arabic_digits)
        
        for num in source_numbers:
            # Check for Latin numerals in translation
            if num not in translation:
                # Check for Arabic numerals
                arabic_num = num.translate(digit_map)
                if arabic_num not in translation:
                    errors.append({
                        'category': 'number',
                        'source_number': num,
                        'severity': 'medium'
                    })
        
        return errors
    
    def analyze_errors(self, 
                      source_texts: List[str], 
                      translations: List[str],
                      reference_translations: Optional[List[str]] = None,
                      source_lang: str = 'en',
                      target_lang: str = 'ar') -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        """
        Analyze errors in translations.
        
        Args:
            source_texts: List of source texts
            translations: List of translations to analyze
            reference_translations: Optional list of reference translations
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of error analyses for each translation
        """
        if len(source_texts) != len(translations):
            logger.error("Mismatched lengths for source_texts and translations")
            return []
        
        if reference_translations and len(source_texts) != len(reference_translations):
            logger.error("Mismatched lengths for source_texts and reference_translations")
            return []
        
        analyses = []
        
        for i in range(len(source_texts)):
            source = source_texts[i]
            translation = translations[i]
            reference = reference_translations[i] if reference_translations else None
            
            # Detect various types of errors
            terminology_errors = self.detect_terminology_errors(source, translation, source_lang, target_lang)
            number_errors = self.detect_number_errors(source, translation)
            
            # Combine all errors
            all_errors = terminology_errors + number_errors
            
            analysis = {
                'source_text': source,
                'translation': translation,
                'reference_translation': reference,
                'errors': all_errors,
                'error_count': len(all_errors),
                'error_categories': Counter([error['category'] for error in all_errors])
            }
            
            analyses.append(analysis)
        
        return analyses
    
    def summarize_errors(self, analyses: List[Dict[str, Union[str, List[Dict[str, str]]]]]) -> Dict[str, Union[int, Dict[str, int]]]:
        """
        Summarize error analyses.
        
        Args:
            analyses: List of error analyses
            
        Returns:
            Summary of error statistics
        """
        if not analyses:
            logger.warning("No analyses provided for summarization")
            return {}
        
        total_errors = 0
        category_counts = Counter()
        severity_counts = Counter()
        
        for analysis in analyses:
            errors = analysis.get('errors', [])
            total_errors += len(errors)
            
            for error in errors:
                category_counts[error['category']] += 1
                severity_counts[error.get('severity', 'medium')] += 1
        
        # Calculate error rate
        total_translations = len(analyses)
        error_rate = total_errors / total_translations if total_translations > 0 else 0
        
        # Calculate category distribution
        category_distribution = {
            category: count / total_errors if total_errors > 0 else 0
            for category, count in category_counts.items()
        }
        
        # Calculate severity distribution
        severity_distribution = {
            severity: count / total_errors if total_errors > 0 else 0
            for severity, count in severity_counts.items()
        }
        
        summary = {
            'total_translations': total_translations,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'category_counts': dict(category_counts),
            'category_distribution': category_distribution,
            'severity_counts': dict(severity_counts),
            'severity_distribution': severity_distribution
        }
        
        return summary
    
    def export_analysis(self, 
                       analyses: List[Dict[str, Union[str, List[Dict[str, str]]]]],
                       summary: Dict[str, Union[int, Dict[str, int]]],
                       output_path: str) -> None:
        """
        Export error analysis to file.
        
        Args:
            analyses: List of error analyses
            summary: Summary of error statistics
            output_path: Path to save analysis
        """
        # Create a structured report
        report = {
            'summary': summary,
            'analyses': analyses
        }
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Error analysis exported to {output_path}")
        
        # Also create an Excel version with summary statistics
        excel_path = os.path.splitext(output_path)[0] + '.xlsx'
        
        # Summary sheet
        summary_data = {
            'Metric': ['Total Translations', 'Total Errors', 'Error Rate'],
            'Value': [
                summary['total_translations'],
                summary['total_errors'],
                f"{summary['error_rate']:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Category distribution sheet
        category_data = {
            'Category': list(summary['category_counts'].keys()),
            'Count': list(summary['category_counts'].values()),
            'Percentage': [f"{summary['category_distribution'][cat]:.2%}" 
                          for cat in summary['category_counts'].keys()]
        }
        
        category_df = pd.DataFrame(category_data)
        
        # Error details sheet
        error_rows = []
        for i, analysis in enumerate(analyses):
            for error in analysis.get('errors', []):
                error_rows.append({
                    'Translation Index': i,
                    'Source Text': analysis['source_text'],
                    'Translation': analysis['translation'],
                    'Error Category': error['category'],
                    'Severity': error.get('severity', 'medium'),
                    'Details': ', '.join([f"{k}: {v}" for k, v in error.items() 
                                         if k not in ['category', 'severity']])
                })
        
        error_df = pd.DataFrame(error_rows)
        
        # Save to Excel
        with pd.ExcelWriter(excel_path) as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            category_df.to_excel(writer, sheet_name='Categories', index=False)
            error_df.to_excel(writer, sheet_name='Error Details', index=False)
        
        logger.info(f"Error analysis summary exported to {excel_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    analyzer = ErrorAnalyzer()
    
    # Test error analysis
    source_texts = [
        "The inflation rate reached 5.2% in December 2022.",
        "The central bank increased the interest rate by 50 basis points."
    ]
    
    translations = [
        "بلغ معدل التضخم 5 في ديسمبر 2022.",  # Missing .2 in 5.2%
        "رفع البنك المركزي معدل الفائدة بمقدار 50 نقطة أساسية."  # Correct
    ]
    
    analyses = analyzer.analyze_errors(source_texts, translations)
    summary = analyzer.summarize_errors(analyses)
    
    print(f"Error summary: {summary}")
    
    # Export analysis
    analyzer.export_analysis(analyses, summary, "evaluation/error_analysis_example.json") 