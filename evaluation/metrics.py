"""
Evaluation metrics module for translation quality assessment.
Implements various metrics to evaluate translation quality.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import yaml
from sacrebleu import corpus_bleu, BLEU
from sacrebleu.metrics import TER
import torch
from bert_score import BERTScorer

logger = logging.getLogger(__name__)

class TranslationEvaluator:
    """Evaluates translation quality using various metrics."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the translation evaluator.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['evaluation']
        
        self.metrics = self.config.get('metrics', ['bleu'])
        self.bert_scorer = None
        if 'bertscore' in self.metrics:
            try:
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                logger.info("Initialized BERTScore")
            except Exception as e:
                logger.error(f"Failed to initialize BERTScore: {str(e)}")
                self.metrics.remove('bertscore')
    
    def calculate_bleu(self, 
                      hypotheses: List[str], 
                      references: List[str],
                      smooth: bool = True) -> float:
        """
        Calculate BLEU score.
        """
        if not hypotheses or not references:
            logger.warning("Empty hypotheses or references for BLEU calculation")
            return 0.0
        refs = [[ref] for ref in references]
        try:
            bleu = BLEU(smooth=smooth)
            score = bleu.corpus_score(hypotheses, refs)
            return score.score
        except Exception as e:
            logger.error(f"Error calculating BLEU: {str(e)}")
            return 0.0
    
    def calculate_ter(self, 
                      hypotheses: List[str], 
                      references: List[str]) -> float:
        """
        Calculate Translation Error Rate (TER).
        """
        if not hypotheses or not references:
            logger.warning("Empty hypotheses or references for TER calculation")
            return 1.0
        refs = [[ref] for ref in references]
        try:
            ter = TER()
            score = ter.corpus_score(hypotheses, refs)
            return score.score
        except Exception as e:
            logger.error(f"Error calculating TER: {str(e)}")
            return 1.0
    
    def calculate_bertscore(self, 
                            hypotheses: List[str], 
                            references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore.
        """
        if not self.bert_scorer:
            logger.error("BERTScore not initialized")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        if not hypotheses or not references:
            logger.warning("Empty hypotheses or references for BERTScore calculation")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        try:
            P, R, F1 = self.bert_scorer.score(hypotheses, references)
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {str(e)}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def calculate_meteor(self, 
                         hypotheses: List[str], 
                         references: List[str]) -> float:
        """
        Calculate METEOR score.
        """
        logger.warning("METEOR calculation not fully implemented")
        return 0.0
    
    def evaluate_all(self, 
                     hypotheses: List[str], 
                     references: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate all configured metrics.
        """
        results = {}
        if 'bleu' in self.metrics:
            results['bleu'] = self.calculate_bleu(hypotheses, references)
        if 'ter' in self.metrics:
            results['ter'] = self.calculate_ter(hypotheses, references)
        if 'bertscore' in self.metrics and self.bert_scorer:
            results['bertscore'] = self.calculate_bertscore(hypotheses, references)
        if 'meteor' in self.metrics:
            results['meteor'] = self.calculate_meteor(hypotheses, references)
        logger.info(f"Evaluation results: {results}")
        return results
    
    def evaluate_by_category(self, 
                             hypotheses: List[str], 
                             references: List[str],
                             categories: List[str]) -> Dict[str, Dict[str, Union[float, Dict[str, float]]]]:
        if len(hypotheses) != len(references) or len(hypotheses) != len(categories):
            logger.error("Mismatched lengths for hypotheses, references, and categories")
            return {}
        category_indices = {}
        for i, category in enumerate(categories):
            category_indices.setdefault(category, []).append(i)
        results = {}
        for category, indices in category_indices.items():
            cat_hypotheses = [hypotheses[i] for i in indices]
            cat_references = [references[i] for i in indices]
            results[category] = self.evaluate_all(cat_hypotheses, cat_references)
            results[category]['count'] = len(indices)
        return results
    
    def save_results(self, 
                     results: Dict[str, Union[float, Dict[str, float]]],
                     output_path: str) -> None:
        flat_results = {}
        for metric, value in results.items():
            if isinstance(value, dict):
                for submetric, subvalue in value.items():
                    flat_results[f"{metric}_{submetric}"] = subvalue
            else:
                flat_results[metric] = value
        try:
            pd.DataFrame([flat_results]).to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

# --- New Enhancement: Domain-Specific Term Accuracy ---

def calculate_term_accuracy(hypotheses: List[str], references: List[str], glossary: dict) -> float:
    """
    Calculate financial term accuracy.
    
    Args:
        hypotheses: List of translated texts.
        references: List of reference translations.
        glossary: Dictionary mapping English terms to Arabic terms.
        
    Returns:
        Term accuracy score.
    """
    if not hypotheses:
        return 0.0
    matches = 0
    for hyp, ref in zip(hypotheses, references):
        for term, translation in glossary.items():
            if term in ref and translation in hyp:
                matches += 1
    return matches / len(hypotheses)
