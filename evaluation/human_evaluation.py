"""
Human evaluation module for translation quality assessment.
Implements tools for collecting and analyzing human feedback on translations.
"""

import os
import logging
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

class HumanEvaluator:
    """Manages human evaluation of translations."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the human evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['evaluation']['human_evaluation']
        
        self.evaluators_per_doc = self.config.get('evaluators_per_document', 2)
        self.evaluation_aspects = self.config.get('evaluation_aspects', 
                                                ['accuracy', 'fluency', 'terminology', 'style'])
        
        self.feedback_dir = "evaluation/human_feedback"
        os.makedirs(self.feedback_dir, exist_ok=True)
    
    def create_evaluation_form(self, 
                              source_texts: List[str],
                              translations: List[str],
                              reference_translations: Optional[List[str]] = None,
                              document_ids: Optional[List[str]] = None) -> str:
        """
        Create an evaluation form for human evaluators.
        
        Args:
            source_texts: List of source texts
            translations: List of machine translations to evaluate
            reference_translations: Optional list of reference translations
            document_ids: Optional list of document identifiers
            
        Returns:
            Path to the created evaluation form
        """
        if len(source_texts) != len(translations):
            logger.error("Mismatched lengths for source_texts and translations")
            return ""
        
        if reference_translations and len(source_texts) != len(reference_translations):
            logger.error("Mismatched lengths for source_texts and reference_translations")
            return ""
        
        if document_ids and len(source_texts) != len(document_ids):
            logger.error("Mismatched lengths for source_texts and document_ids")
            return ""
        
        # Create evaluation data
        evaluation_data = []
        for i in range(len(source_texts)):
            item = {
                'id': i,
                'document_id': document_ids[i] if document_ids else f"doc_{i}",
                'source_text': source_texts[i],
                'machine_translation': translations[i]
            }
            
            if reference_translations:
                item['reference_translation'] = reference_translations[i]
            
            evaluation_data.append(item)
        
        # Create form metadata
        form_data = {
            'created_at': datetime.now().isoformat(),
            'evaluation_aspects': self.evaluation_aspects,
            'rating_scale': {
                '1': 'Poor',
                '2': 'Fair',
                '3': 'Good',
                '4': 'Very Good',
                '5': 'Excellent'
            },
            'items': evaluation_data
        }
        
        # Save form to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        form_path = os.path.join(self.feedback_dir, f"evaluation_form_{timestamp}.json")
        
        with open(form_path, 'w', encoding='utf-8') as f:
            json.dump(form_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation form created at {form_path}")
        return form_path
    
    def collect_feedback(self, 
                        evaluator_id: str,
                        form_path: str,
                        ratings: List[Dict[str, int]],
                        comments: Optional[List[str]] = None) -> str:
        """
        Collect feedback from a human evaluator.
        
        Args:
            evaluator_id: Identifier for the evaluator
            form_path: Path to the evaluation form
            ratings: List of dictionaries with ratings for each aspect
            comments: Optional list of comments for each item
            
        Returns:
            Path to the saved feedback file
        """
        # Load the form
        try:
            with open(form_path, 'r', encoding='utf-8') as f:
                form_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading form from {form_path}: {str(e)}")
            return ""
        
        # Validate ratings
        if len(ratings) != len(form_data['items']):
            logger.error(f"Mismatched number of ratings ({len(ratings)}) and form items ({len(form_data['items'])})")
            return ""
        
        # Validate aspects in ratings
        for i, rating_dict in enumerate(ratings):
            for aspect in rating_dict:
                if aspect not in form_data['evaluation_aspects']:
                    logger.error(f"Invalid aspect '{aspect}' in ratings for item {i}")
                    return ""
        
        # Create feedback data
        feedback_data = {
            'evaluator_id': evaluator_id,
            'form_path': form_path,
            'submission_time': datetime.now().isoformat(),
            'feedback': []
        }
        
        for i, item in enumerate(form_data['items']):
            feedback_item = {
                'id': item['id'],
                'document_id': item['document_id'],
                'ratings': ratings[i],
                'comment': comments[i] if comments and i < len(comments) else ""
            }
            feedback_data['feedback'].append(feedback_item)
        
        # Save feedback to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feedback_path = os.path.join(self.feedback_dir, f"feedback_{evaluator_id}_{timestamp}.json")
        
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Feedback from {evaluator_id} saved to {feedback_path}")
        return feedback_path
    
    def analyze_feedback(self, feedback_paths: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Analyze collected feedback.
        
        Args:
            feedback_paths: List of paths to feedback files
            
        Returns:
            Dictionary with analysis results
        """
        if not feedback_paths:
            logger.warning("No feedback files provided for analysis")
            return {}
        
        # Load all feedback
        all_feedback = []
        for path in feedback_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    all_feedback.append(feedback_data)
            except Exception as e:
                logger.error(f"Error loading feedback from {path}: {str(e)}")
        
        if not all_feedback:
            logger.warning("No valid feedback loaded")
            return {}
        
        # Aggregate ratings by aspect
        aspect_ratings = {}
        document_ratings = {}
        
        for feedback in all_feedback:
            evaluator_id = feedback['evaluator_id']
            
            for item in feedback['feedback']:
                doc_id = item['document_id']
                
                if doc_id not in document_ratings:
                    document_ratings[doc_id] = {}
                
                for aspect, rating in item['ratings'].items():
                    # Aggregate by aspect
                    if aspect not in aspect_ratings:
                        aspect_ratings[aspect] = []
                    aspect_ratings[aspect].append(rating)
                    
                    # Aggregate by document
                    if aspect not in document_ratings[doc_id]:
                        document_ratings[doc_id][aspect] = []
                    document_ratings[doc_id][aspect].append(rating)
        
        # Calculate average ratings
        results = {
            'overall': {},
            'by_document': {},
            'evaluator_count': len(all_feedback)
        }
        
        # Overall averages by aspect
        for aspect, ratings in aspect_ratings.items():
            results['overall'][aspect] = sum(ratings) / len(ratings)
        
        # Averages by document
        for doc_id, aspects in document_ratings.items():
            results['by_document'][doc_id] = {}
            for aspect, ratings in aspects.items():
                results['by_document'][doc_id][aspect] = sum(ratings) / len(ratings)
        
        logger.info(f"Feedback analysis completed for {len(all_feedback)} evaluators")
        return results
    
    def export_analysis(self, 
                       analysis_results: Dict[str, Union[float, Dict[str, float]]],
                       output_path: str) -> None:
        """
        Export analysis results to file.
        
        Args:
            analysis_results: Dictionary with analysis results
            output_path: Path to save results
        """
        # Convert to DataFrame for easier export
        
        # Overall results
        overall_df = pd.DataFrame({
            'aspect': list(analysis_results['overall'].keys()),
            'average_rating': list(analysis_results['overall'].values())
        })
        
        # Document results
        doc_rows = []
        for doc_id, aspects in analysis_results['by_document'].items():
            for aspect, rating in aspects.items():
                doc_rows.append({
                    'document_id': doc_id,
                    'aspect': aspect,
                    'average_rating': rating
                })
        
        doc_df = pd.DataFrame(doc_rows)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_path) as writer:
            overall_df.to_excel(writer, sheet_name='Overall', index=False)
            doc_df.to_excel(writer, sheet_name='By Document', index=False)
            
            # Add metadata
            meta_df = pd.DataFrame({
                'key': ['evaluator_count'],
                'value': [analysis_results['evaluator_count']]
            })
            meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Analysis results exported to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    evaluator = HumanEvaluator()
    
    # Create sample evaluation form
    source_texts = [
        "The central bank increased interest rates to combat inflation.",
        "Economic growth slowed to 2.5% in the last quarter."
    ]
    
    translations = [
        "رفع البنك المركزي أسعار الفائدة لمكافحة التضخم.",
        "تباطأ النمو الاقتصادي إلى 2.5٪ في الربع الأخير."
    ]
    
    form_path = evaluator.create_evaluation_form(source_texts, translations)
    
    # Simulate feedback collection
    ratings = [
        {'accuracy': 4, 'fluency': 5, 'terminology': 4, 'style': 3},
        {'accuracy': 5, 'fluency': 4, 'terminology': 5, 'style': 4}
    ]
    
    comments = [
        "Good translation but could improve style",
        "Excellent technical accuracy"
    ]
    
    feedback_path = evaluator.collect_feedback("evaluator1", form_path, ratings, comments)
    
    # Analyze feedback
    analysis = evaluator.analyze_feedback([feedback_path])
    print(f"Analysis results: {analysis}") 