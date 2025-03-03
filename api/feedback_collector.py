"""
Feedback collection module for translation service.
Implements tools for collecting and analyzing user feedback.
"""

import os
import logging
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import yaml
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class FeedbackCollector:
    """Collects and analyzes user feedback on translations."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the feedback collector.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feedback_dir = "feedback"
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # Define error categories
        self.error_categories = [
            "terminology",
            "grammar",
            "style",
            "omission",
            "addition",
            "word_order",
            "punctuation",
            "number"
        ]
    
    def save_feedback(self, feedback_data: Dict[str, any]) -> str:
        """
        Save feedback data to storage.
        
        Args:
            feedback_data: Feedback data to save
            
        Returns:
            Path to saved feedback file
        """
        try:
            # Generate feedback ID if not provided
            if 'feedback_id' not in feedback_data:
                feedback_data['feedback_id'] = f"feedback_{int(time.time())}"
            
            # Add timestamp if not provided
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = time.time()
            
            # Save feedback to JSON file
            feedback_path = os.path.join(self.feedback_dir, f"{feedback_data['feedback_id']}.json")
            
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Feedback saved to {feedback_path}")
            return feedback_path
        
        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}")
            return ""
    
    def load_feedback(self, feedback_id: str) -> Dict[str, any]:
        """
        Load feedback data from storage.
        
        Args:
            feedback_id: ID of the feedback to load
            
        Returns:
            Feedback data
        """
        try:
            feedback_path = os.path.join(self.feedback_dir, f"{feedback_id}.json")
            
            if not os.path.exists(feedback_path):
                logger.error(f"Feedback file not found: {feedback_path}")
                return {}
            
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            return feedback_data
        
        except Exception as e:
            logger.error(f"Error loading feedback: {str(e)}")
            return {}
    
    def list_feedback(self, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     min_rating: Optional[int] = None,
                     max_rating: Optional[int] = None,
                     error_category: Optional[str] = None) -> List[Dict[str, any]]:
        """
        List feedback matching the specified criteria.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            min_rating: Minimum rating for filtering
            max_rating: Maximum rating for filtering
            error_category: Error category for filtering
            
        Returns:
            List of feedback data matching the criteria
        """
        try:
            all_feedback = []
            
            # Load all feedback files
            for filename in os.listdir(self.feedback_dir):
                if filename.endswith('.json'):
                    feedback_path = os.path.join(self.feedback_dir, filename)
                    
                    with open(feedback_path, 'r', encoding='utf-8') as f:
                        feedback_data = json.load(f)
                        all_feedback.append(feedback_data)
            
            # Apply filters
            filtered_feedback = all_feedback
            
            if start_date:
                start_timestamp = start_date.timestamp()
                filtered_feedback = [f for f in filtered_feedback 
                                    if f.get('timestamp', 0) >= start_timestamp]
            
            if end_date:
                end_timestamp = end_date.timestamp()
                filtered_feedback = [f for f in filtered_feedback 
                                    if f.get('timestamp', 0) <= end_timestamp]
            
            if min_rating is not None:
                filtered_feedback = [f for f in filtered_feedback 
                                    if f.get('rating', 0) >= min_rating]
            
            if max_rating is not None:
                filtered_feedback = [f for f in filtered_feedback 
                                    if f.get('rating', 0) <= max_rating]
            
            if error_category:
                filtered_feedback = [f for f in filtered_feedback 
                                    if error_category in f.get('error_tags', [])]
            
            return filtered_feedback
        
        except Exception as e:
            logger.error(f"Error listing feedback: {str(e)}")
            return []
    
    def analyze_feedback(self, feedback_list: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Analyze feedback data to extract insights.
        
        Args:
            feedback_list: List of feedback data to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if not feedback_list:
                return {
                    'count': 0,
                    'average_rating': 0,
                    'error_categories': {},
                    'comments_count': 0,
                    'improvement_suggestions': []
                }
            
            # Calculate statistics
            ratings = [f.get('rating', 0) for f in feedback_list if f.get('rating') is not None]
            average_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Count error categories
            error_counts = {}
            for category in self.error_categories:
                count = sum(1 for f in feedback_list if category in f.get('error_tags', []))
                if count > 0:
                    error_counts[category] = count
            
            # Extract improvement suggestions from comments
            comments = [f.get('comments', '') for f in feedback_list if f.get('comments')]
            
            # In a real system, you would use NLP to extract suggestions
            # For this POC, we'll just count comments as suggestions
            
            analysis = {
                'count': len(feedback_list),
                'average_rating': average_rating,
                'error_categories': error_counts,
                'comments_count': len(comments)
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing feedback: {str(e)}")
            return {
                'count': 0,
                'average_rating': 0,
                'error_categories': {},
                'comments_count': 0,
                'error': str(e)
            }
    
    def export_feedback(self, 
                       feedback_list: List[Dict[str, any]],
                       output_path: str) -> bool:
        """
        Export feedback data to file.
        
        Args:
            feedback_list: List of feedback data to export
            output_path: Path to save exported data
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Convert to DataFrame for easier export
            rows = []
            
            for feedback in feedback_list:
                row = {
                    'feedback_id': feedback.get('feedback_id', ''),
                    'translation_id': feedback.get('translation_id', ''),
                    'source_text': feedback.get('source_text', ''),
                    'translation': feedback.get('translation', ''),
                    'corrected_translation': feedback.get('corrected_translation', ''),
                    'rating': feedback.get('rating', ''),
                    'comments': feedback.get('comments', ''),
                    'error_tags': ', '.join(feedback.get('error_tags', [])),
                    'timestamp': datetime.fromtimestamp(feedback.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')
                }
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Determine export format based on file extension
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                df.to_excel(output_path, index=False)
            elif output_path.endswith('.json'):
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(feedback_list, f, ensure_ascii=False, indent=2)
            else:
                logger.error(f"Unsupported export format: {output_path}")
                return False
            
            logger.info(f"Feedback exported to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting feedback: {str(e)}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    collector = FeedbackCollector()
    
    # Create sample feedback
    sample_feedback = {
        'translation_id': 'trans_123456',
        'source_text': 'The central bank increased interest rates to combat inflation.',
        'translation': 'رفع البنك المركزي أسعار الفائدة لمكافحة التضخم.',
        'corrected_translation': 'قام البنك المركزي برفع أسعار الفائدة لمكافحة التضخم.',
        'rating': 4,
        'comments': 'Good translation but could use more formal language.',
        'error_tags': ['style']
    }
    
    # Save feedback
    feedback_path = collector.save_feedback(sample_feedback)
    
    # List and analyze feedback
    all_feedback = collector.list_feedback()
    analysis = collector.analyze_feedback(all_feedback)
    
    print(f"Feedback analysis: {analysis}")
    
    # Export feedback
    collector.export_feedback(all_feedback, "feedback_export.csv") 