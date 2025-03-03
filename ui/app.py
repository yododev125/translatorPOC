"""
User interface module for translation system.
Implements a web interface for interacting with the translation service.
"""

import os
import logging
import json
import requests
from typing import Dict, List, Optional
import yaml
from datetime import datetime
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session

logger = logging.getLogger(__name__)

class TranslationUI:
    """Web interface for the translation service."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the translation UI.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                         template_folder='templates',
                         static_folder='static')
        self.app.secret_key = os.urandom(24)
        
        # API configuration
        self.api_host = self.config['api'].get('host', '0.0.0.0')
        self.api_port = self.config['api'].get('port', 8000)
        self.api_url = f"http://{self.api_host}:{self.api_port}"
        
        # Register routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/translate', methods=['GET', 'POST'])
        def translate():
            if request.method == 'POST':
                # Get form data
                source_text = request.form.get('source_text', '')
                source_lang = request.form.get('source_language', 'en')
                target_lang = request.form.get('target_language', 'ar')
                glossary_id = request.form.get('glossary_id')
                
                if not source_text:
                    flash('Please enter text to translate', 'error')
                    return render_template('translate.html')
                
                try:
                    # Call translation API
                    translation_result = self._call_translation_api(
                        source_text, source_lang, target_lang, glossary_id
                    )
                    
                    # Store in session for feedback
                    session['last_translation'] = {
                        'source_text': source_text,
                        'translation': translation_result.get('translation', ''),
                        'model_used': translation_result.get('model_used', ''),
                        'processing_time': translation_result.get('processing_time', 0),
                        'glossary_terms_used': translation_result.get('glossary_terms_used', [])
                    }
                    
                    return render_template(
                        'translate.html',
                        source_text=source_text,
                        translation=translation_result.get('translation', ''),
                        source_language=source_lang,
                        target_language=target_lang,
                        model_used=translation_result.get('model_used', ''),
                        processing_time=translation_result.get('processing_time', 0),
                        glossary_terms_used=translation_result.get('glossary_terms_used', [])
                    )
                
                except Exception as e:
                    logger.error(f"Translation error: {str(e)}")
                    flash(f"Translation error: {str(e)}", 'error')
            
            # GET request or error in POST
            return render_template('translate.html')
        
        @self.app.route('/feedback', methods=['GET', 'POST'])
        def feedback():
            if request.method == 'POST':
                # Get form data
                translation_id = request.form.get('translation_id', '')
                source_text = request.form.get('source_text', '')
                translation = request.form.get('translation', '')
                corrected_translation = request.form.get('corrected_translation', '')
                rating = request.form.get('rating')
                comments = request.form.get('comments', '')
                error_tags = request.form.getlist('error_tags')
                
                try:
                    # Convert rating to integer
                    rating = int(rating) if rating else None
                    
                    # Call feedback API
                    feedback_result = self._submit_feedback(
                        translation_id, source_text, translation,
                        corrected_translation, rating, comments, error_tags
                    )
                    
                    flash('Thank you for your feedback!', 'success')
                    return redirect(url_for('index'))
                
                except Exception as e:
                    logger.error(f"Feedback submission error: {str(e)}")
                    flash(f"Feedback submission error: {str(e)}", 'error')
            
            # GET request or error in POST
            last_translation = session.get('last_translation', {})
            
            return render_template(
                'feedback.html',
                source_text=last_translation.get('source_text', ''),
                translation=last_translation.get('translation', ''),
                error_categories=self._get_error_categories()
            )
        
        @self.app.route('/batch', methods=['GET', 'POST'])
        def batch_translate():
            if request.method == 'POST':
                # Handle file upload
                if 'file' not in request.files:
                    flash('No file part', 'error')
                    return redirect(request.url)
                
                file = request.files['file']
                
                if file.filename == '':
                    flash('No selected file', 'error')
                    return redirect(request.url)
                
                if file:
                    try:
                        # Read file content
                        content = file.read().decode('utf-8')
                        lines = content.strip().split('\n')
                        
                        source_lang = request.form.get('source_language', 'en')
                        target_lang = request.form.get('target_language', 'ar')
                        glossary_id = request.form.get('glossary_id')
                        
                        # Call batch translation API
                        translation_result = self._call_batch_translation_api(
                            lines, source_lang, target_lang, glossary_id
                        )
                        
                        translations = translation_result.get('translations', [])
                        
                        # Create result for download
                        result_lines = []
                        for i, (source, translation) in enumerate(zip(lines, translations)):
                            result_lines.append(f"Source {i+1}: {source}")
                            result_lines.append(f"Translation {i+1}: {translation}")
                            result_lines.append("")
                        
                        result_text = '\n'.join(result_lines)
                        
                        return render_template(
                            'batch.html',
                            result_text=result_text,
                            source_language=source_lang,
                            target_language=target_lang,
                            model_used=translation_result.get('model_used', ''),
                            processing_time=translation_result.get('processing_time', 0)
                        )
                    
                    except Exception as e:
                        logger.error(f"Batch translation error: {str(e)}")
                        flash(f"Batch translation error: {str(e)}", 'error')
            
            # GET request or error in POST
            return render_template('batch.html')
        
        @self.app.route('/glossary', methods=['GET', 'POST'])
        def glossary():
            if request.method == 'POST':
                # Handle glossary term addition
                english_term = request.form.get('english_term', '')
                arabic_term = request.form.get('arabic_term', '')
                category = request.form.get('category', '')
                
                if not english_term or not arabic_term:
                    flash('Please enter both English and Arabic terms', 'error')
                    return redirect(url_for('glossary'))
                
                try:
                    # Call glossary API to add term
                    result = self._add_glossary_term(english_term, arabic_term, category)
                    
                    flash('Term added to glossary', 'success')
                    return redirect(url_for('glossary'))
                
                except Exception as e:
                    logger.error(f"Error adding glossary term: {str(e)}")
                    flash(f"Error adding glossary term: {str(e)}", 'error')
            
            # GET request or error in POST
            try:
                # Get glossary terms
                glossary_terms = self._get_glossary_terms()
                
                return render_template(
                    'glossary.html',
                    glossary_terms=glossary_terms
                )
            
            except Exception as e:
                logger.error(f"Error getting glossary terms: {str(e)}")
                flash(f"Error getting glossary terms: {str(e)}", 'error')
                return render_template('glossary.html', glossary_terms=[])
    
    def _call_translation_api(self, 
                             text: str, 
                             source_lang: str, 
                             target_lang: str,
                             glossary_id: Optional[str] = None) -> Dict[str, any]:
        """
        Call translation API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            glossary_id: ID of glossary to use
            
        Returns:
            Translation result
        """
        url = f"{self.api_url}/translate"
        
        payload = {
            "text": text,
            "source_language": source_lang,
            "target_language": target_lang
        }
        
        if glossary_id:
            payload["glossary_id"] = glossary_id
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise Exception(f"API request error: {str(e)}")
    
    def _call_batch_translation_api(self, 
                                   texts: List[str], 
                                   source_lang: str, 
                                   target_lang: str,
                                   glossary_id: Optional[str] = None) -> Dict[str, any]:
        """
        Call batch translation API.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            glossary_id: ID of glossary to use
            
        Returns:
            Batch translation result
        """
        url = f"{self.api_url}/translate/batch"
        
        payload = {
            "texts": texts,
            "source_language": source_lang,
            "target_language": target_lang
        }
        
        if glossary_id:
            payload["glossary_id"] = glossary_id
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise Exception(f"API request error: {str(e)}")
    
    def _submit_feedback(self,
                        translation_id: str,
                        source_text: str,
                        translation: str,
                        corrected_translation: Optional[str] = None,
                        rating: Optional[int] = None,
                        comments: Optional[str] = None,
                        error_tags: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Submit feedback to API.
        
        Args:
            translation_id: ID of the translation
            source_text: Original source text
            translation: Machine translation
            corrected_translation: Corrected translation
            rating: Rating (1-5)
            comments: Feedback comments
            error_tags: Error category tags
            
        Returns:
            Feedback submission result
        """
        url = f"{self.api_url}/feedback"
        
        payload = {
            "translation_id": translation_id or f"ui_{int(time.time())}",
            "source_text": source_text,
            "translation": translation
        }
        
        if corrected_translation:
            payload["corrected_translation"] = corrected_translation
        
        if rating is not None:
            payload["rating"] = rating
        
        if comments:
            payload["comments"] = comments
        
        if error_tags:
            payload["error_tags"] = error_tags
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise Exception(f"API request error: {str(e)}")
    
    def _get_glossary_terms(self) -> List[Dict[str, str]]:
        """
        Get glossary terms from API.
        
        Returns:
            List of glossary terms
        """
        url = f"{self.api_url}/glossary/terms"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json().get('terms', [])
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise Exception(f"API request error: {str(e)}")
    
    def _add_glossary_term(self, 
                          english_term: str, 
                          arabic_term: str,
                          category: Optional[str] = None) -> Dict[str, any]:
        """
        Add term to glossary via API.
        
        Args:
            english_term: English term
            arabic_term: Arabic term
            category: Term category
            
        Returns:
            Result of adding term
        """
        url = f"{self.api_url}/glossary/terms"
        
        payload = {
            "english_term": english_term,
            "arabic_term": arabic_term
        }
        
        if category:
            payload["category"] = category
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            raise Exception(f"API request error: {str(e)}")
    
    def _get_error_categories(self) -> List[str]:
        """
        Get error categories for feedback form.
        
        Returns:
            List of error categories
        """
        return [
            "terminology",
            "grammar",
            "style",
            "omission",
            "addition",
            "word_order",
            "punctuation",
            "number"
        ]
    
    def run(self, host: str = "0.0.0.0", port: int = 5000):
        """
        Run the web interface.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        self.app.run(host=host, port=port, debug=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and run the UI
    ui = TranslationUI()
    ui.run() 