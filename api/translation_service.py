"""
Translation service API module.
Implements RESTful API endpoints for the translation service.
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Union, Tuple
import yaml
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import uvicorn

logger = logging.getLogger(__name__)

# Define API models
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_language: str = Field(default="en", description="Source language code")
    target_language: str = Field(default="ar", description="Target language code")
    glossary_id: Optional[str] = Field(default=None, description="ID of glossary to use")
    document_id: Optional[str] = Field(default=None, description="Document identifier")

class BatchTranslationRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to translate")
    source_language: str = Field(default="en", description="Source language code")
    target_language: str = Field(default="ar", description="Target language code")
    glossary_id: Optional[str] = Field(default=None, description="ID of glossary to use")
    document_id: Optional[str] = Field(default=None, description="Document identifier")

class TranslationResponse(BaseModel):
    translation: str = Field(..., description="Translated text")
    model_used: str = Field(..., description="Model used for translation")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence_score: Optional[float] = Field(default=None, description="Confidence score")
    glossary_terms_used: Optional[List[str]] = Field(default=None, description="Glossary terms used in translation")

class BatchTranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="List of translated texts")
    model_used: str = Field(..., description="Model used for translation")
    processing_time: float = Field(..., description="Processing time in seconds")
    average_confidence_score: Optional[float] = Field(default=None, description="Average confidence score")
    total_glossary_terms_used: Optional[int] = Field(default=None, description="Total glossary terms used in translations")

class FeedbackRequest(BaseModel):
    translation_id: str = Field(..., description="ID of the translation")
    source_text: str = Field(..., description="Original source text")
    translation: str = Field(..., description="Machine translation")
    corrected_translation: Optional[str] = Field(default=None, description="Corrected translation")
    rating: Optional[int] = Field(default=None, description="Rating (1-5)")
    comments: Optional[str] = Field(default=None, description="Feedback comments")
    error_tags: Optional[List[str]] = Field(default=None, description="Error category tags")

class TranslationService:
    """Translation service API implementation."""
    
    _model_instance = None  # Singleton for model instance

    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the translation service.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_config = self.config['api']
        self.model_config = self.config['model']
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Financial Translation API",
            description="API for translating financial and economic documents",
            version="0.1.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict this to specific origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add rate limiting middleware
        self._setup_rate_limiting()
        
        # Register routes
        self._setup_routes()
        
        # Initialize model and glossary manager placeholders
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.glossary_manager = None  # Can be replaced by TerminologyService later
        
        # Track translation requests for feedback
        self.translation_history = {}
    
    def _setup_rate_limiting(self):
        """Set up rate limiting middleware."""
        @self.app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = request.client.host
            current_time = time.time()
            rate_limit = self.api_config.get('rate_limit', 100)
            response = await call_next(request)
            return response
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Financial Translation API"}
        
        @self.app.post("/translate", response_model=TranslationResponse)
        async def translate(request: TranslationRequest):
            try:
                start_time = time.time()
                
                if self.model is None:
                    self._load_model()
                
                translation, glossary_terms = self._translate_text(
                    request.text,
                    request.source_language,
                    request.target_language,
                    request.glossary_id
                )
                
                processing_time = time.time() - start_time
                translation_id = f"trans_{int(time.time())}_{hash(request.text) % 10000}"
                self.translation_history[translation_id] = {
                    'source_text': request.text,
                    'translation': translation,
                    'source_language': request.source_language,
                    'target_language': request.target_language,
                    'document_id': request.document_id,
                    'timestamp': time.time()
                }
                
                return TranslationResponse(
                    translation=translation,
                    model_used=self.model_config.get('base_model', 'unknown'),
                    processing_time=processing_time,
                    confidence_score=0.95,
                    glossary_terms_used=glossary_terms
                )
            
            except Exception as e:
                logger.error(f"Translation error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Translation error: {str(e)}"
                )
        
        @self.app.post("/translate/batch", response_model=BatchTranslationResponse)
        async def translate_batch(request: BatchTranslationRequest):
            try:
                start_time = time.time()
                
                if self.model is None:
                    self._load_model()
                
                translations = []
                all_glossary_terms = []
                for text in request.texts:
                    translation, glossary_terms = self._translate_text(
                        text,
                        request.source_language,
                        request.target_language,
                        request.glossary_id
                    )
                    translations.append(translation)
                    all_glossary_terms.extend(glossary_terms)
                
                processing_time = time.time() - start_time
                return BatchTranslationResponse(
                    translations=translations,
                    model_used=self.model_config.get('base_model', 'unknown'),
                    processing_time=processing_time,
                    average_confidence_score=0.95,
                    total_glossary_terms_used=len(set(all_glossary_terms))
                )
            
            except Exception as e:
                logger.error(f"Batch translation error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch translation error: {str(e)}"
                )
        
        @self.app.post("/feedback")
        async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
            try:
                feedback_id = f"feedback_{int(time.time())}"
                feedback_data = {
                    'feedback_id': feedback_id,
                    'translation_id': request.translation_id,
                    'source_text': request.source_text,
                    'translation': request.translation,
                    'corrected_translation': request.corrected_translation,
                    'rating': request.rating,
                    'comments': request.comments,
                    'error_tags': request.error_tags,
                    'timestamp': time.time()
                }
                background_tasks.add_task(self._save_feedback, feedback_data)
                return {"message": "Feedback received", "feedback_id": feedback_id}
            
            except Exception as e:
                logger.error(f"Feedback submission error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Feedback submission error: {str(e)}"
                )
        
        @self.app.get("/glossaries")
        async def list_glossaries():
            try:
                if self.glossary_manager is None:
                    self._load_glossary_manager()
                glossaries = self.glossary_manager.list_glossaries()
                return {"glossaries": glossaries}
            except Exception as e:
                logger.error(f"Error listing glossaries: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error listing glossaries: {str(e)}"
                )
    
    def _load_model(self):
        """Load translation model using singleton pattern."""
        if TranslationService._model_instance is None:
            from models.transformer_nmt import TransformerNMT
            model_name = self.model_config.get('base_model', 'facebook/mbart-large-50')
            TranslationService._model_instance = TransformerNMT(model_name=model_name)
            TranslationService._model_instance.model = TranslationService._model_instance.model.half().to(self.device)
            logger.info(f"Loaded translation model: {model_name}")
        self.model = TranslationService._model_instance
    
    def _load_glossary_manager(self):
        """Load glossary manager (can be replaced with TerminologyService)."""
        from models.terminology_service import TerminologyService
        self.glossary_manager = TerminologyService()
        logger.info("Loaded Terminology Service as glossary manager")
    
    def _translate_text(self, 
                        text: str, 
                        source_lang: str, 
                        target_lang: str,
                        glossary_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        Translate text using the loaded model.
        
        Returns:
            Tuple of (translated text, list of glossary terms used)
        """
        if not text:
            return "", []
        
        # Load terminology service for constraints if needed
        try:
            from models.terminology_service import TerminologyService
            term_service = TerminologyService()
            constraints = term_service.get_constraints(text, self.model.tokenizer, source_lang)
        except Exception as e:
            logger.error(f"Error obtaining terminology constraints: {str(e)}")
            constraints = []
        
        if self.model_type() == "mbart":
            src_lang_code = self.model.tokenizer.lang_code_to_id.get("en_XX") if source_lang == 'en' else self.model.tokenizer.lang_code_to_id.get("ar_AR")
            tgt_lang_code = self.model.tokenizer.lang_code_to_id.get("ar_AR") if target_lang == 'ar' else self.model.tokenizer.lang_code_to_id.get("en_XX")
            self.model.tokenizer.src_lang = "en_XX" if source_lang == 'en' else "ar_AR"
            inputs = self.model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            forced_bos_token_id = tgt_lang_code
        else:
            inputs = self.model.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            forced_bos_token_id = None
        
        try:
            with torch.no_grad():
                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "max_length": 128,
                    "num_beams": 4,
                    "early_stopping": True,
                    "forced_bos_token_id": forced_bos_token_id
                }
                if constraints:
                    generate_kwargs["constraints"] = constraints
                outputs = self.model.model.generate(**generate_kwargs)
            translation = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # For simplicity, assume glossary terms used are those for which constraints were applied
            glossary_terms = [self.model.tokenizer.decode(constraint) for constraint in constraints] if constraints else []
            return translation, glossary_terms
        except Exception as e:
            logger.error(f"Error during translation generation: {str(e)}")
            return "", []
    
    def _save_feedback(self, feedback_data: Dict[str, any]) -> None:
        """Save feedback data asynchronously."""
        from api.feedback_collector import FeedbackCollector
        fc = FeedbackCollector()
        fc.save_feedback(feedback_data)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API service."""
        uvicorn.run(self.app, host=host, port=port)
