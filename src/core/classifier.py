# Comment Classifier Module
#
# This module provides the CommentClassifier class for multi-label classification
# of Ukrainian text comments. It handles model loading, input validation, and 
# batch prediction with progress tracking.

import torch
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from transformers import PreTrainedTokenizer, PreTrainedModel
from src.utils.logger import get_logger
from src.core.errors import PipelineError
from src.core.model_utils import load_model_and_tokenizer
from src.utils.memory import auto_tune_batch_size, check_memory_sufficient

logger = get_logger(__name__)


class CommentClassifier:
    # Main classifier class for handling multi-label comment classification.
    #
    # This class manages the entire classification pipeline including:
    # - Model and tokenizer initialization
    # - Input data validation
    # - Batch prediction with progress tracking
    # - Multi-label output formatting
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the classifier with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - model.path: Path to model or Hugging Face model ID
                - model.device: Device to run on ('cpu' or 'cuda')
                - model.batch_size: Batch size for predictions
                - data.input_validation: Validation settings
        
        Raises:
            KeyError: If required config keys are missing
            ValueError: If model cannot be loaded
        """
        
        self.config: Dict[str, Any] = config
        self.device: torch.device = torch.device(config['model']['device'])
        
        # Check memory and auto-tune batch size
        if not check_memory_sufficient(2.0):
            logger.warning("Insufficient memory detected. Performance may be degraded.")
        
        # Auto-tune batch size based on available memory
        default_batch_size = config['model'].get('batch_size', 16)
        tuned_batch_size = auto_tune_batch_size(default_batch_size)
        config['model']['batch_size'] = tuned_batch_size
        
        self.tokenizer: PreTrainedTokenizer
        self.model: PreTrainedModel
        self.tokenizer, self.model = load_model_and_tokenizer(config['model']['path'])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device} (batch_size={tuned_batch_size})")

    def validate_input_data(self, df: pd.DataFrame, text_column: str) -> None:
        """Validate input DataFrame for required columns and constraints.
        
        Args:
            df: Input pandas DataFrame
            text_column: Name of column containing text data
        
        Raises:
            PipelineError: If validation fails (missing column, null values, text too long)
        """
        
        if text_column not in df.columns:
            raise PipelineError(f"Required column '{text_column}' not found in input data")
        if len(df) == 0:
            return  # Empty DataFrame is valid
        if df[text_column].isnull().any():
            raise PipelineError("Input data contains null values in the text column")
        if df[text_column].astype(str).str.len().max() > self.config['data']['input_validation']['max_length']:
            raise PipelineError("Input data contains text exceeding maximum allowed length")

    def predict_batch(
        self,
        texts: List[str],
        categories: List[str]
    ) -> List[List[str]]:
        """Predict categories for a batch of texts.
        
        Args:
            texts: List of input texts
            categories: List of category labels
        
        Returns:
            List of predicted categories for each text
        """
        
        inputs = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
        
        preds = (probs >= 0.5).astype(int)
        return [
            [categories[j] for j, val in enumerate(pred_row) if val == 1]
            for pred_row in preds
        ]
        
    def predict_comments(
        self,
        df: pd.DataFrame,
        text_column: str,
        categories: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """Predict categories for comments in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            categories: List of category labels
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of predicted categories for each comment
        """
        
        self.validate_input_data(df, text_column)
        texts = df[text_column].tolist()
        batch_size = self.config['model']['batch_size']
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_preds = self.predict_batch(batch_texts, categories)
            results.extend(batch_preds)
            
            if progress_callback:
                progress_callback(min(i + batch_size, len(texts)), len(texts))
        
        return results
    # Implementation here
        pass