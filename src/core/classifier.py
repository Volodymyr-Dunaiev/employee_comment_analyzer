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
from src.core.skip_reasons import SkipReason
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
    
    def classify_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Classify a batch of texts and return probabilities for each category.
        
        This method is designed for batch processing workflows where we need
        probability scores for all categories rather than just predicted labels.
        
        Args:
            texts: List of input texts to classify
        
        Returns:
            Dictionary containing:
            - Category names mapped to lists of probabilities (one per text)
            - 'skip_reason': List[str] with reason for each row (none/empty/whitespace/nan/non_text)
            - '_metadata': Dict with 'skipped_indices' and 'skip_reason_counts'
            
        Example:
            >>> result = classifier.classify_batch(["Good team", "", "Bad management"])
            >>> result['Колектив']  # [0.85, 0.0, 0.12]
            >>> result['skip_reason']  # ["none", "empty", "none"]
            >>> result['_metadata']['skip_reason_counts']  # {"empty": 1, "none": 2}
        """
        if not texts:
            return {'_metadata': {'skipped_indices': [], 'skip_reason_counts': {}}}
        
        # Get categories from config
        categories = self.config.get('categories', [])
        if not categories:
            raise PipelineError("No categories defined in configuration")
        
        # Determine skip reason for each text (no length restriction)
        skip_reasons = [SkipReason.from_text(text) for text in texts]
        
        # Filter valid texts
        valid_texts = []
        valid_indices = []
        skipped_indices = []
        skip_reason_counts = {}
        
        for i, (text, reason) in enumerate(zip(texts, skip_reasons)):
            # Count all reasons
            reason_str = str(reason)
            skip_reason_counts[reason_str] = skip_reason_counts.get(reason_str, 0) + 1
            
            if reason.should_skip():
                skipped_indices.append(i)
            else:
                valid_texts.append(str(text))  # Convert to string for safety
                valid_indices.append(i)
        
        # Log warning if any texts were skipped
        # For large batches, only log count to avoid log flooding
        if skipped_indices:
            reason_summary = ', '.join(f"{k}={v}" for k, v in sorted(skip_reason_counts.items()) if k != 'none')
            logger.warning(
                f"Skipped {len(skipped_indices)} texts (out of {len(texts)} total): {reason_summary}"
            )
        
        # If all texts are invalid, return zeros for all categories with metadata
        if not valid_texts:
            result = {category: [0.0] * len(texts) for category in categories}
            result['skip_reason'] = [str(r) for r in skip_reasons]
            result['_metadata'] = {
                'skipped_indices': skipped_indices,
                'skip_reason_counts': skip_reason_counts
            }
            logger.warning(f"All {len(texts)} texts were invalid - returning zero probabilities")
            return result
        
        # Tokenize and run inference on valid texts only
        inputs = self.tokenizer(valid_texts, truncation=True, padding=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
        
        # Reconstruct full results with zeros for skipped texts
        result = {}
        for i, category in enumerate(categories):
            # Handle case where model has fewer outputs than categories
            if i < probs.shape[1]:
                # Create array with zeros for all positions
                full_probs = [0.0] * len(texts)
                # Fill in predictions for valid texts
                for valid_idx, orig_idx in enumerate(valid_indices):
                    full_probs[orig_idx] = float(probs[valid_idx, i])
                result[category] = full_probs
            else:
                # Category not in model output, use zeros
                result[category] = [0.0] * len(texts)
        
        # Add skip_reason column for operator triage
        result['skip_reason'] = [str(r) for r in skip_reasons]
        
        # Add metadata about skipped rows
        result['_metadata'] = {
            'skipped_indices': skipped_indices,
            'skip_reason_counts': skip_reason_counts
        }
        
        return result
        
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