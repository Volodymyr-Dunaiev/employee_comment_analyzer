# Main Inference Pipeline
#
# This module provides the main inference pipeline for classifying comments in batch.
# It handles:
# - Loading and processing Excel files in chunks
# - Running predictions using the CommentClassifier
# - Progress tracking and error handling
# - Saving results back to Excel

from typing import List, Optional, Callable, Union, TYPE_CHECKING
from io import BytesIO
import pandas as pd
from src.io.excel_io import read_excel_in_chunks, write_excel
from src.utils.logger import get_logger
from src.utils.config import load_config, ConfigError
from src.core.errors import PipelineError

if TYPE_CHECKING:
    from src.core.classifier import CommentClassifier

logger = get_logger(__name__)

# Lazy-load classifier to avoid loading model and heavy imports at module import time
_classifier: Optional['CommentClassifier'] = None

def _get_classifier() -> 'CommentClassifier':
    """Lazy-load the classifier instance and import CommentClassifier only when needed."""
    global _classifier
    if _classifier is None:
        from src.core.classifier import CommentClassifier  # Import only when actually needed
        config = load_config()
        _classifier = CommentClassifier(config)
    return _classifier

def run_inference(
    input_path: Union[str, BytesIO],
    output_path: Union[str, BytesIO],
    text_column: str,
    categories: List[str],
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> None:
    """Run the complete inference pipeline on an input file.
    
    This is the main entry point for batch classification. It:
    1. Loads configuration
    2. Processes file in chunks for memory efficiency
    3. Runs predictions on each chunk
    4. Combines results and saves to output file
    
    Args:
        input_path: Path to input Excel file (or BytesIO object)
        output_path: Path to output Excel file (or BytesIO object)
        text_column: Name of the column containing text to classify
        categories: List of possible category labels
        progress_callback: Optional function(current, total) for progress updates
    
    Raises:
        PipelineError: If any step fails (file not found, invalid format, etc.)
        ConfigError: If configuration is invalid
    
    Example:
        >>> run_inference(
        ...     "input.xlsx",
        ...     "output.xlsx", 
        ...     "comment",
        ...     ["Category1", "Category2"],
        ...     lambda curr, tot: print(f"{curr}/{tot}")
        ... )
    """
    
    logger.info("Starting inference pipeline...")
    
    try:
        config = load_config()
        chunk_size = config['data']['chunk_size']
        classifier = _get_classifier()  # Lazy-load classifier
        
        # Process in chunks to handle large files
        all_results = []
        for df_chunk in read_excel_in_chunks(input_path, chunk_size=chunk_size):
            if text_column not in df_chunk.columns:
                raise PipelineError(f"Required column '{text_column}' not found in input file")
                
            df_chunk["Predicted_Categories"] = classifier.predict_comments(
                df_chunk, text_column, categories, progress_callback
            )
            all_results.append(df_chunk)
            
        # Combine results and save
        final_df = pd.concat(all_results, ignore_index=True)
        write_excel(final_df, output_path)
        logger.info("Inference pipeline completed successfully")
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        raise PipelineError(error_msg)
