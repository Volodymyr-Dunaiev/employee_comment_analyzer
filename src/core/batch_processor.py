"""
Batch file processing module for handling multiple files concurrently.

Supports:
- Multiple Excel/CSV files
- Parallel processing with configurable workers
- Progress tracking
- Error handling per file
- Consolidated results
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from datetime import datetime
import logging

from src.core.classifier import CommentClassifier
from src.core.errors import PipelineError

logger = logging.getLogger(__name__)


class ClassificationError(PipelineError):
    """Exception raised during classification."""
    pass


class ExcelIO:
    """Helper class for reading/writing Excel files."""
    
    @staticmethod
    def read_file(file_path: Path, text_column: str = "text") -> pd.DataFrame:
        """Read Excel or CSV file."""
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        if text_column not in df.columns:
            raise ClassificationError(f"Column '{text_column}' not found in {file_path.name}")
        
        return df
    
    @staticmethod
    def save_results(df: pd.DataFrame, output_path: Path):
        """Save DataFrame to Excel file."""
        df.to_excel(output_path, index=False)

logger = logging.getLogger(__name__)


class BatchProcessingResult:
    """Container for batch processing results."""
    
    def __init__(self):
        self.successful_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []  # (filename, error_message)
        self.total_comments_processed: int = 0
        self.processing_time: float = 0.0
        self.results_by_file: Dict[str, pd.DataFrame] = {}
    
    def add_success(self, filename: str, df: pd.DataFrame, comments_count: int):
        """Record successful file processing."""
        self.successful_files.append(filename)
        self.results_by_file[filename] = df
        self.total_comments_processed += comments_count
    
    def add_failure(self, filename: str, error: str):
        """Record failed file processing."""
        self.failed_files.append((filename, error))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = len(self.successful_files) + len(self.failed_files)
        if total == 0:
            return 0.0
        return (len(self.successful_files) / total) * 100
    
    def get_summary(self) -> Dict:
        """Get processing summary."""
        return {
            "total_files": len(self.successful_files) + len(self.failed_files),
            "successful_files": len(self.successful_files),
            "failed_files": len(self.failed_files),
            "success_rate": f"{self.success_rate:.1f}%",
            "total_comments": self.total_comments_processed,
            "processing_time": f"{self.processing_time:.2f}s"
        }


class BatchProcessor:
    """Process multiple files in batch mode."""
    
    def __init__(
        self,
        classifier: CommentClassifier,
        max_workers: int = 3,
        text_column: str = "text"
    ):
        """
        Initialize batch processor.
        
        Args:
            classifier: CommentClassifier instance for classification
            max_workers: Maximum concurrent file processing (default: 3)
            text_column: Name of text column in input files
        """
        self.classifier = classifier
        self.max_workers = max_workers
        self.text_column = text_column
        self.excel_io = ExcelIO()
        
        logger.info(f"Batch processor initialized with {max_workers} workers")
    
    def process_files(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path] = None,
        output_prefix: str = "classified_",
        combine_results: bool = False
    ) -> BatchProcessingResult:
        """
        Process multiple files in batch.
        
        Args:
            file_paths: List of input file paths
            output_dir: Directory for output files (None = same as input)
            output_prefix: Prefix for output filenames
            combine_results: If True, create combined output file
        
        Returns:
            BatchProcessingResult with processing details
        """
        start_time = datetime.now()
        result = BatchProcessingResult()
        
        logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        # Process files concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._process_single_file,
                    file_path,
                    output_dir,
                    output_prefix
                ): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = file_path.name
                
                try:
                    output_path, df, comments_count = future.result()
                    result.add_success(filename, df, comments_count)
                    logger.info(f"✓ Processed {filename}: {comments_count} comments")
                    
                except Exception as e:
                    error_msg = str(e)
                    result.add_failure(filename, error_msg)
                    logger.error(f"✗ Failed {filename}: {error_msg}")
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create combined output if requested
        if combine_results and result.successful_files:
            self._create_combined_output(result, output_dir, output_prefix)
        
        # Log summary
        summary = result.get_summary()
        logger.info(f"Batch processing complete: {summary}")
        
        return result
    
    def _process_single_file(
        self,
        file_path: Path,
        output_dir: Optional[Path],
        output_prefix: str
    ) -> Tuple[Path, pd.DataFrame, int]:
        """
        Process a single file.
        
        Returns:
            Tuple of (output_path, dataframe, comments_count)
        """
        # Read input file
        df = self.excel_io.read_file(file_path, text_column=self.text_column)
        comments_count = len(df)
        
        if comments_count == 0:
            raise ClassificationError(f"No comments found in {file_path.name}")
        
        # Classify comments
        texts = df[self.text_column].tolist()
        predictions = self.classifier.classify_batch(texts)
        
        # Add predictions to dataframe
        for category, probs in predictions.items():
            df[category] = probs
        
        # Determine output path
        if output_dir is None:
            output_dir = file_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{output_prefix}{file_path.name}"
        output_path = output_dir / output_filename
        
        # Save results
        self.excel_io.save_results(df, output_path)
        
        return output_path, df, comments_count
    
    def _create_combined_output(
        self,
        result: BatchProcessingResult,
        output_dir: Optional[Path],
        output_prefix: str
    ):
        """Create a single combined output file from all successful results."""
        if not result.results_by_file:
            return
        
        # Add source filename column to each dataframe
        dfs_with_source = []
        for filename, df in result.results_by_file.items():
            df_copy = df.copy()
            df_copy.insert(0, 'source_file', filename)
            dfs_with_source.append(df_copy)
        
        # Combine all dataframes
        combined_df = pd.concat(dfs_with_source, ignore_index=True)
        
        # Determine output path
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"{output_prefix}combined_{timestamp}.xlsx"
        combined_path = output_dir / combined_filename
        
        # Save combined results
        self.excel_io.save_results(combined_df, combined_path)
        
        logger.info(f"Combined output saved to {combined_path}")
        result.combined_output_path = combined_path
    
    def validate_files(self, file_paths: List[Path]) -> Tuple[List[Path], List[str]]:
        """
        Validate input files before processing.
        
        Args:
            file_paths: List of file paths to validate
        
        Returns:
            Tuple of (valid_files, error_messages)
        """
        valid_files = []
        errors = []
        
        for file_path in file_paths:
            # Check file exists
            if not file_path.exists():
                errors.append(f"{file_path.name}: File not found")
                continue
            
            # Check file extension
            if file_path.suffix.lower() not in ['.xlsx', '.xls', '.csv']:
                errors.append(f"{file_path.name}: Unsupported format (use .xlsx, .xls, or .csv)")
                continue
            
            # Check file is readable
            try:
                df = self.excel_io.read_file(file_path, text_column=self.text_column)
                if len(df) == 0:
                    errors.append(f"{file_path.name}: No data found")
                    continue
            except Exception as e:
                errors.append(f"{file_path.name}: Cannot read file - {str(e)}")
                continue
            
            valid_files.append(file_path)
        
        return valid_files, errors
