"""
Batch file processing module for handling multiple files sequentially.

Supports:
- Multiple Excel/CSV files
- Sequential processing with progress tracking
- Error handling per file
- Consolidated results
- Timestamped output filenames
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import pandas as pd
from datetime import datetime

from src.core.classifier import CommentClassifier
from src.core.errors import PipelineError

logger = logging.getLogger(__name__)


class ClassificationError(PipelineError):
    """Exception raised during classification."""
    pass


class ExcelIO:
    """Helper class for reading/writing Excel and CSV files with chunking support."""
    
    # Default chunk size for large CSV files (in rows)
    DEFAULT_CHUNK_SIZE = 10000
    
    # File size threshold to trigger chunked processing (in bytes, default 50MB)
    CHUNK_THRESHOLD_BYTES = 50 * 1024 * 1024
    
    @staticmethod
    def should_chunk(file_path: Path) -> bool:
        """Determine if file should be processed in chunks based on size."""
        if file_path.suffix.lower() != '.csv':
            return False  # Only chunk CSV files for now
        
        try:
            file_size = file_path.stat().st_size
            return file_size > ExcelIO.CHUNK_THRESHOLD_BYTES
        except Exception:
            return False
    
    @staticmethod
    def read_file(file_path: Path, text_column: str = "text", encoding: str = 'utf-8') -> pd.DataFrame:
        """Read Excel or CSV file.
        
        Args:
            file_path: Path to the file
            text_column: Name of text column to validate
            encoding: Character encoding for CSV files (default: utf-8)
        
        Returns:
            DataFrame containing file contents
        """
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path, encoding=encoding)
        else:
            df = pd.read_excel(file_path)
        
        if text_column not in df.columns:
            raise ClassificationError(f"Column '{text_column}' not found in {file_path.name}")
        
        return df
    
    @staticmethod
    def read_csv_chunks(
        file_path: Path,
        text_column: str = "text",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        encoding: str = 'utf-8'
    ):
        """Read CSV file in chunks for memory-efficient processing.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column to validate
            chunk_size: Number of rows per chunk
            encoding: Character encoding
            
        Yields:
            Tuples of (chunk_df, chunk_index, chunk_start_row)
        """
        # Read first chunk to validate column exists
        first_chunk = pd.read_csv(file_path, nrows=1, encoding=encoding)
        if text_column not in first_chunk.columns:
            raise ClassificationError(f"Column '{text_column}' not found in {file_path.name}")
        
        # Read file in chunks
        chunk_index = 0
        chunk_start_row = 0
        
        for chunk_df in pd.read_csv(file_path, chunksize=chunk_size, encoding=encoding):
            yield chunk_df, chunk_index, chunk_start_row
            chunk_index += 1
            chunk_start_row += len(chunk_df)
    
    @staticmethod
    def save_results(df: pd.DataFrame, output_path: Path):
        """Save DataFrame to Excel file."""
        df.to_excel(output_path, index=False)
    
    @staticmethod
    def append_to_csv(df: pd.DataFrame, output_path: Path, write_header: bool = False):
        """Append DataFrame to CSV file.
        
        Args:
            df: DataFrame to append
            output_path: Path to output CSV
            write_header: Whether to write header (True for first chunk)
        """
        mode = 'w' if write_header else 'a'
        df.to_csv(output_path, mode=mode, header=write_header, index=False)

logger = logging.getLogger(__name__)


class BatchProcessingResult:
    """Container for batch processing results."""
    
    def __init__(self):
        self.successful_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []  # (filename, error_message)
        self.total_comments_processed: int = 0  # Comments that received predictions
        self.total_skipped_comments: int = 0  # Empty/invalid texts skipped
        self.processing_time: float = 0.0
        self.results_by_file: Dict[str, pd.DataFrame] = {}
        self.combined_output_path: Optional[Path] = None  # Path to combined output file if created
        self.skip_counts_by_file: Dict[str, int] = {}  # Track skipped rows per file
    
    def add_success(self, filename: str, df: pd.DataFrame, comments_count: int, skipped_count: int = 0):
        """Record successful file processing.
        
        Args:
            filename: Name of the processed file
            df: Resulting DataFrame with predictions
            comments_count: Total number of rows in file (processed + skipped)
            skipped_count: Number of rows that were skipped (empty/invalid)
        """
        self.successful_files.append(filename)
        self.results_by_file[filename] = df
        # Only count rows that received real predictions as "processed"
        processed_count = comments_count - skipped_count
        self.total_comments_processed += processed_count
        self.total_skipped_comments += skipped_count
        self.skip_counts_by_file[filename] = skipped_count
    
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
        total_rows = self.total_comments_processed + self.total_skipped_comments
        return {
            "total_files": len(self.successful_files) + len(self.failed_files),
            "successful_files": len(self.successful_files),
            "failed_files": len(self.failed_files),
            "success_rate": f"{self.success_rate:.1f}%",
            "total_rows": total_rows,  # All rows including skipped
            "processed_comments": self.total_comments_processed,  # Only rows with predictions
            "skipped_comments": self.total_skipped_comments,
            "processing_time": f"{self.processing_time:.2f}s"
        }


class BatchProcessor:
    """Process multiple files sequentially with progress tracking.
    
    Optimized for typical use case of 2-3 files with 5-10k rows each.
    Sequential processing is simpler and sufficient for this scale.
    """
    
    def __init__(
        self,
        classifier: CommentClassifier,
        text_column: str = "text"
    ):
        """
        Initialize batch processor.
        
        Args:
            classifier: CommentClassifier instance for classification
            text_column: Name of text column in input files
        """
        self.classifier = classifier
        self.text_column = text_column
        self.excel_io = ExcelIO()
        logger.info("Batch processor initialized for sequential processing")
    
    def process_files(
        self,
        file_paths: List[Path],
        output_dir: Optional[Path] = None,
        output_prefix: str = "classified_",
        combine_results: bool = False,
        validated_data: Optional[Dict[Path, pd.DataFrame]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> BatchProcessingResult:
        """
        Process multiple files sequentially.
        
        Args:
            file_paths: List of input file paths
            output_dir: Directory for output files (None = same as input)
            output_prefix: Prefix for output filenames
            combine_results: If True, create combined output file
            validated_data: Pre-loaded DataFrames from validation (avoids duplicate I/O)
            progress_callback: Optional callback(filename, current, total) for progress updates
        
        Returns:
            BatchProcessingResult with processing details
        """
        start_time = datetime.now()
        result = BatchProcessingResult()
        total_files = len(file_paths)
        
        logger.info(f"Starting sequential processing of {total_files} files")
        
        # Process files one by one
        for idx, file_path in enumerate(file_paths, 1):
            filename = file_path.name
            
            try:
                # Process single file
                cached_df = validated_data.get(file_path) if validated_data else None
                output_path, df, comments_count, skipped_count = self._process_single_file(
                    file_path,
                    output_dir,
                    output_prefix,
                    cached_df
                )
                
                result.add_success(filename, df, comments_count, skipped_count)
                
                # Invoke progress callback
                if progress_callback:
                    try:
                        progress_callback(filename, idx, total_files)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
                
                # Log per-file skip count
                if skipped_count > 0:
                    pct = (skipped_count / comments_count * 100) if comments_count > 0 else 0
                    logger.warning(
                        f"✓ Processed {filename}: {comments_count} total rows, "
                        f"{comments_count - skipped_count} classified, {skipped_count} skipped ({pct:.1f}%)"
                    )
                else:
                    logger.info(f"✓ Processed {filename}: {comments_count} comments")
                    
            except Exception as e:
                error_msg = str(e)
                result.add_failure(filename, error_msg)
                
                # Invoke progress callback even on failure
                if progress_callback:
                    try:
                        progress_callback(filename, idx, total_files)
                    except Exception as e2:
                        logger.warning(f"Progress callback failed: {e2}")
                
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
        output_prefix: str,
        cached_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Path, pd.DataFrame, int, int]:
        """
        Process a single file with automatic chunking for large CSVs.
        Generates timestamped output filename.
        
        Args:
            file_path: Path to input file
            output_dir: Directory for output file
            output_prefix: Prefix for output filename
            cached_df: Pre-loaded DataFrame from validation (avoids re-reading file)
        
        Returns:
            Tuple of (output_path, dataframe, comments_count, skipped_count)
        """
        # Determine output path with timestamp
        if output_dir is None:
            output_dir = file_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped filename: classified_OriginalName_YYYYMMDD_HHMMSS.xlsx
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_stem = file_path.stem  # filename without extension
        output_filename = f"{output_prefix}{file_stem}_{timestamp}.xlsx"
        output_path = output_dir / output_filename
        
        # Check if file should be processed in chunks (large CSVs only)
        use_chunking = (cached_df is None and self.excel_io.should_chunk(file_path))
        
        if use_chunking:
            # Chunked processing for large CSV files
            return self._process_file_chunked(file_path, output_path)
        else:
            # Standard in-memory processing
            return self._process_file_inmemory(file_path, output_path, cached_df)
    
    def _process_file_inmemory(
        self,
        file_path: Path,
        output_path: Path,
        cached_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Path, pd.DataFrame, int, int]:
        """Process file entirely in memory (standard path)."""
        # Use cached DataFrame if available, otherwise read file
        if cached_df is not None:
            df = cached_df.copy(deep=False)  # Shallow copy to save memory
        else:
            df = self.excel_io.read_file(file_path, text_column=self.text_column)
        
        comments_count = len(df)
        
        if comments_count == 0:
            raise ClassificationError(f"No comments found in {file_path.name}")
        
        # Classify comments
        texts = df[self.text_column].tolist()
        predictions = self.classifier.classify_batch(texts)
        
        # Extract metadata about skipped rows
        metadata = predictions.pop('_metadata', {})
        skipped_indices = metadata.get('skipped_indices', [])
        skipped_count = len(skipped_indices)
        
        # Add predictions to dataframe
        for category, probs in predictions.items():
            df[category] = probs
        
        # Atomic write: write to temp file first, then rename
        output_dir = output_path.parent
        temp_output_path = output_dir / f".{output_path.name}.tmp"
        
        try:
            # Save results to temp file
            self.excel_io.save_results(df, temp_output_path)
            
            # Atomic rename (overwrites if exists on most systems)
            if output_path.exists():
                output_path.unlink()  # Delete existing file first on Windows
            temp_output_path.rename(output_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise
        
        return output_path, df, comments_count, skipped_count
    
    def _process_file_chunked(
        self,
        file_path: Path,
        output_path: Path
    ) -> Tuple[Path, pd.DataFrame, int, int]:
        """Process large CSV file in chunks to avoid memory issues."""
        logger.info(f"Processing {file_path.name} in chunks (file size > 50MB)")
        
        output_dir = output_path.parent
        temp_output_path = output_dir / f".{output_path.name}.tmp"
        
        total_comments = 0
        total_skipped = 0
        all_chunks = []  # Store processed chunks for return value
        
        try:
            # Process file chunk by chunk
            for chunk_df, chunk_idx, chunk_start_row in self.excel_io.read_csv_chunks(
                file_path, text_column=self.text_column
            ):
                chunk_size = len(chunk_df)
                total_comments += chunk_size
                
                # Classify chunk
                texts = chunk_df[self.text_column].tolist()
                predictions = self.classifier.classify_batch(texts)
                
                # Extract metadata
                metadata = predictions.pop('_metadata', {})
                skipped_indices = metadata.get('skipped_indices', [])
                total_skipped += len(skipped_indices)
                
                # Add predictions to chunk
                for category, probs in predictions.items():
                    chunk_df[category] = probs
                
                # Append chunk to output file (write header only for first chunk)
                write_header = (chunk_idx == 0)
                self.excel_io.append_to_csv(chunk_df, temp_output_path, write_header=write_header)
                
                # Keep chunk in memory for return value (if not too many chunks)
                if chunk_idx < 100:  # Limit memory usage
                    all_chunks.append(chunk_df)
                
                logger.debug(
                    f"Processed chunk {chunk_idx + 1} of {file_path.name}: "
                    f"rows {chunk_start_row}-{chunk_start_row + chunk_size - 1}"
                )
            
            # Atomic rename
            if output_path.exists():
                output_path.unlink()
            temp_output_path.rename(output_path)
            
            # Combine chunks for return value (or read back from file if too many chunks)
            if all_chunks:
                combined_df = pd.concat(all_chunks, ignore_index=True)
            else:
                # Fallback: read the output file we just created
                combined_df = pd.read_csv(output_path)
            
            logger.info(
                f"Chunked processing complete for {file_path.name}: "
                f"{total_comments} rows, {total_skipped} skipped"
            )
            
        except Exception as e:
            # Clean up temp file on error
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise
        
        return output_path, combined_df, total_comments, total_skipped
    
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
    
    def validate_files(
        self, 
        file_paths: List[Path]
    ) -> Tuple[List[Path], List[str], Dict[Path, pd.DataFrame]]:
        """
        Validate input files before processing and cache loaded DataFrames.
        
        Args:
            file_paths: List of file paths to validate
        
        Returns:
            Tuple of (valid_files, error_messages, cached_dataframes)
            - valid_files: List of paths that passed validation
            - error_messages: List of error descriptions for failed files
            - cached_dataframes: Dict mapping valid file paths to their loaded DataFrames
        """
        valid_files = []
        errors = []
        cached_data = {}
        
        for file_path in file_paths:
            # Check file exists
            if not file_path.exists():
                errors.append(f"{file_path.name}: File not found")
                continue
            
            # Check file extension
            if file_path.suffix.lower() not in ['.xlsx', '.xls', '.csv']:
                errors.append(f"{file_path.name}: Unsupported format (use .xlsx, .xls, or .csv)")
                continue
            
            # Check file is readable and cache DataFrame
            try:
                df = self.excel_io.read_file(file_path, text_column=self.text_column)
                if len(df) == 0:
                    errors.append(f"{file_path.name}: No data found")
                    continue
                
                # Cache the loaded DataFrame to avoid re-reading in process_files
                cached_data[file_path] = df
            except Exception as e:
                errors.append(f"{file_path.name}: Cannot read file - {str(e)}")
                continue
            
            valid_files.append(file_path)
        
        return valid_files, errors, cached_data
    
    def dry_run_validation(
        self,
        file_paths: List[Path]
    ) -> Dict[str, Dict]:
        """
        Perform comprehensive validation without classification (dry-run mode).
        
        Checks:
        - File readability and format
        - Schema (required columns)
        - Data quality (empty rate, text length distribution)
        - Encoding issues
        - Memory estimates
        
        Args:
            file_paths: List of file paths to validate
        
        Returns:
            Dict mapping filenames to validation reports:
            {
                'filename.xlsx': {
                    'status': 'valid' | 'warning' | 'error',
                    'total_rows': 1000,
                    'empty_rate': 0.05,
                    'avg_text_length': 250,
                    'max_text_length': 5000,
                    'encoding': 'utf-8',
                    'estimated_memory_mb': 15.2,
                    'issues': ['5% of rows are empty', ...],
                    'error': None or error message
                }
            }
        """
        from src.core.skip_reasons import SkipReason
        import sys
        
        validation_results = {}
        
        for file_path in file_paths:
            filename = file_path.name
            report = {
                'status': 'valid',
                'total_rows': 0,
                'empty_rate': 0.0,
                'avg_text_length': 0,
                'max_text_length': 0,
                'encoding': 'unknown',
                'estimated_memory_mb': 0.0,
                'issues': [],
                'error': None
            }
            
            # Check file exists
            if not file_path.exists():
                report['status'] = 'error'
                report['error'] = 'File not found'
                validation_results[filename] = report
                continue
            
            # Check file extension
            if file_path.suffix.lower() not in ['.xlsx', '.xls', '.csv']:
                report['status'] = 'error'
                report['error'] = 'Unsupported format (use .xlsx, .xls, or .csv)'
                validation_results[filename] = report
                continue
            
            # Try to read and analyze file
            try:
                # Detect encoding for CSV files
                if file_path.suffix.lower() == '.csv':
                    # Simple encoding detection
                    with open(file_path, 'rb') as f:
                        raw_sample = f.read(10000)
                    try:
                        raw_sample.decode('utf-8')
                        report['encoding'] = 'utf-8'
                    except UnicodeDecodeError:
                        try:
                            raw_sample.decode('latin-1')
                            report['encoding'] = 'latin-1'
                            report['issues'].append('Non-UTF8 encoding detected (latin-1)')
                        except:
                            report['encoding'] = 'unknown'
                            report['issues'].append('Could not determine encoding')
                
                # Read file
                df = self.excel_io.read_file(file_path, text_column=self.text_column)
                
                if len(df) == 0:
                    report['status'] = 'error'
                    report['error'] = 'No data found in file'
                    validation_results[filename] = report
                    continue
                
                report['total_rows'] = len(df)
                
                # Analyze text column
                texts = df[self.text_column].tolist()
                
                # Count skip reasons
                skip_reasons = [SkipReason.from_text(text) for text in texts]
                skip_counts = {}
                for reason in skip_reasons:
                    reason_str = str(reason)
                    skip_counts[reason_str] = skip_counts.get(reason_str, 0) + 1
                
                # Calculate metrics
                total_skipped = sum(count for reason, count in skip_counts.items() if reason != 'none')
                report['empty_rate'] = total_skipped / len(texts) if texts else 0
                
                # Text length stats (only for valid texts)
                valid_texts = [text for text, reason in zip(texts, skip_reasons) if reason == SkipReason.NONE]
                if valid_texts:
                    text_lengths = [len(str(text)) for text in valid_texts]
                    report['avg_text_length'] = int(sum(text_lengths) / len(text_lengths))
                    report['max_text_length'] = max(text_lengths)
                
                # Memory estimate (rough)
                report['estimated_memory_mb'] = round(sys.getsizeof(df) / 1024 / 1024, 2)
                
                # Generate issues list
                if report['empty_rate'] > 0.1:
                    report['status'] = 'warning'
                    report['issues'].append(f"{report['empty_rate']*100:.1f}% of rows are empty/invalid")
                
                if skip_counts:
                    for reason, count in skip_counts.items():
                        if reason != 'none' and count > 0:
                            pct = (count / len(texts)) * 100
                            report['issues'].append(f"{count} rows ({pct:.1f}%) - {reason}")
                
                if report['max_text_length'] > 5000:
                    report['status'] = 'warning'
                    report['issues'].append(f"Some texts are very long (max: {report['max_text_length']} chars)")
                
                if not report['issues']:
                    report['issues'].append('No issues detected')
                
            except Exception as e:
                report['status'] = 'error'
                report['error'] = str(e)
            
            validation_results[filename] = report
        
        return validation_results
