"""
Integration tests for BatchProcessor with real CommentClassifier.

These tests use a mock model by default to avoid network dependencies.
Set environment variable USE_REAL_MODEL=1 to test with actual HuggingFace models.

Run normally (mock model, fast): pytest tests/test_batch_processor_integration.py
Run with real model (slow): USE_REAL_MODEL=1 pytest tests/test_batch_processor_integration.py -m slow
Skip slow tests: pytest -m "not slow"
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import os

from src.core.batch_processor import BatchProcessor
from src.core.classifier import CommentClassifier
from tests.fixtures.mock_model import mock_transformers_loading


# Check if we should use real model (for thorough testing)
USE_REAL_MODEL = os.environ.get('USE_REAL_MODEL', '0') == '1'

if USE_REAL_MODEL:
    # Set cache directory for Hugging Face to avoid re-downloading in CI
    os.environ['HF_HOME'] = os.path.join(tempfile.gettempdir(), 'hf_cache')


@pytest.fixture
def mock_hf_models(monkeypatch):
    """Mock HuggingFace model loading to avoid network access."""
    if not USE_REAL_MODEL:
        mock_transformers_loading(monkeypatch, num_labels=3)


@pytest.fixture
def test_config():
    """Create test configuration with tiny model."""
    return {
        'model': {
            'path': 'prajjwal1/bert-tiny',  # 17MB tiny BERT for testing
            'batch_size': 4,
            'device': 'cpu'
        },
        'data': {
            'text_column': 'text',
            'chunk_size': 100,
            'input_validation': {
                'max_length': 512,
                'required_columns': ['text']
            }
        },
        'categories': [
            'Category1',
            'Category2',
            'Category3'
        ]
    }


@pytest.fixture
def real_classifier(test_config, mock_hf_models):
    """Create real CommentClassifier with tiny model (or mock if USE_REAL_MODEL=0)."""
    try:
        classifier = CommentClassifier(test_config)
        return classifier
    except Exception as e:
        pytest.skip(f"Cannot load test model: {e}")


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_excel_files(temp_dir):
    """Create sample Excel files for testing."""
    files = []
    
    # File 1
    df1 = pd.DataFrame({
        'text': [
            'This is a positive comment',
            'Great team and management',
            'Excellent work environment'
        ]
    })
    file1 = temp_dir / "sample1.xlsx"
    df1.to_excel(file1, index=False)
    files.append(file1)
    
    # File 2
    df2 = pd.DataFrame({
        'text': [
            'Need better training programs',
            'Good benefits package'
        ]
    })
    file2 = temp_dir / "sample2.xlsx"
    df2.to_excel(file2, index=False)
    files.append(file2)
    
    return files


@pytest.mark.integration
class TestBatchProcessorIntegration:
    """Integration tests with real classifier (using mock model by default)."""
    
    def test_classify_batch_method_exists(self, real_classifier):
        """Verify classify_batch method exists and has correct signature."""
        assert hasattr(real_classifier, 'classify_batch')
        assert callable(real_classifier.classify_batch)
    
    def test_classify_batch_returns_correct_format(self, real_classifier):
        """Test that classify_batch returns Dict[str, Any] with probabilities and metadata."""
        texts = ["Test comment 1", "Test comment 2"]
        
        result = real_classifier.classify_batch(texts)
        
        # Should return dictionary
        assert isinstance(result, dict)
        
        # Should have skip_reason column
        assert 'skip_reason' in result
        assert isinstance(result['skip_reason'], list)
        assert len(result['skip_reason']) == len(texts)
        assert all(isinstance(v, str) for v in result['skip_reason'])
        
        # Should have _metadata with skip_reason_counts
        assert '_metadata' in result
        assert 'skipped_indices' in result['_metadata']
        assert 'skip_reason_counts' in result['_metadata']
        
        # Each category should map to list of floats (excluding special keys)
        special_keys = {'skip_reason', '_metadata'}
        for category, probs in result.items():
            if category in special_keys:
                continue
            assert isinstance(category, str)
            assert isinstance(probs, list)
            assert len(probs) == len(texts)
            assert all(isinstance(p, float) for p in probs)
            assert all(0.0 <= p <= 1.0 for p in probs)
    
    def test_batch_processor_single_file(self, real_classifier, sample_excel_files, temp_dir):
        """Test processing single file with real classifier."""
        processor = BatchProcessor(real_classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            [sample_excel_files[0]],
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        
        # Verify processing succeeded
        assert len(result.successful_files) == 1
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 3
        
        # Verify output file exists
        output_file = temp_dir / "output" / "classified_sample1.xlsx"
        assert output_file.exists()
        
        # Verify output contains predictions
        df = pd.read_excel(output_file)
        assert 'text' in df.columns
        assert 'skip_reason' in df.columns
        
        # Should have category columns with probabilities (exclude text and skip_reason)
        category_cols = [col for col in df.columns if col not in ['text', 'skip_reason']]
        assert len(category_cols) > 0
        
        # All probabilities should be between 0 and 1
        for col in category_cols:
            assert df[col].min() >= 0.0
            assert df[col].max() <= 1.0
    
    def test_batch_processor_multiple_files(self, real_classifier, sample_excel_files, temp_dir):
        """Test processing multiple files with real classifier."""
        processor = BatchProcessor(real_classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            sample_excel_files,
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        
        # Verify both files processed
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 5  # 3 + 2
        
        # Verify output files exist
        for sample_file in sample_excel_files:
            output_file = temp_dir / "output" / f"classified_{sample_file.name}"
            assert output_file.exists()
    
    def test_batch_processor_with_validation_cache(self, real_classifier, sample_excel_files, temp_dir):
        """Test that validation caching works correctly."""
        processor = BatchProcessor(real_classifier, max_workers=1, text_column='text')
        
        # Validate and get cached data
        valid_files, errors, cached_data = processor.validate_files(sample_excel_files)
        
        assert len(valid_files) == 2
        assert len(errors) == 0
        assert len(cached_data) == 2
        
        # Process with cached data
        result = processor.process_files(
            valid_files,
            output_dir=temp_dir / "output",
            validated_data=cached_data
        )
        
        assert len(result.successful_files) == 2
        assert result.total_comments_processed == 5
    
    def test_batch_processor_combined_output(self, real_classifier, sample_excel_files, temp_dir):
        """Test combined output generation."""
        processor = BatchProcessor(real_classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            sample_excel_files,
            output_dir=temp_dir / "output",
            combine_results=True
        )
        
        # Verify combined output was created
        assert result.combined_output_path is not None
        assert result.combined_output_path.exists()
        
        # Read combined output
        df_combined = pd.read_excel(result.combined_output_path)
        
        # Should have source_file column
        assert 'source_file' in df_combined.columns
        
        # Should have all rows from both files
        assert len(df_combined) == 5
        
        # Should have both source files
        source_files = df_combined['source_file'].unique()
        assert len(source_files) == 2
    
    def test_thread_safety_with_concurrent_processing(self, real_classifier, sample_excel_files, temp_dir):
        """Test that thread-safe locking works with max_workers > 1."""
        # Note: CPU-only, so we can test with max_workers > 1
        processor = BatchProcessor(real_classifier, max_workers=2, text_column='text')
        
        result = processor.process_files(
            sample_excel_files,
            output_dir=temp_dir / "output"
        )
        
        # Should complete without errors despite concurrent access
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 5
    
    def test_empty_text_handling(self, real_classifier, temp_dir):
        """Test handling of files with empty text and verify skip tracking."""
        # Create file with empty text
        df = pd.DataFrame({
            'text': ['Valid text', '', 'Another valid text']
        })
        file_path = temp_dir / "with_empty.xlsx"
        df.to_excel(file_path, index=False)
        
        processor = BatchProcessor(real_classifier, max_workers=1, text_column='text')
        
        # Should process successfully with skip tracking
        result = processor.process_files(
            [file_path],
            output_dir=temp_dir / "output"
        )
        
        assert len(result.successful_files) == 1
        # Only valid texts count as "processed"
        assert result.total_comments_processed == 2
        assert result.total_skipped_comments == 1
        
        # Verify output DataFrame has skip_reason column
        output_df = result.results_by_file['with_empty.xlsx']
        assert 'skip_reason' in output_df.columns
        # Empty string becomes NaN when written to/read from Excel
        assert output_df['skip_reason'].tolist() == ['none', 'nan', 'none']


@pytest.mark.integration
class TestBatchProcessorAPIContract:
    """Tests that verify the API contract between components (using mock model)."""
    
    def test_classify_batch_signature(self, real_classifier):
        """Verify classify_batch has expected signature."""
        import inspect
        
        sig = inspect.signature(real_classifier.classify_batch)
        params = list(sig.parameters.keys())
        
        # Should accept texts parameter
        assert 'texts' in params
    
    def test_classify_batch_with_empty_list(self, real_classifier):
        """Test classify_batch handles empty input."""
        result = real_classifier.classify_batch([])
        
        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)
    
    def test_classify_batch_categories_from_config(self, real_classifier, test_config):
        """Test that classify_batch uses categories from config."""
        texts = ["Test comment"]
        result = real_classifier.classify_batch(texts)
        
        # Should return categories matching config (plus skip_reason and _metadata)
        expected_categories = test_config['categories']
        special_keys = {'skip_reason', '_metadata'}
        category_keys = [k for k in result.keys() if k not in special_keys]
        
        assert len(category_keys) == len(expected_categories)
        
        for category in expected_categories:
            assert category in result


@pytest.mark.integration
class TestBatchProcessorPerformance:
    """Performance-related integration tests (using mock model for speed)."""
    
    def test_large_batch_processing(self, real_classifier, temp_dir):
        """Test processing larger batch of comments."""
        # Create file with 100 comments
        df = pd.DataFrame({
            'text': [f'Comment number {i}' for i in range(100)]
        })
        file_path = temp_dir / "large_file.xlsx"
        df.to_excel(file_path, index=False)
        
        processor = BatchProcessor(real_classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            [file_path],
            output_dir=temp_dir / "output"
        )
        
        assert len(result.successful_files) == 1
        assert result.total_comments_processed == 100
        assert result.processing_time > 0
