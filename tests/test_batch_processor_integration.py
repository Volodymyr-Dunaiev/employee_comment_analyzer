"""
Integration tests for BatchProcessor with real CommentClassifier.

These tests require a trained model in model/ukr_multilabel/.
Tests are skipped if the model is not available.

Run with: pytest tests/test_batch_processor_integration.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import time

from src.core.batch_processor import BatchProcessor
from src.core.classifier import CommentClassifier


def get_real_classifier():
    """Get real classifier if model available, otherwise skip test."""
    model_path = Path("model/ukr_multilabel")
    if not model_path.exists():
        pytest.skip("Model not available - run training first")
    
    from src.utils.config import load_config
    config = load_config()
    return CommentClassifier(config)


def generate_realistic_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic test data."""
    np.random.seed(seed)
    
    comments = [
        "Хороша зарплата та умови",
        "Погана атмосфера в команді",
        "Чудовий колектив",
        "Низька оплата праці",
        "Дружня команда",
        "",  # Empty
        "   ",  # Whitespace
        None,  # Null
    ]
    
    texts = []
    for i in range(n_rows):
        if i % 50 == 0:
            texts.append("")
        elif i % 47 == 0:
            texts.append(None)
        else:
            texts.append(np.random.choice(comments[:5]) + f" {i % 10}")
    
    return pd.DataFrame({'text': texts})

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def realistic_files_10k(temp_dir):
    """Create two 5k-row files (10k total)."""
    files = []
    
    df1 = generate_realistic_data(5000, seed=42)
    file1 = temp_dir / "batch1.xlsx"
    df1.to_excel(file1, index=False)
    files.append(file1)
    
    df2 = generate_realistic_data(5000, seed=43)
    file2 = temp_dir / "batch2.xlsx"
    df2.to_excel(file2, index=False)
    files.append(file2)
    
    return files


@pytest.fixture
def sample_excel_files(temp_dir):
    """Create small sample files for quick tests."""
    files = []
    
    df1 = pd.DataFrame({
        'text': [
            'Positive comment about salary',
            'Great team and management',
            'Excellent work environment'
        ]
    })
    file1 = temp_dir / "sample1.xlsx"
    df1.to_excel(file1, index=False)
    files.append(file1)
    
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
    """Integration tests with real classifier."""
    
    def test_classify_batch_method(self):
        """Verify classify_batch method returns correct format."""
        classifier = get_real_classifier()
        
        texts = ["Test comment 1", "Test comment 2"]
        result = classifier.classify_batch(texts)
        
        # Should return dictionary
        assert isinstance(result, dict)
        
        # Should have skip_reason column
        assert 'skip_reason' in result
        assert isinstance(result['skip_reason'], list)
        assert len(result['skip_reason']) == len(texts)
        
        # Should have _metadata
        assert '_metadata' in result
        assert 'skipped_indices' in result['_metadata']
        assert 'skip_reason_counts' in result['_metadata']
        
        # Category columns should have probabilities
        special_keys = {'skip_reason', '_metadata'}
        for category, probs in result.items():
            if category in special_keys:
                continue
            assert isinstance(probs, list)
            assert len(probs) == len(texts)
            assert all(isinstance(p, float) for p in probs)
            assert all(0.0 <= p <= 1.0 for p in probs)
    
    def test_single_file_processing(self, sample_excel_files, temp_dir):
        """Test processing single file."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            [sample_excel_files[0]],
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        
        assert len(result.successful_files) == 1
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 3
        
        # Verify output file
        output_file = temp_dir / "output" / "classified_sample1.xlsx"
        assert output_file.exists()
        
        df = pd.read_excel(output_file)
        assert 'text' in df.columns
        assert 'skip_reason' in df.columns
        
        category_cols = [col for col in df.columns if col not in ['text', 'skip_reason']]
        assert len(category_cols) > 0
    
    def test_multiple_files_processing(self, sample_excel_files, temp_dir):
        """Test processing multiple files."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            sample_excel_files,
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 5  # 3 + 2
    
    def test_multiprocessing_2_files(self, realistic_files_10k, temp_dir):
        """Test actual use case: 2 files with multiprocessing."""
        classifier = get_real_classifier()
        processor = BatchProcessor(
            classifier,
            max_workers=2,
            use_multiprocessing=True,
            text_column='text'
        )
        
        start_time = time.time()
        result = processor.process_files(
            realistic_files_10k,
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        processing_time = time.time() - start_time
        
        # Verify success
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 10000
        
        # Verify outputs exist
        for file in realistic_files_10k:
            output_file = temp_dir / "output" / f"classified_{file.name}"
            assert output_file.exists()
            df = pd.read_excel(output_file)
            assert len(df) == 5000
            assert 'skip_reason' in df.columns
        
        print(f"\n2-file multiprocessing (10k rows): {processing_time:.2f}s")
        print(f"Throughput: {10000/processing_time:.0f} rows/sec")
    
    def test_combined_output_mode(self, sample_excel_files, temp_dir):
        """Test combined output file generation."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier, max_workers=1, text_column='text')
        
        result = processor.process_files(
            sample_excel_files,
            output_dir=temp_dir / "output",
            combine_results=True
        )
        
        # Verify combined file created
        assert result.combined_output_path is not None
        assert result.combined_output_path.exists()
        
        df = pd.read_excel(result.combined_output_path)
        assert 'source_file' in df.columns
        assert len(df) == 5
        assert 'sample1.xlsx' in df['source_file'].values
        assert 'sample2.xlsx' in df['source_file'].values
    
    def test_skip_reason_handling(self, temp_dir):
        """Test skip_reason categorization."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier, max_workers=1, text_column='text')
        
        # Create file with various skip scenarios
        df = pd.DataFrame({
            'text': [
                'Valid comment',
                '',  # empty
                '   ',  # whitespace
                None,  # nan
                123,  # non_text
                'Another valid comment'
            ]
        })
        file_path = temp_dir / "test.xlsx"
        df.to_excel(file_path, index=False)
        
        result = processor.process_files(
            [file_path],
            output_dir=temp_dir / "output"
        )
        
        assert result.total_comments_processed == 6
        
        # Check skip_reason column
        output_file = temp_dir / "output" / f"classified_{file_path.name}"
        df_result = pd.read_excel(output_file)
        
        skip_reasons = df_result['skip_reason'].value_counts()
        assert 'none' in skip_reasons  # Valid comments
        assert skip_reasons['none'] >= 2  # At least 2 valid comments
    
    def test_dry_run_validation(self, realistic_files_10k):
        """Test dry-run validation without classification."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier, text_column='text')
        
        validation = processor.dry_run_validation(realistic_files_10k)
        
        assert validation['total_files'] == 2
        assert validation['total_rows'] == 10000
        assert len(validation['files']) == 2
        assert len(validation['issues']) == 0
        
        for file_name, stats in validation['files'].items():
            assert 'total_rows' in stats
            assert 'empty_rows' in stats
            assert 'empty_rate' in stats
            assert stats['total_rows'] == 5000
