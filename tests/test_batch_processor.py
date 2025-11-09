"""
Tests for batch file processing functionality with realistic data scales.

Focus on real-world scenarios:
- 10k-20k row datasets (actual use case)
- 2-file parallel processing
- Memory profiling
- Skip if model unavailable (no mocks)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import tracemalloc
import time

from src.core.batch_processor import BatchProcessor, BatchProcessingResult, ExcelIO
from src.core.classifier import CommentClassifier


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


def generate_realistic_text_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate realistic employee comment data for testing.
    
    Args:
        n_rows: Number of rows to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with realistic employee comments
    """
    np.random.seed(seed)
    
    # Realistic comment templates
    comment_templates = [
        "Хороша зарплата та соціальний пакет",
        "Погана атмосфера в команді",
        "Чудовий колектив та підтримка",
        "Низька зарплата для цієї позиції",
        "Дружній колектив, але багато роботи",
        "Важка робота без додаткової оплати",
        "Хороші умови праці та офіс",
        "Погане керівництво та організація",
        "Відмінна компанія для розвитку",
        "Немає можливостей для кар'єрного зростання",
        "Цікаві проєкти та технології",
        "Мало навчання та розвитку",
        "Гнучкий графік роботи",
        "Багато переробок та стресу",
        "Професійна команда експертів",
        "",  # Empty rows
        "   ",  # Whitespace rows
        "Excellent workplace culture and benefits",  # English text
        "123",  # Numbers
        None,  # None values
    ]
    
    # Generate rows with realistic distribution
    texts = []
    for i in range(n_rows):
        if i % 100 == 0:  # 1% empty
            texts.append("")
        elif i % 97 == 0:  # ~1% whitespace
            texts.append("   ")
        elif i % 89 == 0:  # ~1% None
            texts.append(None)
        elif i % 83 == 0:  # ~1% numbers
            texts.append(str(np.random.randint(1, 1000)))
        else:
            # Pick random comment template
            base_text = np.random.choice(comment_templates[:15])
            # Add variation
            variation = np.random.randint(1, 10)
            texts.append(f"{base_text} ({variation})")
    
    return pd.DataFrame({'text': texts})


@pytest.fixture
def realistic_files_10k(temp_dir):
    """Create two 5k-row files (10k total) matching actual use case."""
    excel_io = ExcelIO()
    files = []
    
    # File 1: 5k rows
    df1 = generate_realistic_text_data(5000, seed=42)
    file1 = temp_dir / "comments_batch1.xlsx"
    excel_io.save_results(df1, file1)
    files.append(file1)
    
    # File 2: 5k rows
    df2 = generate_realistic_text_data(5000, seed=43)
    file2 = temp_dir / "comments_batch2.xlsx"
    excel_io.save_results(df2, file2)
    files.append(file2)
    
    return files


@pytest.fixture
def realistic_file_20k(temp_dir):
    """Create single 20k-row file (upper bound use case)."""
    excel_io = ExcelIO()
    
    df = generate_realistic_text_data(20000, seed=44)
    file_path = temp_dir / "comments_large.xlsx"
    excel_io.save_results(df, file_path)
    
    return file_path


@pytest.fixture
def sample_files(temp_dir):
    """Create small sample files for quick unit tests."""
    excel_io = ExcelIO()
    files = []
    
    # File 1: Small dataset
    df1 = pd.DataFrame({
        'text': [
            'Хороша зарплата',
            'Погана атмосфера',
            'Чудовий колектив'
        ]
    })
    file1 = temp_dir / "sample1.xlsx"
    excel_io.save_results(df1, file1)
    files.append(file1)
    
    # File 2: Medium dataset
    df2 = pd.DataFrame({
        'text': [
            'Низька зарплата',
            'Дружній колектив',
            'Важка робота',
            'Хороші умови',
            'Погане керівництво'
        ]
    })
    file2 = temp_dir / "sample2.xlsx"
    excel_io.save_results(df2, file2)
    files.append(file2)
    
    return files


@pytest.fixture
def empty_file(temp_dir):
    """Create empty Excel file."""
    df = pd.DataFrame({'text': []})
    file_path = temp_dir / "empty.xlsx"
    ExcelIO().save_results(df, file_path)
    return file_path


def get_real_classifier():
    """Get real classifier if model available, otherwise skip test."""
    model_path = Path("model/ukr_multilabel")
    if not model_path.exists():
        pytest.skip("Model not available - run training first or use small test model")
    
    from src.utils.config import load_config
    config = load_config()
    return CommentClassifier(config)


class TestBatchProcessingResult:
    """Test BatchProcessingResult class."""
    
    def test_initialization(self):
        """Test result object initialization."""
        result = BatchProcessingResult()
        assert result.successful_files == []
        assert result.failed_files == []
        assert result.total_comments_processed == 0
        assert result.total_skipped_comments == 0
        assert result.processing_time == 0.0
        assert result.combined_output_path is None
    
    def test_add_success(self):
        """Test adding successful file result."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test1', 'test2']})
        
        result.add_success('file1.xlsx', df, 2)
        
        assert 'file1.xlsx' in result.successful_files
        assert result.total_comments_processed == 2
        assert 'file1.xlsx' in result.results_by_file
    
    def test_add_failure(self):
        """Test adding failed file result."""
        result = BatchProcessingResult()
        
        result.add_failure('file1.xlsx', 'File not found')
        
        assert len(result.failed_files) == 1
        assert result.failed_files[0] == ('file1.xlsx', 'File not found')
    
    def test_success_rate_all_success(self):
        """Test success rate calculation with all successful."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 1)
        result.add_success('file2.xlsx', df, 1)
        
        assert result.success_rate == 100.0
    
    def test_success_rate_mixed(self):
        """Test success rate calculation with mixed results."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 1)
        result.add_failure('file2.xlsx', 'Error')
        result.add_failure('file3.xlsx', 'Error')
        
        assert result.success_rate == pytest.approx(33.33, rel=0.1)
    
    def test_success_rate_empty(self):
        """Test success rate with no files."""
        result = BatchProcessingResult()
        assert result.success_rate == 0.0
    
    def test_get_summary(self):
        """Test summary generation."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 5, skipped_count=0)
        result.add_failure('file2.xlsx', 'Error')
        result.processing_time = 12.345
        
        summary = result.get_summary()
        
        assert summary['total_files'] == 2
        assert summary['successful_files'] == 1
        assert summary['failed_files'] == 1
        assert summary['total_rows'] == 5
        assert summary['processed_comments'] == 5
        assert summary['skipped_comments'] == 0
        assert '12.35' in summary['processing_time']


class TestBatchProcessorUnit:
    """Unit tests for BatchProcessor (no model required)."""
    
    def test_validate_files_nonexistent(self, temp_dir):
        """Test file validation with nonexistent file."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        fake_file = temp_dir / "nonexistent.xlsx"
        
        valid, errors, cached_data = processor.validate_files([fake_file])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert len(cached_data) == 0
        assert 'not found' in errors[0].lower()
    
    def test_validate_files_wrong_format(self, temp_dir):
        """Test file validation with wrong file format."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        # Create text file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test data")
        
        valid, errors, cached_data = processor.validate_files([txt_file])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert len(cached_data) == 0
        assert 'Unsupported format' in errors[0]
    
    def test_validate_files_empty_data(self, empty_file):
        """Test file validation with empty file."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        valid, errors, cached_data = processor.validate_files([empty_file])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert len(cached_data) == 0
        assert 'No data' in errors[0]


@pytest.mark.integration
class TestBatchProcessorRealistic:
    """Integration tests with realistic data scales (10k-20k rows).
    
    Tests sequential processing optimized for 2-3 files with 5-10k rows each.
    """
    
    def test_process_10k_rows_2_files(self, realistic_files_10k, temp_dir):
        """Test processing 10k rows across 2 files (actual use case)."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        start_time = time.time()
        result = processor.process_files(
            realistic_files_10k,
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        processing_time = time.time() - start_time
        
        # Verify processing succeeded
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 10000
        
        # Verify output files exist
        output_file1 = temp_dir / "output" / "classified_comments_batch1.xlsx"
        output_file2 = temp_dir / "output" / "classified_comments_batch2.xlsx"
        assert output_file1.exists()
        assert output_file2.exists()
        
        # Verify output structure
        df1 = pd.read_excel(output_file1)
        assert 'text' in df1.columns
        assert 'skip_reason' in df1.columns
        assert len(df1) == 5000
        
        # Check category columns exist
        category_cols = [col for col in df1.columns if col not in ['text', 'skip_reason']]
        assert len(category_cols) > 0
        
        print(f"\n10k rows (2 files) processing time: {processing_time:.2f}s")
        print(f"Throughput: {10000/processing_time:.0f} rows/sec")
    
    def test_process_20k_rows_single_file(self, realistic_file_20k, temp_dir):
        """Test processing 20k rows in single file (upper bound use case)."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        start_time = time.time()
        result = processor.process_files(
            [realistic_file_20k],
            output_dir=temp_dir / "output",
            output_prefix="classified_"
        )
        processing_time = time.time() - start_time
        
        # Verify processing succeeded
        assert len(result.successful_files) == 1
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 20000
        
        # Verify output
        output_file = temp_dir / "output" / "classified_comments_large.xlsx"
        assert output_file.exists()
        
        df = pd.read_excel(output_file)
        assert len(df) == 20000
        assert 'skip_reason' in df.columns
        
        print(f"\n20k rows (1 file) processing time: {processing_time:.2f}s")
        print(f"Throughput: {20000/processing_time:.0f} rows/sec")
    
    def test_memory_profiling_10k_rows(self, realistic_files_10k, temp_dir):
        """Test memory usage during 10k row processing."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        # Start memory tracking
        tracemalloc.start()
        baseline_memory = tracemalloc.get_traced_memory()[0]
        
        result = processor.process_files(
            realistic_files_10k,
            output_dir=temp_dir / "output"
        )
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_used_mb = (peak - baseline_memory) / 1024 / 1024
        
        # Verify processing succeeded
        assert result.total_comments_processed == 10000
        
        # Memory should be reasonable (< 500MB for 10k rows)
        assert memory_used_mb < 500, f"Memory usage too high: {memory_used_mb:.1f}MB"
        
        print(f"\nMemory usage for 10k rows: {memory_used_mb:.1f}MB")
        print(f"Per-row memory: {memory_used_mb / 10:.2f}KB")
    
    def test_skip_reason_distribution_realistic(self, realistic_files_10k, temp_dir):
        """Test skip reason distribution with realistic data."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        result = processor.process_files(
            realistic_files_10k[:1],  # Just first file (5k rows)
            output_dir=temp_dir / "output"
        )
        
        # Read results
        output_file = temp_dir / "output" / f"classified_{realistic_files_10k[0].name}"
        df = pd.read_excel(output_file)
        
        # Analyze skip reasons
        skip_counts = df['skip_reason'].value_counts()
        
        print("\nSkip reason distribution (5k rows):")
        for reason, count in skip_counts.items():
            pct = count / len(df) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Should have mix of reasons based on generated data
        assert 'none' in skip_counts  # Most should be processed
        assert skip_counts['none'] > 0.9 * len(df)  # >90% processed


class TestBatchProcessorSmallScale:
    """Quick tests with small datasets for fast validation."""
    
    def test_process_single_file(self, sample_files, temp_dir):
        """Test processing a single small file."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        result = processor.process_files(
            [sample_files[0]],
            output_dir=temp_dir,
            output_prefix="classified_"
        )
        
        assert len(result.successful_files) == 1
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 3
        assert result.success_rate == 100.0
    
    def test_output_files_created(self, sample_files, temp_dir):
        """Test that output files are created correctly."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        output_dir = temp_dir / "output"
        
        result = processor.process_files(
            [sample_files[0]],
            output_dir=output_dir,
            output_prefix="test_"
        )
        
        # Check output file exists
        output_file = output_dir / "test_sample1.xlsx"
        assert output_file.exists()
        
        # Check output contains predictions
        df = pd.read_excel(output_file)
        assert 'skip_reason' in df.columns
        
        # Should have category columns
        category_cols = [col for col in df.columns if col not in ['text', 'skip_reason']]
        assert len(category_cols) > 0
        assert len(df) == 3
    
    def test_combined_output(self, sample_files, temp_dir):
        """Test combined output file creation."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        result = processor.process_files(
            sample_files[:2],
            output_dir=temp_dir,
            combine_results=True
        )
        
        # Check combined file was created
        assert result.combined_output_path is not None
        assert result.combined_output_path.exists()
        
        # Check combined file contents
        df = pd.read_excel(result.combined_output_path)
        assert 'source_file' in df.columns
        assert len(df) == 8  # 3 + 5 comments
        assert 'sample1.xlsx' in df['source_file'].values
        assert 'sample2.xlsx' in df['source_file'].values


class TestBatchProcessorValidation:
    """Tests for file validation functionality."""
    
    def test_validate_files_all_valid(self, sample_files):
        """Test file validation with all valid files."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        valid, errors, cached_data = processor.validate_files(sample_files)
        
        assert len(valid) == 2
        assert len(errors) == 0
        assert len(cached_data) == 2
        for file_path in sample_files:
            assert file_path in cached_data
            assert isinstance(cached_data[file_path], pd.DataFrame)
    
    def test_validate_files_mixed(self, sample_files, empty_file, temp_dir):
        """Test file validation with mixed valid/invalid files."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        fake_file = temp_dir / "fake.xlsx"
        all_files = sample_files + [empty_file, fake_file]
        
        valid, errors, cached_data = processor.validate_files(all_files)
        
        assert len(valid) == 2  # Only the sample files
        assert len(errors) == 2  # Empty file + fake file
        assert len(cached_data) == 2  # Only valid files are cached


class TestBatchProcessorDryRun:
    """Tests for dry-run validation functionality."""
    
    def test_dry_run_validation_10k(self, realistic_files_10k):
        """Test dry-run validation with 10k rows."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        result = processor.dry_run_validation(realistic_files_10k)
        
        # Should return validation results without classification
        assert result['total_files'] == 2
        assert result['total_rows'] == 10000
        assert len(result['files']) == 2
        assert len(result['issues']) == 0
        
        # Each file should have stats
        for file_name, stats in result['files'].items():
            assert 'total_rows' in stats
            assert 'empty_rows' in stats
            assert 'empty_rate' in stats
            assert 'columns' in stats
            assert stats['total_rows'] == 5000


class TestBatchProcessorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_process_with_empty_file(self, empty_file, temp_dir):
        """Test processing empty file."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        result = processor.process_files(
            [empty_file],
            output_dir=temp_dir
        )
        
        assert len(result.successful_files) == 0
        assert len(result.failed_files) == 1
        assert 'No comments' in result.failed_files[0][1] or 'No data' in result.failed_files[0][1]
    
    def test_process_mixed_success_failure(self, sample_files, empty_file, temp_dir):
        """Test processing with both successful and failed files."""
        classifier = get_real_classifier()
        processor = BatchProcessor(classifier)
        
        all_files = sample_files + [empty_file]
        result = processor.process_files(all_files, output_dir=temp_dir)
        
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 1
        assert result.total_comments_processed == 8


class TestExcelIO:
    """Tests for ExcelIO helper class."""
    
    def test_read_write_excel(self, temp_dir):
        """Test reading and writing Excel files."""
        df = pd.DataFrame({'text': ['test1', 'test2']})
        file_path = temp_dir / "test.xlsx"
        
        ExcelIO().save_results(df, file_path)
        assert file_path.exists()
        
        df_read = ExcelIO().read_file(file_path, text_column='text')
        assert len(df_read) == 2
        assert 'text' in df_read.columns
    
    def test_should_chunk_large_csv(self, temp_dir):
        """Test chunking detection for large CSV files."""
        # Create large CSV (>50MB triggers chunking)
        # Each row ~60 bytes, need ~900k rows for 50MB+
        large_csv = temp_dir / "large.csv"
        
        # Write header
        with open(large_csv, 'w', encoding='utf-8') as f:
            f.write("text\n")
            # Write enough data to exceed 50MB (900k rows * 60 bytes ~= 54MB)
            for i in range(900000):
                f.write(f"Comment line {i} with some text to reach size threshold\n")
        
        # Should trigger chunking
        assert ExcelIO.should_chunk(large_csv) == True
    
    def test_should_not_chunk_small_csv(self, temp_dir):
        """Test that small CSV files don't trigger chunking."""
        small_csv = temp_dir / "small.csv"
        df = pd.DataFrame({'text': ['test'] * 100})
        df.to_csv(small_csv, index=False)
        
        assert ExcelIO.should_chunk(small_csv) == False
    
    def test_should_not_chunk_excel(self, temp_dir):
        """Test that Excel files don't trigger chunking."""
        excel_file = temp_dir / "test.xlsx"
        df = pd.DataFrame({'text': ['test'] * 1000})
        ExcelIO().save_results(df, excel_file)
        
        # Excel files never chunk (only CSV)
        assert ExcelIO.should_chunk(excel_file) == False

    """Test BatchProcessingResult class."""
    
    def test_initialization(self):
        """Test result object initialization."""
        result = BatchProcessingResult()
        assert result.successful_files == []
        assert result.failed_files == []
        assert result.total_comments_processed == 0
        assert result.total_skipped_comments == 0
        assert result.processing_time == 0.0
        assert result.combined_output_path is None  # Verify initialization
    
    def test_add_success(self):
        """Test adding successful file result."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test1', 'test2']})
        
        result.add_success('file1.xlsx', df, 2)
        
        assert 'file1.xlsx' in result.successful_files
        assert result.total_comments_processed == 2
        assert 'file1.xlsx' in result.results_by_file
    
    def test_add_failure(self):
        """Test adding failed file result."""
        result = BatchProcessingResult()
        
        result.add_failure('file1.xlsx', 'File not found')
        
        assert len(result.failed_files) == 1
        assert result.failed_files[0] == ('file1.xlsx', 'File not found')
    
    def test_success_rate_all_success(self):
        """Test success rate calculation with all successful."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 1)
        result.add_success('file2.xlsx', df, 1)
        
        assert result.success_rate == 100.0
    
    def test_success_rate_mixed(self):
        """Test success rate calculation with mixed results."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 1)
        result.add_failure('file2.xlsx', 'Error')
        result.add_failure('file3.xlsx', 'Error')
        
        assert result.success_rate == pytest.approx(33.33, rel=0.1)
    
    def test_success_rate_empty(self):
        """Test success rate with no files."""
        result = BatchProcessingResult()
        assert result.success_rate == 0.0
    
    def test_get_summary(self):
        """Test summary generation."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 5, skipped_count=0)
        result.add_failure('file2.xlsx', 'Error')
        result.processing_time = 12.345
        
        summary = result.get_summary()
        
        assert summary['total_files'] == 2
        assert summary['successful_files'] == 1
        assert summary['failed_files'] == 1
        assert summary['total_rows'] == 5  # Updated key name
        assert summary['processed_comments'] == 5  # New key for actual processed
        assert summary['skipped_comments'] == 0
        assert '12.35' in summary['processing_time']
