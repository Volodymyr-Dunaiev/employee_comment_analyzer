"""
Tests for batch file processing functionality.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.core.batch_processor import BatchProcessor, BatchProcessingResult, ExcelIO
from src.core.classifier import CommentClassifier


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_files(temp_dir):
    """Create sample Excel files for testing."""
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
    
    # File 3: CSV format
    df3 = pd.DataFrame({
        'text': [
            'Відмінна компанія',
            'Багато роботи'
        ]
    })
    file3 = temp_dir / "sample3.csv"
    df3.to_csv(file3, index=False)
    files.append(file3)
    
    return files


@pytest.fixture
def empty_file(temp_dir):
    """Create empty Excel file."""
    df = pd.DataFrame({'text': []})
    file_path = temp_dir / "empty.xlsx"
    ExcelIO().save_results(df, file_path)
    return file_path


@pytest.fixture
def mock_classifier(mocker):
    """Create mock classifier for testing."""
    classifier = mocker.Mock(spec=CommentClassifier)
    
    # Mock classify_batch to return dummy predictions with metadata and skip_reason
    def mock_classify(texts):
        return {
            'Category1': [0.8] * len(texts),
            'Category2': [0.3] * len(texts),
            'Category3': [0.5] * len(texts),
            'skip_reason': ['none'] * len(texts),  # All processed successfully
            '_metadata': {
                'skipped_indices': [],
                'skip_reason_counts': {'none': len(texts)}
            }
        }
    
    classifier.classify_batch = mock_classify
    return classifier


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


class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_initialization(self, mock_classifier):
        """Test processor initialization."""
        processor = BatchProcessor(mock_classifier, max_workers=5)
        
        assert processor.classifier == mock_classifier
        assert processor.max_workers == 5
        assert processor.text_column == 'text'
    
    def test_process_single_file(self, mock_classifier, sample_files, temp_dir):
        """Test processing a single file."""
        processor = BatchProcessor(mock_classifier, max_workers=1)
        
        result = processor.process_files(
            [sample_files[0]],
            output_dir=temp_dir,
            output_prefix="classified_"
        )
        
        assert len(result.successful_files) == 1
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 3
        assert result.success_rate == 100.0
    
    def test_process_multiple_files(self, mock_classifier, sample_files, temp_dir):
        """Test processing multiple files."""
        processor = BatchProcessor(mock_classifier, max_workers=2)
        
        result = processor.process_files(
            sample_files,
            output_dir=temp_dir,
            output_prefix="batch_"
        )
        
        assert len(result.successful_files) == 3
        assert len(result.failed_files) == 0
        assert result.total_comments_processed == 10  # 3 + 5 + 2
        assert result.processing_time > 0
    
    def test_process_with_empty_file(self, mock_classifier, empty_file, temp_dir):
        """Test processing empty file."""
        processor = BatchProcessor(mock_classifier, max_workers=1)
        
        result = processor.process_files(
            [empty_file],
            output_dir=temp_dir
        )
        
        assert len(result.successful_files) == 0
        assert len(result.failed_files) == 1
        assert 'No comments' in result.failed_files[0][1]
    
    def test_process_mixed_success_failure(self, mock_classifier, sample_files, empty_file, temp_dir):
        """Test processing with both successful and failed files."""
        processor = BatchProcessor(mock_classifier, max_workers=2)
        
        all_files = sample_files + [empty_file]
        result = processor.process_files(all_files, output_dir=temp_dir)
        
        assert len(result.successful_files) == 3
        assert len(result.failed_files) == 1
        assert result.total_comments_processed == 10
    
    def test_output_files_created(self, mock_classifier, sample_files, temp_dir):
        """Test that output files are created correctly."""
        processor = BatchProcessor(mock_classifier, max_workers=1)
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
        assert 'Category1' in df.columns
        assert 'Category2' in df.columns
        assert 'Category3' in df.columns
        assert len(df) == 3
    
    def test_combined_output(self, mock_classifier, sample_files, temp_dir):
        """Test combined output file creation."""
        processor = BatchProcessor(mock_classifier, max_workers=2)
        
        result = processor.process_files(
            sample_files[:2],  # Use first 2 files
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
    
    def test_validate_files_all_valid(self, mock_classifier, sample_files):
        """Test file validation with all valid files."""
        processor = BatchProcessor(mock_classifier)
        
        valid, errors, cached_data = processor.validate_files(sample_files)
        
        assert len(valid) == 3
        assert len(errors) == 0
        assert len(cached_data) == 3  # All files should be cached
        for file_path in sample_files:
            assert file_path in cached_data
            assert isinstance(cached_data[file_path], pd.DataFrame)
    
    def test_validate_files_nonexistent(self, mock_classifier, temp_dir):
        """Test file validation with nonexistent file."""
        processor = BatchProcessor(mock_classifier)
        fake_file = temp_dir / "nonexistent.xlsx"
        
        valid, errors, cached_data = processor.validate_files([fake_file])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert len(cached_data) == 0
        assert 'not found' in errors[0].lower()
    
    def test_validate_files_wrong_format(self, mock_classifier, temp_dir):
        """Test file validation with wrong file format."""
        processor = BatchProcessor(mock_classifier)
        
        # Create text file
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test data")
        
        valid, errors, cached_data = processor.validate_files([txt_file])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert len(cached_data) == 0
        assert 'Unsupported format' in errors[0]
    
    def test_validate_files_empty_data(self, mock_classifier, empty_file):
        """Test file validation with empty file."""
        processor = BatchProcessor(mock_classifier)
        
        valid, errors, cached_data = processor.validate_files([empty_file])
        
        assert len(valid) == 0
        assert len(errors) == 1
        assert len(cached_data) == 0
        assert 'No data' in errors[0]
    
    def test_validate_files_mixed(self, mock_classifier, sample_files, empty_file, temp_dir):
        """Test file validation with mixed valid/invalid files."""
        processor = BatchProcessor(mock_classifier)
        
        fake_file = temp_dir / "fake.xlsx"
        all_files = sample_files + [empty_file, fake_file]
        
        valid, errors, cached_data = processor.validate_files(all_files)
        
        assert len(valid) == 3  # Only the sample files
        assert len(errors) == 2  # Empty file + fake file
        assert len(cached_data) == 3  # Only valid files are cached
    
    def test_different_output_prefix(self, mock_classifier, sample_files, temp_dir):
        """Test using different output prefix."""
        processor = BatchProcessor(mock_classifier, max_workers=1)
        
        result = processor.process_files(
            [sample_files[0]],
            output_dir=temp_dir,
            output_prefix="result_"
        )
        
        output_file = temp_dir / "result_sample1.xlsx"
        assert output_file.exists()
    
    def test_parallel_processing(self, mock_classifier, sample_files, temp_dir):
        """Test that parallel processing works correctly."""
        processor = BatchProcessor(mock_classifier, max_workers=3)
        
        result = processor.process_files(
            sample_files,
            output_dir=temp_dir
        )
        
        # All files should be processed
        assert len(result.successful_files) == 3
        assert result.total_comments_processed == 10
    
    def test_csv_file_support(self, mock_classifier, sample_files, temp_dir):
        """Test processing CSV files."""
        processor = BatchProcessor(mock_classifier, max_workers=1)
        
        # sample_files[2] is CSV
        result = processor.process_files(
            [sample_files[2]],
            output_dir=temp_dir
        )
        
        assert len(result.successful_files) == 1
        assert result.total_comments_processed == 2


class TestBatchProcessorIntegration:
    """Integration tests with real classifier (if model available)."""
    
    @pytest.mark.skipif(
        not Path("model/ukr_multilabel").exists(),
        reason="Model not available"
    )
    def test_real_classifier_integration(self, sample_files, temp_dir):
        """Test with real classifier model."""
        from src.utils.config import Config
        
        config = Config()
        classifier = CommentClassifier(config)
        processor = BatchProcessor(classifier, max_workers=1)
        
        result = processor.process_files(
            [sample_files[0]],
            output_dir=temp_dir
        )
        
        assert len(result.successful_files) == 1
        assert result.total_comments_processed > 0
        
        # Check predictions are realistic
        df = pd.read_excel(temp_dir / f"classified_{sample_files[0].name}")
        
        # Should have category columns
        category_cols = [col for col in df.columns if col not in ['text']]
        assert len(category_cols) > 0
        
        # Probabilities should be between 0 and 1
        for col in category_cols:
            assert df[col].min() >= 0
            assert df[col].max() <= 1
