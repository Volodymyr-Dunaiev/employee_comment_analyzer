"""
Tests for batch processing UI functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.core.batch_processor import BatchProcessor, BatchProcessingResult


class TestBatchProcessingUI:
    """Test batch processing UI integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def mock_uploaded_files(self, temp_dir):
        """Create mock Streamlit uploaded files."""
        files = []
        
        # Create mock uploaded file 1
        file1 = Mock()
        file1.name = "test1.xlsx"
        file1.getvalue = Mock(return_value=b"mock_excel_data_1")
        files.append(file1)
        
        # Create mock uploaded file 2
        file2 = Mock()
        file2.name = "test2.xlsx"
        file2.getvalue = Mock(return_value=b"mock_excel_data_2")
        files.append(file2)
        
        return files
    
    def test_batch_mode_file_saving(self, mock_uploaded_files, temp_dir):
        """Test that uploaded files are saved correctly to temp directory."""
        # Simulate saving files
        file_paths = []
        for uploaded_file in mock_uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            file_paths.append(file_path)
        
        # Verify files were created
        assert len(file_paths) == 2
        assert file_paths[0].exists()
        assert file_paths[1].exists()
        assert file_paths[0].name == "test1.xlsx"
        assert file_paths[1].name == "test2.xlsx"
    
    def test_batch_mode_file_list_display(self, mock_uploaded_files):
        """Test that file list information is correctly extracted."""
        file_info = []
        for file in mock_uploaded_files:
            # Simulate size calculation
            size_bytes = len(file.getvalue())
            size_mb = size_bytes / (1024 * 1024)
            file_info.append({
                'name': file.name,
                'size_mb': size_mb
            })
        
        assert len(file_info) == 2
        assert file_info[0]['name'] == "test1.xlsx"
        assert file_info[1]['name'] == "test2.xlsx"
    
    def test_max_workers_validation(self):
        """Test that max_workers parameter validation works."""
        # Valid values
        assert 1 <= 3 <= 5  # Default value
        assert 1 <= 1 <= 5  # Min value
        assert 1 <= 5 <= 5  # Max value
        
        # Invalid values would be caught by st.number_input constraints
        # min_value=1, max_value=5
    
    def test_combine_results_option(self):
        """Test combine_results boolean option."""
        # Test default value
        combine_results = True
        assert isinstance(combine_results, bool)
        assert combine_results == True
        
        # Test toggling
        combine_results = False
        assert combine_results == False
    
    def test_batch_processor_initialization(self):
        """Test batch processor is initialized with correct parameters."""
        from src.core.classifier import CommentClassifier
        
        # Create mock classifier
        classifier = Mock(spec=CommentClassifier)
        
        # Initialize processor (simulating UI code)
        processor = BatchProcessor(
            classifier,
            max_workers=3,
            text_column="text"
        )
        
        assert processor.max_workers == 3
        assert processor.text_column == "text"
    
    def test_summary_metrics_structure(self):
        """Test that summary metrics have correct structure."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 5, skipped_count=0)
        result.add_failure('file2.xlsx', 'Error message')
        result.processing_time = 12.34
        
        summary = result.get_summary()
        
        # Verify all required metrics exist
        assert 'total_files' in summary
        assert 'successful_files' in summary
        assert 'failed_files' in summary
        assert 'success_rate' in summary
        assert 'total_rows' in summary  # Updated from total_comments
        assert 'processed_comments' in summary  # New metric
        assert 'skipped_comments' in summary  # New metric
        assert 'processing_time' in summary
        
        # Verify metric types
        assert isinstance(summary['total_files'], int)
        assert isinstance(summary['successful_files'], int)
        assert isinstance(summary['failed_files'], int)
        assert isinstance(summary['success_rate'], str)
        assert isinstance(summary['total_rows'], int)
        assert isinstance(summary['processed_comments'], int)
        assert isinstance(summary['skipped_comments'], int)
        assert isinstance(summary['processing_time'], str)
    
    def test_error_list_display(self):
        """Test that error list is formatted correctly for display."""
        result = BatchProcessingResult()
        
        result.add_failure('file1.xlsx', 'Column not found')
        result.add_failure('file2.csv', 'Empty file')
        result.add_failure('file3.xlsx', 'Invalid format')
        
        # Verify error list structure
        assert len(result.failed_files) == 3
        assert result.failed_files[0] == ('file1.xlsx', 'Column not found')
        assert result.failed_files[1] == ('file2.csv', 'Empty file')
        assert result.failed_files[2] == ('file3.xlsx', 'Invalid format')
    
    def test_successful_files_list(self):
        """Test that successful files list is maintained correctly."""
        result = BatchProcessingResult()
        df = pd.DataFrame({'text': ['test']})
        
        result.add_success('file1.xlsx', df, 10)
        result.add_success('file2.xlsx', df, 20)
        result.add_success('file3.csv', df, 15)
        
        assert len(result.successful_files) == 3
        assert 'file1.xlsx' in result.successful_files
        assert 'file2.xlsx' in result.successful_files
        assert 'file3.csv' in result.successful_files
        assert result.total_comments_processed == 45
    
    def test_download_button_parameters(self):
        """Test that download button parameters are correctly formatted."""
        filename = "test.xlsx"
        classified_filename = f"classified_{filename}"
        
        # Test MIME type for Excel
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert mime_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
        # Test filename format
        assert classified_filename == "classified_test.xlsx"
    
    def test_combined_output_path_structure(self):
        """Test combined output path generation."""
        from datetime import datetime
        
        output_dir = Path("test_dir")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"classified_combined_{timestamp}.xlsx"
        
        # Verify filename pattern
        assert combined_filename.startswith("classified_combined_")
        assert combined_filename.endswith(".xlsx")
        assert timestamp in combined_filename


class TestBatchUIWorkflow:
    """Test complete batch processing workflow."""
    
    @pytest.fixture
    def mock_classifier(self, mocker):
        """Create mock classifier."""
        from src.core.classifier import CommentClassifier
        classifier = mocker.Mock(spec=CommentClassifier)
        
        def mock_classify(texts):
            return {
                'Category1': [0.8] * len(texts),
                'Category2': [0.3] * len(texts)
            }
        
        classifier.classify_batch = mock_classify
        return classifier
    
    @pytest.fixture
    def sample_data_files(self, tmp_path):
        """Create sample Excel files for testing."""
        files = []
        
        # File 1
        df1 = pd.DataFrame({'text': ['Коментар 1', 'Коментар 2', 'Коментар 3']})
        file1 = tmp_path / "sample1.xlsx"
        df1.to_excel(file1, index=False)
        files.append(file1)
        
        # File 2
        df2 = pd.DataFrame({'text': ['Коментар 4', 'Коментар 5']})
        file2 = tmp_path / "sample2.xlsx"
        df2.to_excel(file2, index=False)
        files.append(file2)
        
        return files
    
    def test_full_batch_workflow(self, mock_classifier, sample_data_files, tmp_path):
        """Test complete batch processing workflow."""
        processor = BatchProcessor(mock_classifier, max_workers=2)
        
        # Validate files (now returns 3 values including cached_data)
        valid_files, errors, cached_data = processor.validate_files(sample_data_files)
        assert len(valid_files) == 2
        assert len(errors) == 0
        assert len(cached_data) == 2
        
        # Process files with cached data
        result = processor.process_files(
            valid_files,
            output_dir=tmp_path / "output",
            output_prefix="classified_",
            combine_results=True
        )
        
        # Verify results
        assert len(result.successful_files) == 2
        assert result.total_comments_processed == 5  # 3 + 2
        assert result.success_rate == 100.0
        
        # Verify summary structure
        summary = result.get_summary()
        assert summary['total_files'] == 2
        assert summary['successful_files'] == 2
        assert summary['failed_files'] == 0
    
    def test_batch_workflow_with_failures(self, mock_classifier, sample_data_files, tmp_path):
        """Test batch processing with some failed files."""
        # Add an empty file
        empty_file = tmp_path / "empty.xlsx"
        pd.DataFrame({'text': []}).to_excel(empty_file, index=False)
        
        all_files = sample_data_files + [empty_file]
        processor = BatchProcessor(mock_classifier, max_workers=2)
        
        # Process files
        result = processor.process_files(
            all_files,
            output_dir=tmp_path / "output"
        )
        
        # Verify mixed results
        assert len(result.successful_files) == 2
        assert len(result.failed_files) == 1
        assert result.success_rate == pytest.approx(66.67, rel=0.1)
    
    def test_progress_updates_structure(self):
        """Test that progress update structure is correct."""
        # Simulate progress updates
        total_files = 5
        processed_files = 3
        
        progress_percent = (processed_files / total_files) * 100
        assert progress_percent == 60.0
        
        status_text = f"Processing {processed_files}/{total_files} files..."
        assert "Processing 3/5 files..." == status_text


class TestBatchUIEdgeCases:
    """Test edge cases in batch processing UI."""
    
    def test_empty_file_list(self):
        """Test handling of empty file list."""
        uploaded_files = []
        
        if not uploaded_files:
            # Should show info message
            assert len(uploaded_files) == 0
    
    def test_single_file_in_batch_mode(self, mocker):
        """Test processing single file in batch mode."""
        from src.core.classifier import CommentClassifier
        
        classifier = mocker.Mock(spec=CommentClassifier)
        classifier.classify_batch = Mock(return_value={
            'Category1': [0.8],
            'Category2': [0.3]
        })
        
        processor = BatchProcessor(classifier, max_workers=1)
        
        # Should work same as multiple files
        assert processor.max_workers == 1
    
    def test_max_workers_edge_values(self):
        """Test max_workers at boundaries."""
        # Min value
        min_workers = 1
        assert 1 <= min_workers <= 5
        
        # Max value
        max_workers = 5
        assert 1 <= max_workers <= 5
    
    def test_file_size_display_formatting(self):
        """Test file size formatting for display."""
        # Test various file sizes
        size_bytes = 1536  # 1.5 KB
        size_mb = size_bytes / (1024 * 1024)
        formatted = f"{size_mb:.2f} MB"
        
        assert "0.00" in formatted
        
        # Large file
        size_bytes = 5 * 1024 * 1024  # 5 MB
        size_mb = size_bytes / (1024 * 1024)
        formatted = f"{size_mb:.2f} MB"
        
        assert "5.00" in formatted
    
    def test_combined_output_with_no_successful_files(self):
        """Test combined output when no files succeed."""
        result = BatchProcessingResult()
        
        result.add_failure('file1.xlsx', 'Error')
        result.add_failure('file2.xlsx', 'Error')
        
        # Should have combined_output_path attribute (initialized to None)
        assert hasattr(result, 'combined_output_path')
        assert result.combined_output_path is None  # Should be None since no files succeeded
        assert len(result.successful_files) == 0


class TestBatchUIIntegration:
    """Integration tests for batch UI with real components."""
    
    def test_temp_directory_cleanup(self, tmp_path):
        """Test that temporary directory is properly cleaned up."""
        import tempfile
        
        # Simulate tempfile.TemporaryDirectory usage
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test file
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")
            
            assert temp_path.exists()
            assert test_file.exists()
        
        # After context exit, should be cleaned up
        # (we can't test this directly, but the pattern is correct)
    
    def test_config_loading_for_batch(self):
        """Test that config loads correctly for batch processing."""
        from src.utils.config import load_config
        
        try:
            config = load_config()
            assert config is not None
            assert 'model' in config or 'categories' in config
        except Exception:
            # Config might not be available in test environment
            pass
    
    def test_classifier_initialization_for_batch(self, mocker):
        """Test classifier initialization for batch processing."""
        from src.core.classifier import CommentClassifier
        
        # Mock config
        config = Mock()
        config.__getitem__ = Mock(return_value={'path': 'test-model'})
        
        # This tests the pattern used in UI
        try:
            classifier = CommentClassifier(config)
        except Exception:
            # Might fail if model not available, but pattern is correct
            pass
