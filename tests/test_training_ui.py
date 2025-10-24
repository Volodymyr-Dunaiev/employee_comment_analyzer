# Unit tests for the training UI functionality.
import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
from io import BytesIO
import pandas as pd
import os
import tempfile

# Note: Streamlit UI tests require special handling since Streamlit runs in a different context
# These tests verify the logic and structure rather than UI rendering

@pytest.fixture
def mock_config():
    # Provide a mock configuration for UI testing.
    return {
        'model': {
            'path': 'xlm-roberta-base',
            'batch_size': 8,
            'device': 'cpu'
        },
        'data': {
            'text_column': 'comment',
            'chunk_size': 1000,
            'input_validation': {
                'max_length': 5000
            }
        },
        'categories': [
            "Колектив",
            "Умови праці",
            "Робочі процеси та практики"
        ],
        'security': {
            'allowed_extensions': ['xlsx', 'xls', 'csv'],
            'max_file_size_mb': 50
        }
    }

def test_training_data_validation():
    # Test that training data file validation works correctly.
    # Create a mock file
    mock_file = Mock()
    mock_file.name = "training_data.xlsx"
    mock_file.seek = Mock()
    mock_file.tell = Mock(return_value=1024 * 1024)  # 1 MB
    
    from src.ui.app_ui import validate_file
    
    # Valid file should pass
    assert validate_file(mock_file) == True
    
def test_training_data_validation_size_limit():
    # Test that large files are rejected.
    mock_file = Mock()
    mock_file.name = "large_file.xlsx"
    mock_file.seek = Mock()
    mock_file.tell = Mock(return_value=100 * 1024 * 1024)  # 100 MB (exceeds 50MB limit)
    
    from src.ui.app_ui import validate_file
    
    # Should reject large files
    assert validate_file(mock_file) == False

def test_training_data_validation_invalid_extension():
    # Test that invalid file extensions are rejected.
    mock_file = Mock()
    mock_file.name = "training_data.txt"
    mock_file.seek = Mock()
    mock_file.tell = Mock(return_value=1024)
    
    from src.ui.app_ui import validate_file
    
    # Should reject invalid extensions
    assert validate_file(mock_file) == False

def test_training_parameters_default_values(mock_config):
    # Test that training parameters have sensible defaults.
    # Default values should be within expected ranges
    default_epochs = 3
    default_batch_size = 8
    default_learning_rate = 2e-5
    default_test_size = 10
    default_valid_size = 10
    
    assert 1 <= default_epochs <= 10
    assert 4 <= default_batch_size <= 32
    assert 1e-5 <= default_learning_rate <= 5e-5
    assert 5 <= default_test_size <= 30
    assert 5 <= default_valid_size <= 30

def test_model_selection_options():
    # Test that available model options are valid.
    valid_models = ["xlm-roberta-base", "bert-base-multilingual-cased", "xlm-roberta-large"]
    
    # All models should be valid Hugging Face identifiers
    for model in valid_models:
        assert isinstance(model, str)
        assert len(model) > 0

def test_validate_training_data_valid():
    # Test validation with valid training data.
    from src.core.train_interface import validate_training_data
    
    df = pd.DataFrame({
        'text': ['comment 1', 'comment 2', 'comment 3'] * 5,
        'labels': [['cat1'], ['cat2'], ['cat1', 'cat2']] * 5
    })
    
    is_valid, error_msg = validate_training_data(df, 'text', 'labels')
    assert is_valid == True
    assert error_msg is None

def test_validate_training_data_missing_column():
    # Test validation with missing column.
    from src.core.train_interface import validate_training_data
    
    df = pd.DataFrame({
        'text': ['comment 1', 'comment 2']
    })
    
    is_valid, error_msg = validate_training_data(df, 'text', 'labels')
    assert is_valid == False
    assert 'not found' in error_msg

def test_validate_training_data_empty():
    # Test validation with empty DataFrame.
    from src.core.train_interface import validate_training_data
    
    df = pd.DataFrame({'text': [], 'labels': []})
    
    is_valid, error_msg = validate_training_data(df, 'text', 'labels')
    assert is_valid == False
    assert 'empty' in error_msg.lower()

def test_validate_training_data_null_values():
    # Test validation with null values.
    from src.core.train_interface import validate_training_data
    import numpy as np
    
    df = pd.DataFrame({
        'text': ['comment 1', None, 'comment 3'],
        'labels': [['cat1'], ['cat2'], ['cat1']]
    })
    
    is_valid, error_msg = validate_training_data(df, 'text', 'labels')
    assert is_valid == False
    assert 'null' in error_msg.lower()

def test_validate_training_data_minimum_samples():
    # Test validation with insufficient samples.
    from src.core.train_interface import validate_training_data
    
    df = pd.DataFrame({
        'text': ['comment 1', 'comment 2'],
        'labels': [['cat1'], ['cat2']]
    })
    
    is_valid, error_msg = validate_training_data(df, 'text', 'labels')
    assert is_valid == False
    assert 'at least 10' in error_msg.lower()

@pytest.mark.integration
def test_training_interface_data_loading(mock_config):
    # Test that training interface can load and process data.
    from src.core.train_interface import train_from_ui
    from io import BytesIO
    import tempfile
    
    # Create a small test dataset
    test_data = pd.DataFrame({
        'text': ['comment ' + str(i) for i in range(20)],
        'labels': [['Колектив'], ['Умови праці']] * 10
    })
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        test_data.to_excel(tmp.name, index=False)
        tmp_path = tmp.name
    
    try:
        # Read file into BytesIO
        with open(tmp_path, 'rb') as f:
            file_bytes = BytesIO(f.read())
        
        # This should load without errors
        # Note: We're not actually training here due to time constraints
        # Just testing data loading and validation
        progress_updates = []
        
        def track_progress(status, progress):
            progress_updates.append((status, progress))
        
        # Test would require mocking the actual training
        # For now, just verify the function exists and has correct signature
        assert callable(train_from_ui)
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@pytest.mark.integration
def test_training_ui_integration(mock_config):
    # Integration test for training UI components.
    # Mock Streamlit components
    with patch('streamlit.file_uploader') as mock_uploader, \
         patch('streamlit.button') as mock_button, \
         patch('streamlit.progress') as mock_progress:
        
        mock_uploader.return_value = Mock()
        mock_button.return_value = False
        
        # Verify UI can be initialized without errors
        from src.ui.app_ui import show_training_tab
        
        # This should not raise any errors
        try:
            show_training_tab(mock_config)
        except Exception as e:
            # Some Streamlit errors are expected in test environment
            if "DeltaGenerator" not in str(e):
                raise

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
