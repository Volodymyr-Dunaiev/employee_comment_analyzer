# Shared fixtures for tests.
import pytest
import pandas as pd

@pytest.fixture
def mock_config():
    # Provide a mock configuration for testing.
    return {
        'model': {
            'path': 'tests/mock_model',
            'batch_size': 2,
            'device': 'cpu'
        },
        'data': {
            'text_column': 'text',
            'chunk_size': 100
        },
        'categories': ['cat1', 'cat2']
    }

@pytest.fixture
def mock_data():
    # Provide mock input data for testing.
    return pd.DataFrame({
        'text': ['Test comment 1', 'Test comment 2'],
        'other_col': [1, 2]
    })