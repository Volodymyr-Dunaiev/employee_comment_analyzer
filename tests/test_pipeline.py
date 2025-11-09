# Unit tests for the pipeline module.
import pytest
import pandas as pd
import asyncio
import torch
from src.core.classifier import CommentClassifier
from src.core.errors import PipelineError
from src.utils.config import load_config

@pytest.fixture
def mock_config():
    # Provide a mock configuration for testing.
    return {
        'model': {
            'path': 'xlm-roberta-base',
            'batch_size': 2,
            'device': 'cpu'
        },
        'data': {
            'text_column': 'text',
            'chunk_size': 100,
            'input_validation': {
                'max_length': 5000
            }
        },
        'categories': ['cat1', 'cat2']
    }

@pytest.fixture
def mock_tokenizer():
    # Mock tokenizer for testing without loading real models.
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            pass
            
        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            batch_size = len(texts)
            
            class TensorDict:
                def __init__(self, batch_size):
                    self.data = {
                        'input_ids': torch.tensor([[1, 1]] * batch_size),
                        'attention_mask': torch.tensor([[1, 1]] * batch_size)
                    }
                
                def to(self, device):
                    return self.data
                
                def __getitem__(self, key):
                    return self.data[key]
            
            return TensorDict(batch_size)
    return MockTokenizer

@pytest.fixture
def mock_model():
    # Mock model for testing without loading real models.
    class MockModel:
        def eval(self): 
            pass
            
        def to(self, device): 
            return self
            
        def __call__(self, **kwargs): 
            batch_size = kwargs['input_ids'].shape[0]
            return type('obj', (object,), {
                'logits': torch.tensor([[0.9, 0.1]] * batch_size)
            })()
    return MockModel

@pytest.fixture
def mock_data():
    # Provide mock input data for testing.
    return pd.DataFrame({
        'text': ['Test comment 1', 'Test comment 2'],
        'other_col': [1, 2]
    })

def test_classifier_initialization(mock_config, mock_tokenizer, mock_model, monkeypatch):
    # Test classifier initialization with valid config.
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer())
    monkeypatch.setattr('transformers.AutoModelForSequenceClassification.from_pretrained', lambda *args, **kwargs: mock_model())
    classifier = CommentClassifier(mock_config)
    assert classifier.config == mock_config

def test_invalid_config(mock_tokenizer, mock_model, monkeypatch):
    # Test classifier initialization with invalid config.
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer())
    monkeypatch.setattr('transformers.AutoModelForSequenceClassification.from_pretrained', lambda *args, **kwargs: mock_model())
    invalid_config = {
        'model': {
            'path': 'xlm-roberta-base',
            'batch_size': 2
        }
    }
    with pytest.raises(KeyError):
        CommentClassifier(invalid_config)

def test_predict_comments_invalid_column(mock_config, mock_data, mock_tokenizer, mock_model, monkeypatch):
    # Test prediction with invalid column name.
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer())
    monkeypatch.setattr('transformers.AutoModelForSequenceClassification.from_pretrained', lambda *args, **kwargs: mock_model())
    classifier = CommentClassifier(mock_config)
    with pytest.raises(PipelineError):
        classifier.predict_comments(mock_data, 'nonexistent_column', mock_config['categories'])

def test_predict_comments_empty_data(mock_config, mock_tokenizer, mock_model, monkeypatch):
    # Test prediction with empty DataFrame.
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer())
    monkeypatch.setattr('transformers.AutoModelForSequenceClassification.from_pretrained', lambda *args, **kwargs: mock_model())
    classifier = CommentClassifier(mock_config)
    empty_df = pd.DataFrame({'text': []})
    predictions = classifier.predict_comments(empty_df, 'text', mock_config['categories'])
    assert len(predictions) == 0

def test_predict_comments_batch(mock_config, mock_data, mock_tokenizer, mock_model, monkeypatch):
    # Test batch prediction.
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: mock_tokenizer())
    monkeypatch.setattr('transformers.AutoModelForSequenceClassification.from_pretrained', lambda *args, **kwargs: mock_model())
    classifier = CommentClassifier(mock_config)
    predictions = classifier.predict_comments(mock_data, 'text', mock_config['categories'])
    assert len(predictions) == len(mock_data)
    assert all(isinstance(pred, list) for pred in predictions)