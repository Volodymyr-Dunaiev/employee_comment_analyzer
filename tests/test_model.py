# Tests for the CommentClassifier model.
# Verifies that the classifier can make predictions correctly.

import torch
import pytest
from src.core.classifier import CommentClassifier
import pandas as pd

@pytest.fixture
def mock_config():
    # Configuration fixture for testing classifier.
    return {
        'model': {
            'path': 'xlm-roberta-base',
            'batch_size': 2,
            'device': 'cpu'
        },
        'data': {
            'input_validation': {
                'max_length': 5000
            }
        }
    }

def test_classifier_predictions(mock_config, monkeypatch):
    # Test that classifier produces valid predictions.
    dummy_df = pd.DataFrame({'text': ['Привіт', 'Як справи?']})
    categories = ['Колектив', 'Умови праці']
    
    # Create a mock tokenizer class
    class MockTokenizer:
        def __init__(self, *args, **kwargs):
            pass
            
        def __call__(self, texts, **kwargs):
            class TensorDict:
                def __init__(self):
                    self.data = {
                        'input_ids': torch.tensor([[1, 1], [1, 1]]),
                        'attention_mask': torch.tensor([[1, 1], [1, 1]])
                    }
                
                def to(self, device):
                    return self.data
                
                def __getitem__(self, key):
                    return self.data[key]
            
            return TensorDict()
    
    # Create a mock model
    class MockModel:
        def eval(self): pass
        def to(self, device): return self
        def __call__(self, **kwargs): 
            return type('obj', (object,), {'logits': torch.tensor([[0.9, 0.1], [0.1, 0.9]])})()

    # Apply mocks
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args: MockTokenizer())
    monkeypatch.setattr('transformers.AutoModelForSequenceClassification.from_pretrained', lambda *args: MockModel())

    # Initialize classifier and test predictions
    classifier = CommentClassifier(mock_config)
    predictions = classifier.predict_comments(dummy_df, 'text', categories)
    
    assert len(predictions) == 2
    assert isinstance(predictions[0], list)
    assert isinstance(predictions[1], list)
