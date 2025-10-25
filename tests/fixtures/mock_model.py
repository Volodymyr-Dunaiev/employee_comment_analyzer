"""
Mock model fixture for testing without network dependencies.

This module provides a lightweight mock of HuggingFace models
that can be used in tests to avoid network downloads while still
testing integration logic.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class MockBertConfig:
    """Mock BERT configuration."""
    
    def __init__(self, num_labels: int = 3):
        self.num_labels = num_labels
        self.hidden_size = 128
        self.vocab_size = 1000
        self.max_position_embeddings = 512


class MockBertModel(nn.Module):
    """Minimal mock of BERT model for testing.
    
    Returns random logits to simulate model predictions without
    requiring actual pre-trained weights.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Simple linear layer to produce logits
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Create mock hidden state
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        
        # Random hidden states
        hidden_states = torch.randn(batch_size, seq_length, self.config.hidden_size)
        
        # Pool to get single vector per sequence (use [CLS] token)
        pooled = hidden_states[:, 0, :]
        
        # Get logits
        logits = self.classifier(pooled)
        
        # Return in HuggingFace format
        return type('ModelOutput', (), {'logits': logits})()


class MockTokenizer:
    """Mock tokenizer that returns proper tensor format."""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.model_max_length = 512
        
    def __call__(self, texts, truncation=True, padding=True, return_tensors=None, **kwargs):
        """Tokenize texts into mock token IDs."""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        # Generate random token IDs (simulate real tokenization)
        max_len = min(50, self.model_max_length)  # Keep it small for tests
        
        input_ids = torch.randint(0, self.vocab_size, (batch_size, max_len))
        attention_mask = torch.ones(batch_size, max_len)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Wrap in mock batch that has .to() method and supports ** unpacking
        class MockBatch(dict):
            """Dict subclass that also has .to() method for device placement."""
            
            def to(self, device):
                """Mock .to() method for device placement."""
                return self
        
        return MockBatch(result)


def create_mock_model_and_tokenizer(num_labels: int = 3) -> tuple:
    """Create mock model and tokenizer for testing.
    
    Args:
        num_labels: Number of output labels/categories
        
    Returns:
        Tuple of (model, tokenizer) that can be used as drop-in replacement
        for real HuggingFace models in tests.
    """
    config = MockBertConfig(num_labels=num_labels)
    model = MockBertModel(config)
    tokenizer = MockTokenizer()
    
    return model, tokenizer


def mock_transformers_loading(monkeypatch, num_labels: int = 3):
    """Monkeypatch transformers library to return mock models.
    
    Use this in pytest fixtures to avoid network access:
    
    ```python
    @pytest.fixture
    def mock_hf(monkeypatch):
        from tests.fixtures.mock_model import mock_transformers_loading
        mock_transformers_loading(monkeypatch, num_labels=3)
    ```
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        num_labels: Number of categories for the mock model
    """
    model, tokenizer = create_mock_model_and_tokenizer(num_labels)
    
    def mock_from_pretrained(*args, **kwargs):
        """Return mock model regardless of path."""
        return model
    
    def mock_tokenizer_from_pretrained(*args, **kwargs):
        """Return mock tokenizer regardless of path."""
        return tokenizer
    
    # Patch AutoModelForSequenceClassification and AutoTokenizer
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        mock_from_pretrained
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        mock_tokenizer_from_pretrained
    )
