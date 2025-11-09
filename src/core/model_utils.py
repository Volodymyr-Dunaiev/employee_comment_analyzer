# Utility functions for model loading and management.
#
# This module provides centralized functions for loading Hugging Face models
# and tokenizers to avoid circular dependencies.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_path):
    # Load a Hugging Face model and tokenizer from a saved directory.
    #
    # This function loads both the tokenizer and model from a directory
    # containing a previously trained or downloaded model. It's used
    # throughout the application for consistent model loading.
    #
    # Args:
    #     model_path (str): Path to the directory containing saved model files
    #         Must contain config.json, pytorch_model.bin, tokenizer files, etc.
    #
    # Returns:
    #     tuple: (tokenizer, model) pair where:
    #         - tokenizer: PreTrainedTokenizer instance
    #         - model: PreTrainedModel instance ready for inference
    #
    # Raises:
    #     ValueError: If model or tokenizer cannot be loaded (missing files,
    #         corrupted data, incompatible versions)
    #
    # Example:
    #     >>> tokenizer, model = load_model_and_tokenizer('./models/my_model')
    #     >>> inputs = tokenizer("Hello world", return_tensors="pt")
    #     >>> outputs = model(**inputs)
    
    try:
        # Force local_files_only=True to prevent any internet access
        # This ensures the model must exist locally or loading will fail
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"Failed to load model and tokenizer: {str(e)}")