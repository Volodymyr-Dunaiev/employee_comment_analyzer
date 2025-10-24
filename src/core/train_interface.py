# Training interface wrapper for Streamlit UI.
# Connects the UI parameters to the training pipeline.

import os
import json
import tempfile
from typing import Optional, Callable
from io import BytesIO
import pandas as pd
from src.core.train import (
    load_dataframe,
    build_datasets,
    tokenize_dataset,
    get_compute_metrics
)
from src.utils.logger import get_logger
from sklearn.preprocessing import MultiLabelBinarizer

logger = get_logger(__name__)


def train_from_ui(
    training_file: BytesIO,
    text_column: str,
    labels_column: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    test_size: float,
    valid_size: float,
    output_dir: str,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> dict:
    # Train a multi-label classification model from UI parameters.
    #
    # This function orchestrates the complete training pipeline:
    # 1. Loads and validates training data
    # 2. Splits data into train/validation/test sets
    # 3. Tokenizes text and prepares datasets
    # 4. Trains the model with progress tracking
    # 5. Evaluates on test set
    # 6. Saves trained model and metadata
    #
    # Args:
    #     training_file: BytesIO object containing training data (Excel/CSV)
    #     text_column: Name of column containing text to classify
    #     labels_column: Name of column containing comma-separated labels
    #     model_name: Hugging Face model identifier (e.g., 'xlm-roberta-base')
    #     epochs: Number of training epochs (typically 3-10)
    #     batch_size: Batch size for training (adjust based on GPU memory)
    #     learning_rate: Learning rate for optimizer (typically 2e-5 to 5e-5)
    #     test_size: Proportion of data for testing (0-1, e.g., 0.15)
    #     valid_size: Proportion of data for validation (0-1, e.g., 0.15)
    #     output_dir: Directory to save trained model and metadata
    #     progress_callback: Optional function(message, progress) for UI updates
    #         - message (str): Current operation description
    #         - progress (float): Progress value between 0 and 1
    #
    # Returns:
    #     dict: Training results with the following structure:
    #         - 'status': 'success' or 'error'
    #         - 'model_path': Path to saved model directory
    #         - 'metrics': Test set evaluation metrics (F1, precision, recall)
    #         - 'num_labels': Number of unique categories
    #         - 'labels': List of category names
    #         - 'epochs': Number of epochs trained
    #         - 'message': Success or error message
    #
    # Raises:
    #     ValueError: If data validation fails or invalid parameters
    #     Exception: If training pipeline encounters errors
    #
    # Example:
    #     >>> from io import BytesIO
    #     >>> file = BytesIO(open('train_data.xlsx', 'rb').read())
    #     >>> results = train_from_ui(
    #     ...     file, 'comment', 'categories', 'xlm-roberta-base',
    #     ...     epochs=3, batch_size=16, learning_rate=2e-5,
    #     ...     test_size=0.15, valid_size=0.15, output_dir='./models'
    #     ... )
    #     >>> print(results['metrics']['f1'])
    
    try:
        if progress_callback:
            progress_callback("Loading training data...", 0.1)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(training_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load and validate data
            df = load_dataframe(tmp_path, text_column, labels_column)
            logger.info(f"Loaded {len(df)} training samples")
            
            if progress_callback:
                progress_callback(f"Loaded {len(df)} samples", 0.2)
            
            # Prepare labels
            if progress_callback:
                progress_callback("Preparing labels...", 0.3)
            
            mlb = MultiLabelBinarizer()
            mlb.fit(df['labels'])
            num_labels = len(mlb.classes_)
            
            logger.info(f"Found {num_labels} unique labels: {mlb.classes_}")
            
            # Build datasets
            if progress_callback:
                progress_callback("Splitting dataset...", 0.4)
            
            datasets = build_datasets(df, mlb, test_size, valid_size)
            
            # Tokenize
            if progress_callback:
                progress_callback("Tokenizing data...", 0.5)
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenized_datasets = tokenize_dataset(datasets, tokenizer)
            
            # Configure training
            if progress_callback:
                progress_callback("Configuring training...", 0.6)
            
            training_args = {
                'output_dir': output_dir,
                'num_train_epochs': epochs,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'learning_rate': learning_rate,
                'evaluation_strategy': 'epoch',
                'save_strategy': 'epoch',
                'load_best_model_at_end': True,
                'metric_for_best_model': 'micro_f1',
                'save_total_limit': 2,
                'logging_steps': 10,
            }
            
            # Initialize model
            if progress_callback:
                progress_callback("Initializing model...", 0.6)
            
            from transformers import (
                AutoModelForSequenceClassification,
                TrainingArguments,
                Trainer,
                EarlyStoppingCallback
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="multi_label_classification"
            )
            
            # Configure training arguments
            if progress_callback:
                progress_callback("Configuring training...", 0.65)
            
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                evaluation_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='micro_f1',
                save_total_limit=2,
                logging_steps=10,
                logging_dir=f"{output_dir}/logs",
                report_to='none',  # Disable wandb/tensorboard
                weight_decay=0.01,
                warmup_ratio=0.1,
            )
            
            # Create trainer with metrics
            compute_metrics = get_compute_metrics()
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train model
            if progress_callback:
                progress_callback("Training model...", 0.7)
            
            logger.info("Starting training...")
            train_result = trainer.train()
            
            if progress_callback:
                progress_callback("Evaluating model...", 0.9)
            
            # Evaluate on test set
            test_metrics = trainer.evaluate(tokenized_datasets['test'])
            logger.info(f"Test metrics: {test_metrics}")
            
            # Save model and tokenizer
            if progress_callback:
                progress_callback("Saving model...", 0.95)
            
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Save label encoder
            import joblib
            label_encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
            joblib.dump(mlb, label_encoder_path)
            
            # Save training metadata
            metadata = {
                'model_name': model_name,
                'num_labels': num_labels,
                'labels': list(mlb.classes_),
                'num_samples': len(df),
                'train_samples': len(datasets['train']),
                'valid_samples': len(datasets['validation']),
                'test_samples': len(datasets['test']),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'train_loss': float(train_result.training_loss),
            }
            
            metadata_path = os.path.join(output_dir, 'training_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            results = {
                'status': 'success',
                'model_path': output_dir,
                'num_samples': len(df),
                'num_labels': num_labels,
                'labels': list(mlb.classes_),
                'train_samples': len(datasets['train']),
                'valid_samples': len(datasets['validation']),
                'test_samples': len(datasets['test']),
                'test_metrics': test_metrics,
                'train_loss': train_result.training_loss,
            }
            
            if progress_callback:
                progress_callback("Training complete!", 1.0)
            
            logger.info("Training completed successfully")
            return results
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.exception("Training failed")
        raise Exception(f"Training failed: {str(e)}")


def validate_training_data(
    df: pd.DataFrame,
    text_column: str,
    labels_column: str
) -> tuple[bool, Optional[str]]:
    # Validate training data format and content before training.
    #
    # Performs comprehensive validation checks:
    # - Required columns exist
    # - Data is not empty
    # - No null values in critical columns
    # - Sufficient samples for training (minimum 10)
    #
    # Args:
    #     df: Training DataFrame to validate
    #     text_column: Name of column containing text data
    #     labels_column: Name of column containing label data
    #
    # Returns:
    #     tuple: (is_valid, error_message)
    #         - is_valid (bool): True if all checks pass, False otherwise
    #         - error_message (str or None): Descriptive error if validation fails
    #
    # Example:
    #     >>> df = pd.DataFrame({'text': ['sample'], 'labels': ['cat1']})
    #     >>> is_valid, error = validate_training_data(df, 'text', 'labels')
    #     >>> if not is_valid:
    #     ...     print(f"Validation error: {error}")
    
    # Check required columns exist
    if text_column not in df.columns:
        return False, f"Column '{text_column}' not found in data"
    
    if labels_column not in df.columns:
        return False, f"Column '{labels_column}' not found in data"
    
    # Check for empty data
    if len(df) == 0:
        return False, "Training data is empty"
    
    # Check for null values
    if df[text_column].isnull().any():
        null_count = df[text_column].isnull().sum()
        return False, f"Found {null_count} null values in text column"
    
    if df[labels_column].isnull().any():
        null_count = df[labels_column].isnull().sum()
        return False, f"Found {null_count} null values in labels column"
    
    # Check minimum samples
    if len(df) < 10:
        return False, "Need at least 10 samples for training"
    
    return True, None
