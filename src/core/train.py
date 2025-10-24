# """
# Fine-tuning script for multi-label Ukrainian text classification (compatible with my_app structure).

# Design goals (engineer-grade):
# - Config-driven, reproducible, robust to common dataset formats (CSV/XLSX)
# - Multi-label setup using BCEWithLogitsLoss via HF Trainer (problem_type set)
# - Uses xlm-roberta-base (default) and saves HF-style model/tokenizer to model_dir
# - Saves label encoder and training metadata for inference pipeline
# - Proper metrics (micro/macro F1, precision, recall, exact match)
# - Early stopping, checkpointing, deterministic seeds, optional ONNX export
# - Memory-friendly tokenization and batching, deterministic worker settings
# - Clear logging and simple CLI
# - Compatible with earlier app code: saves label_encoder.pkl and training_meta.pkl into model_dir
# """
import os
import argparse
import logging
import random
import json
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset, DatasetDict
import evaluate

from src.utils.training_validation import validate_training_quality, get_training_recommendations

logger = logging.getLogger(__name__)

# Helper functions for model saving
def save_label_encoder(mlb: MultiLabelBinarizer, model_dir: str) -> None:
    # Save the label encoder to the model directory.
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(mlb, os.path.join(model_dir, "label_encoder.pkl"))

def safe_save_model_and_tokenizer(model, tokenizer, model_dir: str) -> None:
    # Safely save the model and tokenizer to disk.
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def export_onnx(tokenizer, model, model_dir: str, device: str = "cpu", max_length: int = 128) -> str:
    # Export the model to ONNX format.
    dummy_input = tokenizer(
        "Example text for ONNX export",
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True
    ).to(device)
    
    onnx_path = os.path.join(model_dir, "model.onnx")
    
    torch.onnx.export(
        model,
        (dummy_input.input_ids, dummy_input.attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        do_constant_folding=True,
        opset_version=12
    )
    
    return onnx_path
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------- Helpers: parse labels & load dataframe ----------
def parse_labels_column(raw) -> List[str]:
    # Robustly parse a labels cell into a list of strings.
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, (int, float)) and not np.isnan(raw):
        return [str(raw)]
    s = str(raw).strip()
    if not s:
        return []
    # try safe eval for Python list-like content
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in ["|", ",", ";"]:
        if sep in s:
            return [p.strip() for p in s.split(sep) if p.strip()]
    return [s]


def load_dataframe(path: str, text_column: str = "text", labels_column: str = "labels") -> pd.DataFrame:
    # Load CSV or Excel into DataFrame and parse labels.
    # Supports three formats:
    # 1. Single column with comma/pipe-separated labels (labels_column="labels")
    # 2. Multiple columns with one label per column (labels_column="label_*" or "category_*")
    # 3. Auto-detect "Category 1", "Category 2", etc. columns
    
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    
    if text_column not in df.columns:
        raise ValueError(f"Input must contain a '{text_column}' column.")
    
    # Auto-detect "Category 1", "Category 2", etc. format
    category_columns = [col for col in df.columns if col.startswith("Category ") and col.split()[-1].isdigit()]
    if category_columns:
        # Sort by number to maintain order
        category_columns = sorted(category_columns, key=lambda x: int(x.split()[-1]))
        
        def combine_category_columns(row):
            labels = []
            for col in category_columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    labels.append(str(val).strip())
            return labels
        
        df["text"] = df[text_column].fillna("").astype(str)
        df["labels"] = df.apply(combine_category_columns, axis=1)
        return df[["text", "labels"]]
    
    # Check if labels_column contains a wildcard pattern (e.g., "label_*", "category_*")
    if "*" in labels_column or labels_column.startswith("label_") or labels_column.startswith("category_"):
        # Multi-column format: find all columns matching the pattern
        if "*" in labels_column:
            prefix = labels_column.replace("*", "")
            label_columns = [col for col in df.columns if col.startswith(prefix)]
        else:
            # Find all columns starting with label_ or category_
            prefix = labels_column.split("_")[0] + "_"
            label_columns = [col for col in df.columns if col.startswith(prefix)]
        
        if not label_columns:
            raise ValueError(f"No columns found matching pattern '{labels_column}'")
        
        # Combine multiple columns into a list of labels per row
        def combine_label_columns(row):
            labels = []
            for col in label_columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    labels.append(str(val).strip())
            return labels
        
        df["text"] = df[text_column].fillna("").astype(str)
        df["labels"] = df.apply(combine_label_columns, axis=1)
        return df[["text", "labels"]]
    else:
        # Single column format: parse labels from one column
        if labels_column not in df.columns:
            raise ValueError(f"Input must contain a '{labels_column}' column.")
        
        df = df[[text_column, labels_column]].rename(columns={text_column: "text", labels_column: "labels_raw"})
        df["text"] = df["text"].fillna("").astype(str)
        df["labels"] = df["labels_raw"].apply(parse_labels_column)
        return df[["text", "labels"]]


# ---------- Dataset split & conversion ----------
def build_datasets(df: pd.DataFrame, mlb: MultiLabelBinarizer, test_size: float = 0.1, valid_size: float = 0.1, seed: int = 42) -> DatasetDict:
    # Shuffle, split into train/valid/test, and create HF DatasetDict with multi-hot labels.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    test_n = int(n * test_size)
    valid_n = int(n * valid_size)
    test_df = df.iloc[:test_n].copy()
    valid_df = df.iloc[test_n:test_n + valid_n].copy()
    train_df = df.iloc[test_n + valid_n:].copy()

    def to_multi(row_labels):
        return mlb.transform([row_labels])[0].astype(int).tolist()

    for d in (train_df, valid_df, test_df):
        d["labels_multi"] = d["labels"].apply(to_multi)

    ds_train = Dataset.from_pandas(train_df[["text", "labels_multi"]].rename(columns={"labels_multi": "labels"}))
    ds_valid = Dataset.from_pandas(valid_df[["text", "labels_multi"]].rename(columns={"labels_multi": "labels"}))
    ds_test = Dataset.from_pandas(test_df[["text", "labels_multi"]].rename(columns={"labels_multi": "labels"}))
    return DatasetDict({"train": ds_train, "validation": ds_valid, "test": ds_test})


# ---------- Metrics ----------
def get_compute_metrics(pred_threshold: float = 0.5):
    # Returns a compute_metrics function that Trainer can use.
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        import torch as _torch
        probs = _torch.sigmoid(_torch.tensor(logits))
        preds = (probs >= pred_threshold).int().numpy()
        labels = labels.astype(int)

        micro_f1 = f1_metric.compute(predictions=preds, references=labels, average="micro")["f1"]
        macro_f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        micro_prec = precision_metric.compute(predictions=preds, references=labels, average="micro")["precision"]
        micro_rec = recall_metric.compute(predictions=preds, references=labels, average="micro")["recall"]
        exact_match = float((preds == labels).all(axis=1).mean())

        return {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "exact_match": exact_match,
        }

    return compute_metrics


# ---------- Tokenization ----------
def tokenize_dataset(dataset: DatasetDict, tokenizer: AutoTokenizer, max_length: int = 128) -> DatasetDict:
    # Tokenize datasets in a memory-friendly batched way and set format for Trainer.
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ---------- Model creation ----------
def create_model(num_labels: int, model_name_or_path: str):
    # Create HF model configured for multi-label classification.
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    return model


# ---------- Training routine ----------
def train(
    data_path: str,
    model_name_or_path: str = "xlm-roberta-base",
    model_dir: str = "model/ukr_multilabel",
    text_column: str = "text",
    labels_column: str = "labels",
    max_length: int = 128,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    seed: int = 42,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    threshold: float = 0.5,
    save_total_limit: int = 2,
    fp16: bool = True,
    output_dir: Optional[str] = None,
    use_onnx_export: bool = False,
):
    # Main training function for multi-label classification.
    # Orchestrates data loading, tokenization, training, and model saving.
    
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(model_dir, exist_ok=True)
    output_dir = output_dir or os.path.join(model_dir, "hf_checkpoints")

    logger.info("Loading data from %s", data_path)
    df = load_dataframe(data_path, text_column=text_column, labels_column=labels_column)
    logger.info("Rows loaded: %d", len(df))
    
    # Validate training data quality
    logger.info("Validating training data quality...")
    is_valid, error_msg = validate_training_quality(df, text_column='text', labels_column='labels')
    
    if not is_valid:
        logger.error(f"Training data validation failed: {error_msg}")
        raise ValueError(f"Training data validation failed: {error_msg}")
    
    logger.info("Training data validation passed ✓")
    
    # Get and log recommendations
    recommendations = get_training_recommendations(df, text_column='text')
    if recommendations:
        logger.warning("Data quality recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.warning(f"  {i}. {rec}")

    # collect labels
    all_labels = sorted({l for labs in df["labels"] for l in labs})
    if not all_labels:
        raise ValueError("No labels found in dataset.")
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([[]])
    num_labels = len(mlb.classes_)
    logger.info("Detected %d labels: %s", num_labels, mlb.classes_)

    # build HF datasets
    ds = build_datasets(df, mlb, test_size=test_size, valid_size=valid_size, seed=seed)

    logger.info("Loading tokenizer: %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    logger.info("Tokenizing datasets (max_length=%d)...", max_length)
    tokenized = tokenize_dataset(ds, tokenizer, max_length=max_length)

    # model
    logger.info("Creating model with %d labels (multi-label).", num_labels)
    model = create_model(num_labels=num_labels, model_name_or_path=model_name_or_path)

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=2,
    )

    compute_metrics = get_compute_metrics(pred_threshold=threshold)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")
    
    # Backup existing model if it exists (versioning)
    if os.path.exists(os.path.join(model_dir, "config.json")):
        import shutil
        from datetime import datetime
        
        # Find next version number
        version = 1
        backup_base = os.path.join(os.path.dirname(model_dir), "model_backups")
        os.makedirs(backup_base, exist_ok=True)
        
        while os.path.exists(os.path.join(backup_base, f"v{version}")):
            version += 1
        
        backup_dir = os.path.join(backup_base, f"v{version}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Backing up existing model to {backup_dir}")
        shutil.copytree(model_dir, backup_dir)
        
        # Save version info
        version_info = {
            "version": version,
            "timestamp": timestamp,
            "backup_path": backup_dir
        }
        with open(os.path.join(backup_base, f"v{version}_info.json"), "w") as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Previous model backed up as version {version}")
    
    logger.info("Saving new model & artifacts to %s", model_dir)

    # save HF model/tokenizer and label encoder and metadata
    safe_save_model_and_tokenizer(model, tokenizer, model_dir)
    save_label_encoder(mlb, model_dir)
    meta = {
        "model_name_or_path": model_name_or_path,
        "max_length": max_length,
        "threshold": threshold,
        "num_labels": num_labels,
        "labels": list(mlb.classes_),
        "training_args": training_args.to_json_string() if hasattr(training_args, "to_json_string") else str(training_args),
    }
    joblib.dump(meta, os.path.join(model_dir, "training_meta.pkl"))
    logger.info("Saved training meta to %s", os.path.join(model_dir, "training_meta.pkl"))

    if use_onnx_export:
        try:
            onnx_path = export_onnx(tokenizer, model, model_dir, device="cpu", max_length=max_length)
            logger.info("ONNX model exported to %s", onnx_path)
        except Exception as e:
            logger.exception("ONNX export failed: %s", e)

    # final evaluation on test
    logger.info("Running final evaluation on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    logger.info("Test metrics: %s", test_metrics)
    return {"model_dir": model_dir, "test_metrics": test_metrics}


# ---------- CLI ----------
def main(argv=None):
    # Command-line interface for training script.
    parser = argparse.ArgumentParser(description="Train multi-label Ukrainian classifier using xlm-roberta-base (default)")
    parser.add_argument("--data", required=True, help="CSV or XLSX path with columns 'text' and 'labels'")
    parser.add_argument("--model_name_or_path", default="xlm-roberta-base", help="Pretrained HF model (default: xlm-roberta-base)")
    parser.add_argument("--model_dir", default="model/ukr_multilabel", help="Directory to save trained model tokenizer and artifacts")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args(argv)

    train(
        data_path=args.data,
        model_name_or_path=args.model_name_or_path,
        model_dir=args.model_dir,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        seed=args.seed,
        threshold=args.threshold,
        fp16=args.fp16,
        use_onnx_export=args.onnx,
    )


if __name__ == "__main__":
    main()
