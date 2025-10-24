"""Training data validation utilities."""
import pandas as pd
from typing import Tuple, Optional, List
from collections import Counter
from src.utils.logger import get_logger

logger = get_logger(__name__)

def validate_training_quality(
    df: pd.DataFrame,
    text_column: str = 'text',
    labels_column: str = 'labels',
    min_samples_total: int = 50,
    min_samples_per_category: int = 5,
    max_imbalance_ratio: float = 20.0
) -> Tuple[bool, Optional[str]]:
    """Validate training data quality beyond basic format checks.
    
    Args:
        df: Training DataFrame
        text_column: Name of text column
        labels_column: Name of labels column (or check for Category columns)
        min_samples_total: Minimum total samples required
        min_samples_per_category: Minimum samples per category
        max_imbalance_ratio: Maximum ratio between most and least common categories
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check total samples
    if len(df) < min_samples_total:
        return False, f"Insufficient training data: {len(df)} samples found, minimum {min_samples_total} required"
    
    # Check for duplicate rows
    duplicates = df.duplicated(subset=[text_column]).sum()
    if duplicates > len(df) * 0.1:  # More than 10% duplicates
        logger.warning(f"High duplication rate: {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
    
    # Check for empty text
    empty_text = df[text_column].str.strip().eq('').sum()
    if empty_text > 0:
        return False, f"Found {empty_text} rows with empty text"
    
    # Check text length distribution
    text_lengths = df[text_column].str.len()
    very_short = (text_lengths < 10).sum()
    if very_short > len(df) * 0.2:  # More than 20% very short texts
        logger.warning(f"Many very short texts: {very_short} texts with <10 characters ({very_short/len(df)*100:.1f}%)")
    
    # Extract all labels
    all_labels = []
    if labels_column in df.columns:
        # Single labels column format
        for labels in df[labels_column]:
            if isinstance(labels, list):
                all_labels.extend(labels)
            elif isinstance(labels, str):
                # Handle comma-separated
                all_labels.extend([l.strip() for l in labels.split(',') if l.strip()])
    else:
        # Check for Category 1, Category 2, etc.
        category_cols = [col for col in df.columns if col.startswith('Category ')]
        if not category_cols:
            return False, "No labels found. Need 'labels' column or 'Category 1', 'Category 2', etc."
        
        for col in category_cols:
            all_labels.extend(df[col].dropna().str.strip().tolist())
    
    if not all_labels:
        return False, "No labels found in the data"
    
    # Count samples per category
    label_counts = Counter(all_labels)
    
    # Check minimum samples per category
    rare_categories = {cat: count for cat, count in label_counts.items() if count < min_samples_per_category}
    if rare_categories:
        rare_list = ', '.join([f"{cat}({count})" for cat, count in rare_categories.items()])
        return False, f"Categories with too few samples (min {min_samples_per_category}): {rare_list}"
    
    # Check class imbalance
    most_common_count = label_counts.most_common(1)[0][1]
    least_common_count = label_counts.most_common()[-1][1]
    imbalance_ratio = most_common_count / least_common_count
    
    if imbalance_ratio > max_imbalance_ratio:
        logger.warning(
            f"High class imbalance detected: {imbalance_ratio:.1f}x "
            f"(most common: {most_common_count}, least common: {least_common_count})"
        )
    
    # Log category distribution
    logger.info(f"Category distribution: {dict(label_counts.most_common())}")
    logger.info(f"Total unique categories: {len(label_counts)}")
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}x")
    
    return True, None

def get_training_recommendations(df: pd.DataFrame, text_column: str = 'text') -> List[str]:
    """Get recommendations for improving training data.
    
    Args:
        df: Training DataFrame
        text_column: Name of text column
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Check duplicates
    duplicates = df.duplicated(subset=[text_column]).sum()
    if duplicates > 0:
        recommendations.append(
            f"Remove {duplicates} duplicate rows to improve model generalization"
        )
    
    # Check text lengths
    text_lengths = df[text_column].str.len()
    very_short = (text_lengths < 10).sum()
    very_long = (text_lengths > 1000).sum()
    
    if very_short > 0:
        recommendations.append(
            f"Consider removing {very_short} very short texts (<10 chars) - may be noise"
        )
    
    if very_long > 0:
        recommendations.append(
            f"{very_long} texts are very long (>1000 chars) - may need truncation or splitting"
        )
    
    # Check dataset size
    if len(df) < 100:
        recommendations.append(
            f"Small dataset ({len(df)} samples). Consider collecting more data for better performance."
        )
    elif len(df) < 500:
        recommendations.append(
            f"Moderate dataset ({len(df)} samples). More data recommended for production use."
        )
    
    return recommendations
