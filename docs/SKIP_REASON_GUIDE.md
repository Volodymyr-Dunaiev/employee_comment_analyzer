# Skip Reason Feature - Quick Reference

## Overview

The `skip_reason` column provides detailed categorization of why texts are skipped during classification, improving operator triage without opening logs.

## Skip Reason Values

| Value        | Meaning                             | Example                     |
| ------------ | ----------------------------------- | --------------------------- |
| `none`       | Text processed successfully         | "Great team and management" |
| `empty`      | Text is empty string or None        | `""` or `None`              |
| `whitespace` | Text contains only whitespace       | `"   \n\t  "`               |
| `nan`        | Text is pandas NaN or string "nan"  | `NaN` or `"nan"`            |
| `non_text`   | Text is not a string (number, bool) | `123` or `True`             |

## Output Format

### classify_batch() Returns

```python
{
    # Category probabilities
    'Колектив': [0.85, 0.0, 0.12],
    'Керівництво': [0.45, 0.0, 0.78],

    # Skip reason per row
    'skip_reason': ['none', 'empty', 'none'],

    # Metadata
    '_metadata': {
        'skipped_indices': [1],
        'skip_reason_counts': {
            'none': 2,
            'empty': 1
        }
    }
}
```

### Output DataFrame Columns

```
| text                  | Колектив | Керівництво | skip_reason |
|-----------------------|----------|-------------|-------------|
| Great team            | 0.85     | 0.45        | none        |
|                       | 0.00     | 0.00        | nan         |
| Good management       | 0.12     | 0.78        | none        |
```

## Usage Examples

### Check Skip Reasons in Results

```python
from src.core.classifier import CommentClassifier

classifier = CommentClassifier(config)
texts = ["Valid text", "", "Another text"]
result = classifier.classify_batch(texts)

# Count by reason
skip_counts = result['_metadata']['skip_reason_counts']
print(f"Skipped: {skip_counts}")
# Output: {'none': 2, 'nan': 1}

# Filter only processed rows
skip_reasons = result['skip_reason']
processed_indices = [i for i, reason in enumerate(skip_reasons) if reason == 'none']
```

### Analyze Skip Reasons in Batch Results

```python
from src.core.batch_processor import BatchProcessor

processor = BatchProcessor(classifier)
result = processor.process_files(file_paths)

# Check per-file skip counts
for filename, skip_count in result.skip_counts_by_file.items():
    if skip_count > 0:
        print(f"{filename}: {skip_count} skipped")

# Analyze output DataFrame
for filename, df in result.results_by_file.items():
    reason_counts = df['skip_reason'].value_counts()
    print(f"\n{filename}:")
    print(reason_counts)
```

## Log Output

### Classifier Warnings

```
WARNING - Skipped 5 texts (out of 100 total): empty=3, whitespace=1, nan=1
```

### Batch Processor Warnings

```
WARNING - ✓ Processed data.xlsx: 1000 total rows, 950 classified, 50 skipped (5.0%)
```

## UI Display

The Streamlit UI shows per-file breakdown:

```
⚠️ 50 empty or invalid comments were skipped (5.0% of 1000 total rows).
These rows have skip_reason set and zero probabilities.

Per-file breakdown:
- data.xlsx: 50 skipped (5.0%)
  - empty: 30
  - whitespace: 15
  - nan: 5
```

## Migration from is_skipped

### Before (v2.2.0)

```python
result = {
    'Category1': [0.8, 0.0, 0.7],
    'is_skipped': [False, True, False],
    '_metadata': {'skipped_indices': [1]}
}
```

### After (v2.3.0+)

```python
result = {
    'Category1': [0.8, 0.0, 0.7],
    'skip_reason': ['none', 'empty', 'none'],
    '_metadata': {
        'skipped_indices': [1],
        'skip_reason_counts': {'none': 2, 'empty': 1}
    }
}
```

## Troubleshooting

### Q: Why are empty strings showing as 'nan'?

**A**: Excel/pandas converts empty strings to NaN when writing/reading. This is expected behavior.

### Q: How do I count only successfully processed rows?

**A**: Filter by `skip_reason == 'none'`:

```python
df_processed = df[df['skip_reason'] == 'none']
processed_count = (df['skip_reason'] == 'none').sum()
```

### Q: Are there length limitations on texts?

**A**: No. As of v2.3.1, all text lengths are processed. Very long texts are automatically truncated by the tokenizer to fit the model's max token length (typically 512 tokens), but they are not skipped.

### Q: Can I customize skip reasons?

**A**: Yes, edit `src/core/skip_reasons.py` and add new enum values:

```python
class SkipReason(str, Enum):
    # ... existing values ...
    PROFANITY = "profanity"  # Custom filter
```

Then update `from_text()` method to detect it.

## Benefits Over is_skipped

1. **Detailed triage**: Know WHY rows were skipped, not just IF
2. **Better logging**: Aggregate counts prevent log flooding
3. **Operator insights**: Fix data quality issues faster
4. **Analytics**: Track skip reason trends over time
5. **Debugging**: Quickly identify encoding vs empty vs NaN issues

## Performance Impact

- **Negligible**: Skip reason detection adds <1ms per 1000 texts
- **Memory**: Stores string per row (~10 bytes vs 1 byte for boolean)
- **Worth it**: Operator time savings >> tiny performance cost
