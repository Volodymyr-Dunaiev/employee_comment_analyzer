# Migration Guide: is_skipped → skip_reason

## Overview

Version 2.3.0 replaces the boolean `is_skipped` column with a more detailed `skip_reason` column.

## What Changed

### Before (v2.2.0)

```python
# API output
result = classifier.classify_batch(["Good", "", "Bad"])
# {
#     'Category1': [0.8, 0.0, 0.3],
#     'is_skipped': [False, True, False],
#     '_metadata': {'skipped_indices': [1]}
# }

# DataFrame output
df = pd.read_excel('output.xlsx')
# Columns: text, Category1, Category2, is_skipped (bool)
```

### After (v2.3.0+)

```python
# API output
result = classifier.classify_batch(["Good", "", "Bad"])
# {
#     'Category1': [0.8, 0.0, 0.3],
#     'skip_reason': ['none', 'nan', 'none'],  # More detail!
#     '_metadata': {
#         'skipped_indices': [1],
#         'skip_reason_counts': {'none': 2, 'nan': 1}  # New!
#     }
# }

# DataFrame output
df = pd.read_excel('output.xlsx')
# Columns: text, Category1, Category2, skip_reason (str)
```

## Why This Change?

### Problem with is_skipped

- Only tells you IF a row was skipped
- Doesn't tell you WHY
- Operators had to dig through logs
- Hard to identify data quality patterns

### Benefits of skip_reason

- Know exactly WHY: empty vs whitespace vs nan vs non-text
- Better logging with aggregated counts
- Faster data quality triage
- Can track skip reason trends over time

**Note:** As of v2.3.1, length limitations have been removed. All text lengths are now processed.

## Migration Steps

### 1. Update Code Reading API Results

**Before:**

```python
result = classifier.classify_batch(texts)
skipped = result['is_skipped']
skipped_count = sum(skipped)
```

**After:**

```python
result = classifier.classify_batch(texts)
skip_reasons = result['skip_reason']
skipped_count = sum(1 for r in skip_reasons if r != 'none')

# Or use metadata (recommended):
skipped_count = len(result['_metadata']['skipped_indices'])
```

### 2. Update Code Reading DataFrames

**Before:**

```python
df = pd.read_excel('output.xlsx')
processed_df = df[~df['is_skipped']]  # Filter processed rows
skipped_df = df[df['is_skipped']]      # Filter skipped rows
```

**After:**

```python
df = pd.read_excel('output.xlsx')
processed_df = df[df['skip_reason'] == 'none']  # Filter processed rows
skipped_df = df[df['skip_reason'] != 'none']     # Filter skipped rows

# Or be more specific:
empty_df = df[df['skip_reason'] == 'empty']
nan_df = df[df['skip_reason'] == 'nan']
```

### 3. Update Analytics Code

**Before:**

```python
# Count skipped vs processed
skipped_count = df['is_skipped'].sum()
processed_count = (~df['is_skipped']).sum()
```

**After:**

```python
# Count by reason
reason_counts = df['skip_reason'].value_counts()
processed_count = (df['skip_reason'] == 'none').sum()
skipped_count = (df['skip_reason'] != 'none').sum()

# More detailed breakdown:
print(f"Empty: {(df['skip_reason'] == 'empty').sum()}")
print(f"Whitespace: {(df['skip_reason'] == 'whitespace').sum()}")
print(f"NaN: {(df['skip_reason'] == 'nan').sum()}")
```

### 4. Update Tests and Mocks

**Before:**

```python
mock_result = {
    'Category1': [0.8, 0.0],
    'is_skipped': [False, True],
    '_metadata': {'skipped_indices': [1]}
}
```

**After:**

```python
mock_result = {
    'Category1': [0.8, 0.0],
    'skip_reason': ['none', 'empty'],
    '_metadata': {
        'skipped_indices': [1],
        'skip_reason_counts': {'none': 1, 'empty': 1}
    }
}
```

## Compatibility Layer (Temporary)

If you need to maintain backward compatibility temporarily:

```python
# Add this to your code:
def add_is_skipped_column(df):
    """Add is_skipped column for backward compatibility."""
    if 'skip_reason' in df.columns:
        df['is_skipped'] = df['skip_reason'] != 'none'
    return df

# Usage:
df = pd.read_excel('output.xlsx')
df = add_is_skipped_column(df)
# Now df has both skip_reason and is_skipped
```

## Breaking Changes Checklist

Check these in your codebase:

- [ ] ✅ Code that reads `result['is_skipped']`
- [ ] ✅ Code that filters by `df['is_skipped']`
- [ ] ✅ Tests that check `is_skipped` values
- [ ] ✅ Analytics that count `is_skipped`
- [ ] ✅ Reports that display `is_skipped`
- [ ] ✅ Database schemas storing `is_skipped`

## Common Patterns

### Pattern 1: Filter Only Valid Rows

**Before:**

```python
valid_rows = df[~df['is_skipped']]
```

**After:**

```python
valid_rows = df[df['skip_reason'] == 'none']
```

### Pattern 2: Count Categories

**Before:**

```python
n_skipped = df['is_skipped'].sum()
n_processed = len(df) - n_skipped
```

**After:**

```python
n_processed = (df['skip_reason'] == 'none').sum()
n_skipped = len(df) - n_processed

# Or use value_counts:
reason_counts = df['skip_reason'].value_counts()
n_processed = reason_counts.get('none', 0)
```

### Pattern 3: Generate Reports

**Before:**

```python
report = {
    'total': len(df),
    'skipped': df['is_skipped'].sum(),
    'processed': (~df['is_skipped']).sum()
}
```

**After:**

```python
reason_counts = df['skip_reason'].value_counts().to_dict()
report = {
    'total': len(df),
    'processed': reason_counts.get('none', 0),
    'skipped': sum(v for k, v in reason_counts.items() if k != 'none'),
    'skip_reasons': {k: v for k, v in reason_counts.items() if k != 'none'}
}
# Example output:
# {
#     'total': 1000,
#     'processed': 950,
#     'skipped': 50,
#     'skip_reasons': {'empty': 30, 'whitespace': 15, 'nan': 5}
# }
```

## FAQ

### Q: Can I keep using is_skipped?

**A:** Not directly. The API no longer returns it. Use the compatibility layer above if needed temporarily.

### Q: How do I convert old output files?

**A:** If you have old files with `is_skipped`, you can't convert them to detailed skip_reason retroactively. Re-process the files with v2.3.0 to get skip_reason.

### Q: What if I just want a boolean?

**A:** Convert it:

```python
df['is_skipped'] = df['skip_reason'] != 'none'
```

### Q: Are there only these skip reasons?

**A:** Currently: `none`, `empty`, `whitespace`, `nan`, `non_text`. You can add custom reasons by editing `src/core/skip_reasons.py`.

**Note:** The `too_long` skip reason was removed in v2.3.1. All text lengths are now processed.

### Q: What happens to old test files?

**A:** Tests that expect `is_skipped` will fail. Update them to use `skip_reason`. See the updated test files for examples.

## Testing Your Migration

### 1. Unit Tests

```python
def test_skip_reason_migration():
    """Test that skip_reason works as expected."""
    from src.core.classifier import CommentClassifier

    classifier = CommentClassifier(config)
    result = classifier.classify_batch(["Valid", "", "  ", "nan"])

    # Check skip_reason column exists
    assert 'skip_reason' in result
    assert 'is_skipped' not in result  # No longer returned

    # Check values
    expected = ['none', 'nan', 'whitespace', 'nan']  # Note: empty→nan in Excel
    assert result['skip_reason'] == expected or result['skip_reason'] == ['none', 'empty', 'whitespace', 'nan']

    # Check metadata
    assert '_metadata' in result
    assert 'skip_reason_counts' in result['_metadata']
```

### 2. Integration Test

```python
def test_batch_processing_migration():
    """Test batch processing with skip_reason."""
    from src.core.batch_processor import BatchProcessor

    processor = BatchProcessor(classifier)
    result = processor.process_files(file_paths)

    # Check outputs
    for filename, df in result.results_by_file.items():
        assert 'skip_reason' in df.columns
        assert 'is_skipped' not in df.columns

        # Verify reasons are valid
        valid_reasons = {'none', 'empty', 'whitespace', 'nan', 'non_text'}
        assert set(df['skip_reason'].unique()).issubset(valid_reasons)
```

## Rollback Plan

If you need to rollback to v2.2.0:

```bash
# Rollback code
git checkout v2.2.0

# Reinstall dependencies
pip install -e .

# Run tests
pytest tests/
```

**Note:** Output files created with v2.3.0 will have `skip_reason` column. If you rollback, you'll need to re-process those files.

## Timeline

- **v2.2.0**: Last version with `is_skipped`
- **v2.3.0**: Introduces `skip_reason` (breaking change)
- **Deprecation period**: None (direct replacement)

## Support

If you encounter migration issues:

1. Check this guide
2. See `docs/SKIP_REASON_GUIDE.md` for detailed reference
3. Review updated tests in `tests/test_batch_processor_integration.py`
4. Check logs for skip_reason values

## Example Migration PR

```
feat: migrate from is_skipped to skip_reason

BREAKING CHANGE: is_skipped column replaced with skip_reason

Changes:
- Update all code reading result['is_skipped'] → result['skip_reason']
- Update DataFrame filters: df['is_skipped'] → df['skip_reason'] != 'none'
- Update analytics to use skip_reason.value_counts()
- Add skip_reason breakdowns to reports

Benefits:
- Know WHY rows were skipped (empty vs whitespace vs nan)
- Better data quality insights
- Faster operator triage

Tested:
- All unit tests passing
- Integration tests with sample data
- Reports show detailed skip reasons
```
