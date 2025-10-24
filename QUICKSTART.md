# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -e .
```

Or with development tools:

```bash
pip install -e ".[dev]"
```

### Step 2: Run the Application

```bash
streamlit run src/ui/app_ui.py
```

The app will open in your browser at `http://localhost:8501`

**Note:** First startup is fast (~1.3 seconds) thanks to lazy-loading optimization. Models load only when you run classification or training.

---

## 📊 Using Classification

1. Go to the **Classification** tab
2. Upload an Excel file with a `comment` column
3. Click **Run Classification**
4. Download the results with predicted categories

### Example Input File

| comment                         |
| ------------------------------- |
| Чудова команда, всі дуже дружні |
| Потрібно покращити умови праці  |
| Дуже цікава робота              |

### Example Output File

| comment                         | Predicted_Categories               |
| ------------------------------- | ---------------------------------- |
| Чудова команда, всі дуже дружні | Колектив, Взаємодія та комунікація |
| Потрібно покращити умови праці  | Умови праці                        |
| Дуже цікава робота              | Зміст роботи, Внутрішня мотивація  |

---

## 🎓 Training Your Own Model

### Step 1: Prepare Training Data

Create an Excel/CSV file with two columns:

| text           | labels                            |
| -------------- | --------------------------------- |
| Чудова команда | Колектив,Взаємодія та комунікація |
| Погані умови   | Умови праці                       |
| Цікава робота  | Зміст роботи                      |

**Requirements:**

- Minimum 10 samples
- Labels can be comma-separated or in list format
- Both columns must be present

### Step 2: Train the Model

1. Go to the **Training** tab
2. Upload your training file
3. Configure parameters:

   - **Text Column**: Name of column with comments (e.g., "text")
   - **Labels Column**: Name of column with categories (e.g., "labels")
   - **Epochs**: Start with 3-5
   - **Batch Size**: 8 works well for most cases
   - **Learning Rate**: Default (2e-5) is usually good
   - **Test/Validation Split**: Keep defaults (10% each)
   - **Base Model**: xlm-roberta-base (best for Ukrainian)
   - **Output Directory**: Where to save the model (e.g., "./my_model")

4. Click **Start Training**
5. Wait for training to complete (can take 10 minutes to several hours)
6. Review the metrics (F1 score, precision, recall)

### Step 3: Use Your Trained Model

Update `config.yaml`:

```yaml
model:
  path: "./my_model" # Your output directory
  batch_size: 8
  device: "cpu"
```

Now you can use the Classification tab with your custom model!

---

## 🛠️ Configuration

Edit `config.yaml` to customize:

### Change Text Column Name

```yaml
data:
  text_column: "my_column_name"
```

### Adjust Batch Size

```yaml
model:
  batch_size: 16 # Increase for faster processing (requires more memory)
```

### Use GPU

```yaml
model:
  device: "cuda" # Requires CUDA-capable GPU
```

### Add/Remove Categories

```yaml
categories:
  - "Category 1"
  - "Category 2"
  # Add your categories
```

---

## 📝 Tips for Better Results

### For Classification:

- Ensure comments are in Ukrainian
- One comment per row
- Remove any special formatting

### For Training:

- **More data is better**: Aim for at least 100 samples
- **Balanced categories**: Try to have similar numbers of samples per category
- **Quality labels**: Ensure labels are accurate and consistent
- **Multi-label**: Each sample can have multiple categories
- **Monitor metrics**: F1 score > 0.7 is generally good

### Troubleshooting:

- **Out of memory**: Reduce batch_size in config.yaml
- **Slow processing**: Enable GPU if available
- **Poor predictions**: Retrain with more data
- **File upload errors**: Check file size limit in config.yaml

---

## 🧪 Testing

Run tests to verify everything works:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_training_ui.py -v

# Test performance improvements
python test_performance.py
```

### Performance Metrics

- **Import Speed**: ~1.3 seconds (87% faster than v2.0)
- **Memory Usage**: Models load on-demand (saves 500MB-2GB)
- **Startup Time**: Instant application launch

---

## 🆘 Troubleshooting Guide

| Error Message                                       | Cause                           | Solution                                                                        |
| --------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'src'`        | Package not installed           | Run `pip install -e .` from project root                                        |
| `FileNotFoundError: config.yaml`                    | Working directory incorrect     | Ensure you're in project root or set `PYTHONPATH`                               |
| `PipelineError: Required column 'X' not found`      | Column name mismatch            | Check `text_column` in config matches your Excel column                         |
| `CUDA out of memory`                                | Batch size too large for GPU    | Reduce `batch_size` in config.yaml or use `device: cpu`                         |
| `ValueError: Model not found at path`               | Model directory missing/invalid | Verify `model.path` in config.yaml points to valid model                        |
| `ImportError: cannot import name 'load_metric'`     | Outdated dependencies           | Run `pip install -e ".[dev]"` to update all dependencies                        |
| `Validation Error: Minimum 10 samples`              | Insufficient training data      | Add more labeled examples to training file                                      |
| `FileUploadError: File size exceeds limit`          | File too large                  | Check `security.max_file_size_mb` in config.yaml                                |
| `KeyError: 'labels'`                                | Labels column missing           | Ensure labels column exists or adjust `labels_column` parameter                 |
| `RuntimeError: Expected all tensors on same device` | CPU/GPU mismatch                | Set `device: cpu` in config.yaml if no CUDA available                           |
| Test failures after upgrade                         | Import path issues              | Delete `__pycache__` dirs: `find . -type d -name __pycache__ -exec rm -rf {} +` |
| Streamlit won't start                               | Port conflict                   | Try: `streamlit run src/ui/app_ui.py --server.port 8502`                        |

### Quick Diagnostics

```bash
# Verify installation
python -c "import src; print('✓ Package installed')"

# Check dependencies
pip list | grep -E "(torch|transformers|streamlit|pandas)"

# Test model loading
python -c "from src.core.classifier import CommentClassifier; print('✓ Can import classifier')"

# Validate config
python -c "from src.utils.config import load_config; c=load_config(); print('✓ Config valid')"
```

---

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check `config.yaml` for all configuration options
- Review test files in `tests/` for usage examples
- Customize categories for your use case

---

## 🤝 Support

For issues and questions:

1. Check this guide first
2. Review error messages in the logs
3. Check GitHub issues
4. Create a new issue with:
   - Error message
   - Steps to reproduce
   - Your configuration
