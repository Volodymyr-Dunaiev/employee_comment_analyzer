# Ukrainian Comments Classifier

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Performance](https://img.shields.io/badge/startup-87%25_faster-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A machine learning application that classifies Ukrainian text comments into multiple predefined categories.

## 📥 Download Options

**No Python Installation Required:**

- Download the portable `.exe` version from [Releases](https://github.com/yourusername/comments-classifier/releases)
- Extract and run `Comments_Classifier.exe` – that's it!
- See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for details

**From Source (Python Required):**

- Clone this repository and install dependencies (see [Installation](#installation) below)
- Run from source code for customization and development

---

## Features

- **Multi-label classification** of Ukrainian text comments
- **Training interface** for fine-tuning models with your own data
- **User-friendly Streamlit interface** with separate tabs for classification and training
- **Batch processing** for large files
- **Real-time progress tracking** during classification and training
- **Configurable categories** (currently 13 Ukrainian workplace categories)
- **Comprehensive error handling** and logging
- **Model metrics tracking** (F1, precision, recall)
- **⚡ High-performance lazy-loading** (87% faster startup)
- **Multi-column label support** for flexible training data formats

## Installation

> **Performance Note:** v2.1.0 features optimized lazy-loading for instant startup (~1.3s import time). Models load only when needed, saving time and memory.

1. Clone the repository:

```bash
git clone https://github.com/yourusername/comments-classifier
cd comments-classifier
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package (installs all dependencies from pyproject.toml):

```bash
pip install -e .
```

**For Development:**

```bash
# Install with development dependencies (testing, formatting, linting)
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Configuration

Edit `config.yaml` to customize:

- Model parameters
- Categories
- Input/output settings
- Logging configuration
- Security settings

Example configuration:

```yaml
model:
  path: "model"
  batch_size: 8
  device: "cuda" # or "cpu"

data:
  text_column: "comment"
  chunk_size: 1000

categories:
  - "category1"
  - "category2"
  # Add your categories
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app_ui.py)                 │
│  ┌──────────────────┐              ┌──────────────────┐        │
│  │ Classification   │              │  Training Tab    │        │
│  │      Tab         │              │                  │        │
│  └────────┬─────────┘              └─────────┬────────┘        │
└───────────┼────────────────────────────────────┼────────────────┘
            │                                    │
            ▼                                    ▼
┌───────────────────────┐          ┌─────────────────────────────┐
│  Pipeline (pipeline.py)│          │ Training Interface          │
│  - Lazy classifier     │          │  (train_interface.py)       │
│  - Chunked processing  │          │  - Data validation          │
│  - Progress tracking   │          │  - Training orchestration   │
└──────────┬─────────────┘          └──────────┬──────────────────┘
           │                                   │
           ▼                                   ▼
┌──────────────────────────┐       ┌─────────────────────────────┐
│ CommentClassifier        │       │  Training Utils (train.py)  │
│  (classifier.py)         │       │  - Dataset loading          │
│  - Model/tokenizer       │       │  - Multi-column labels      │
│  - Batch prediction      │       │  - HuggingFace Trainer      │
│  - Input validation      │       │  - Metrics computation      │
└──────────┬───────────────┘       └──────────┬──────────────────┘
           │                                   │
           └───────────┬───────────────────────┘
                       │
                       ▼
          ┌────────────────────────────┐
          │  Core Utilities            │
          ├────────────────────────────┤
          │ • model_utils.py           │
          │ • excel_io.py (chunking)   │
          │ • logger.py (structured)   │
          │ • config.py (YAML)         │
          │ • errors.py (custom types) │
          └────────────────────────────┘
```

### Data Flow

**Classification:**

1. User uploads Excel → UI validates file
2. Pipeline loads config + lazy-initializes classifier
3. File processed in chunks (memory-efficient)
4. Classifier predicts categories for each chunk
5. Results combined and returned as downloadable Excel

**Training:**

1. User uploads labeled data → UI validates format
2. Training interface loads DataFrame (single/multi-column labels)
3. Data split into train/val/test sets
4. HuggingFace Trainer fine-tunes model
5. Model + metadata saved to output directory

## Performance Optimizations

### Recent Improvements (v2.1.0)

- **Lazy-Loading Architecture**: 87% faster application startup (10.07s → 1.29s import time)
  - ML libraries (torch, transformers) load only when needed
  - Deferred model initialization until inference is requested
  - Reduced memory footprint by 500MB-2GB until classification runs
- **Code Optimization**: Cleaner, more maintainable codebase
  - Removed duplicate functions and unused imports
  - Eliminated 16% of redundant code files
  - Inlined helper functions where appropriate
- **Memory Management**: Efficient resource utilization
  - In-memory file handling with `BytesIO`
  - Chunked processing for large datasets
  - On-demand model loading

### Architecture Benefits

- **Fast Startup**: Import pipeline module in ~1.3 seconds instead of ~10 seconds
- **Memory Efficient**: Models load only when classification/training is executed
- **Scalable**: Chunked batch processing handles large files efficiently
- **Clean Code**: Single source of truth for all functions, no duplicates

## Usage

### Running the Application

1. Start the application:

```bash
streamlit run src/ui/app_ui.py
```

The app will open in your browser with two tabs:

### Tab 1: Classification

Use this tab to classify comments with an existing trained model:

1. Upload an Excel file containing comments
2. Click "Run Classification"
3. Monitor progress
4. Download the results

### Tab 2: Training

Use this tab to train a new model or fine-tune an existing one:

1. **Upload Training Data**: Excel/CSV file with labeled comments

   - Required columns: `text` (comments) and `labels` (categories)
   - Labels format: comma-separated or list format
   - Minimum 10 samples required

2. **Configure Training Parameters**:

   - Text/Labels column names
   - Number of epochs (1-10)
   - Batch size (4-32)
   - Learning rate (1e-5 to 5e-5)
   - Test/validation split percentages
   - Base model selection (xlm-roberta-base, bert-base-multilingual-cased, etc.)
   - Output directory for trained model

3. **Start Training**:

   - Click "Start Training"
   - Monitor progress and metrics
   - View training results (loss, F1 scores, precision, recall)

4. **Use Trained Model**:
   - Update `config.yaml` to point to your trained model
   - Switch to Classification tab to test

### Input Format

**For Classification:**

- Excel file with a text column (default: "comment")
- One comment per row

**For Training:**

- Excel/CSV file with two columns:
  - Text column (e.g., "text"): Contains comments
  - Labels column (e.g., "labels"): Contains categories

Example training data formats:

**Format 1: Single column (comma-separated labels)**

```
text                                    | labels
"Чудова команда, дружня атмосфера"    | "Колектив,Взаємодія та комунікація"
"Потрібно більше навчань"             | "Навчання та розвиток"
"Добра зарплата"                       | "Визнання співробітників та винагорода"
```

**Format 2: Multiple columns (one label per column)**

```
text                                    | label_1      | label_2                      | label_3
"Чудова команда, дружня атмосфера"    | "Колектив"   | "Взаємодія та комунікація"  |
"Потрібно більше навчань"             | "Навчання та розвиток" |                    |
"Добра зарплата"                       | "Визнання співробітників та винагорода" | |
```

**Note:** In the UI, select the appropriate format and specify column pattern (e.g., `label_*`, `category_*`, `cat_*`)

### Output Format

**Classification Output:**

- All original columns
- Additional "Predicted_Categories" column with predicted labels

**Training Output:**

- Trained model files in specified directory
- Training metadata (metrics, parameters)
- Label encoder file
- Training logs

## Development

### Project Structure

```
comments-classifier/
├── src/
│   ├── core/
│   │   ├── classifier.py       # CommentClassifier class
│   │   ├── pipeline.py         # Main inference pipeline (lazy-loading)
│   │   ├── train.py            # Training utilities with inline helpers
│   │   ├── train_interface.py  # Training UI wrapper
│   │   ├── model_utils.py      # Model loading utilities
│   │   └── errors.py           # Custom exceptions
│   ├── io/
│   │   └── excel_io.py         # File handling
│   ├── ui/
│   │   └── app_ui.py           # Streamlit interface (classification + training)
│   └── utils/
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging setup
├── tests/
│   ├── test_model.py           # Model tests
│   ├── test_pipeline.py        # Pipeline tests
│   ├── test_io.py              # I/O tests
│   ├── test_training_ui.py     # Training interface tests
│   ├── test_end_to_end.py      # End-to-end tests
│   └── conftest.py             # Shared test fixtures
├── config.yaml                 # Configuration
├── pyproject.toml              # Project metadata & dependencies
├── CHANGELOG.md                # Version history
└── README.md                   # This file
```

### Testing

Run tests with:

```bash
pytest tests/
```

For coverage report:

```bash
pytest --cov=src tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes (pre-commit will run automatically)
5. Run tests: `pytest tests/ -v`
6. Run type checks: `mypy src --ignore-missing-imports`
7. Submit a pull request

**Code Quality Standards:**

- All code passes black, isort, flake8, mypy
- Test coverage maintained above 70%
- Docstrings for public APIs
- Type hints for function signatures

## Security

For security policy, vulnerability reporting, and data handling practices, see [SECURITY.md](SECURITY.md).

**Quick Security Checklist:**

- Input validation for all uploads
- File size and type restrictions
- Local processing (no external data transmission)
- Log redaction (no PII in logs)
- Dependency scanning via `pip-audit`

## Documentation

### Core Documentation

- **[README.md](README.md)** - This file: Overview, installation, architecture
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute getting started guide with examples
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Portable .exe deployment for non-Python users

### Reference Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Version history and detailed changes (v2.1.0+)
- **[SECURITY.md](SECURITY.md)** - Security policy, data handling, privacy guidelines
- **[model_backups/README.md](model_backups/README.md)** - Model versioning and rollback guide

### User Guides

- **For End Users**: Start with QUICKSTART.md → DEPLOYMENT_GUIDE.md
- **For Developers**: README.md → CHANGELOG.md → SECURITY.md
- **For Training**: QUICKSTART.md (Training section) → DEPLOYMENT_GUIDE.md (Training details)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

**Troubleshooting:** See [QUICKSTART.md](QUICKSTART.md#-troubleshooting-guide) for common issues and solutions.

**Issues:** For bugs and feature requests, use the [GitHub issue tracker](https://github.com/yourusername/comments-classifier/issues).

**Security:** For security vulnerabilities, see [SECURITY.md](SECURITY.md#vulnerability-reporting).
