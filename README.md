# Ukrainian Comments Classifier

![Version](https://img.shields.io/badge/version-2.5.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Performance](https://img.shields.io/badge/startup-87%25_faster-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A machine learning application that classifies Ukrainian text comments into multiple predefined categories.

**What's New in v2.5.0:**

- ✅ **Simplified architecture** - Sequential processing optimized for 2-3 file workflows
- ✅ **Output directory selection** - Choose where to save results (single & batch mode)
- ✅ **Timestamped filenames** - Automatic: `classified_FileName_YYYYMMDD_HHMMSS.xlsx`
- ✅ **Results summary** - View processing stats: total comments, labels, top categories
- ✅ **Reduced complexity** - Removed multiprocessing overhead (~200 lines)

## 📥 Download & Installation

### Option 1: Portable Application (No Python Required)

**For End Users on Windows:**

1. Download `Comments_Classifier_Portable.zip` from [Releases](https://github.com/Volodymyr-Dunaiev/employee_comment_analyzer/releases)
2. Extract anywhere on your computer
3. Double-click `Comments_Classifier.exe` or `START_APP.bat`
4. App opens automatically in your browser at http://localhost:8501
5. Upload Excel → Classify → Download results

**What's included:**

- `Comments_Classifier.exe` - Main application
- `Train_Model.exe` - Training utility
- `config.yaml` - Settings
- `data/`, `model/`, `logs/` - Working directories
- Quick-start documentation

**System Requirements:**

- Windows 10/11
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Option 2: From Source (Python Required)

**For Developers and Customization:**

1. **Clone the repository:**

```bash
git clone https://github.com/Volodymyr-Dunaiev/employee_comment_analyzer
cd employee_comment_analyzer
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install package:**

```bash
pip install -e .
```

**For Development:**

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Start the Application

**From portable .exe:**

```cmd
START_APP.bat
```

**From source:**

```bash
streamlit run src/ui/app_ui.py
```

The app opens in your browser at `http://localhost:8501`

**Note:** First startup is fast (~1.3 seconds) thanks to lazy-loading optimization.

### Step 2: Classify Comments

1. Go to the **Classification** tab
2. Upload an Excel file with a `comment` column
3. Click **Run Classification**
4. Download the results with predicted categories

**Example Input:**

| comment                         |
| ------------------------------- |
| Чудова команда, всі дуже дружні |
| Потрібно покращити умови праці  |

**Example Output:**

| comment                         | Predicted_Categories               | skip_reason |
| ------------------------------- | ---------------------------------- | ----------- |
| Чудова команда, всі дуже дружні | Колектив, Взаємодія та комунікація | none        |
| Потрібно покращити умови праці  | Умови праці                        | none        |

### Step 3: Train Your Own Model (Optional)

1. Prepare training data (Excel/CSV):

| text           | labels                            |
| -------------- | --------------------------------- |
| Чудова команда | Колектив,Взаємодія та комунікація |
| Погані умови   | Умови праці                       |

**Requirements:**

- Minimum 50 samples recommended
- Labels can be comma-separated or in multiple columns
- At least 5 samples per category

2. Go to **Training** tab, upload file, configure parameters
3. Click **Start Training**
4. Wait for completion (10 minutes to several hours)
5. Review metrics (F1 score > 0.7 is good)

---

## Features

- **Multi-label classification** of Ukrainian text comments
- **Training interface** for fine-tuning models with your own data
- **User-friendly Streamlit interface** with separate tabs for classification and training
- **Batch file processing** - Upload and classify multiple files sequentially with progress tracking
- **Real-time progress tracking** during classification and training
- **Configurable categories** (currently 13 Ukrainian workplace categories)
- **Comprehensive error handling** and logging
- **Model metrics tracking** (F1, precision, recall)
- **⚡ High-performance lazy-loading** (87% faster startup)
- **Multi-column label support** for flexible training data formats
- **Automatic memory management** - Auto-tunes batch size based on available RAM
- **Model versioning** - Auto-backup before overwriting trained models

---

## 🎓 Training Guide

### Preparing Training Data

Create an Excel/CSV file with comments and their categories. The app supports three formats:

**Format 1: Single labels column (comma-separated)**

```
| text                    | labels                    |
|------------------------|---------------------------|
| Чудова команда         | Зарплата,Колектив        |
| Погані умови           | Умови праці              |
```

**Format 2: Category columns (recommended)**

```
| text              | Category 1  | Category 2    |
|------------------|-------------|---------------|
| Чудова команда   | Зарплата    | Колектив      |
| Погані умови     | Умови праці |               |
```

**Format 3: Pattern columns**

```
| text              | label_1     | label_2       |
|------------------|-------------|---------------|
| Чудова команда   | Зарплата    | Колектив      |
```

### Training via UI

1. **Upload Training File** in Training tab
2. **Configure Parameters:**

   - **Text Column**: Column name with comments (default: "text")
   - **Labels Column**: Column name with categories (default: "labels")
   - **Epochs**: 3-5 for new models, 1-2 for refinement
   - **Batch Size**: 8 (reduce to 4 if out of memory)
   - **Learning Rate**: 2e-5 (default, usually good)
   - **Base Model**: xlm-roberta-base (best for Ukrainian)
   - **Output Directory**: Where to save (e.g., "./my_model")

3. **Start Training** and monitor progress
4. **Review Metrics**: F1, precision, recall (F1 > 0.7 is good)

### Training via Command Line (Portable .exe)

```cmd
Train_Model.exe --data data\train.xlsx --epochs 3
```

**Options:**

```cmd
--data              Training file path (required)
--epochs            Number of epochs (default: 3)
--batch_size        Batch size (default: 8)
--lr                Learning rate (default: 2e-5)
--model_dir         Output directory (default: model\ukr_multilabel)
--text_column       Text column name (default: text)
--labels_column     Labels column name (default: labels)
```

### Model Refinement

To update an existing model with new data:

```cmd
Train_Model.exe --data data\new_samples.xlsx --model_name_or_path model\ukr_multilabel --epochs 2
```

**Previous model automatically backed up** to `model_backups/v1/`, `v2/`, etc.

### Restoring Previous Model

If new model performs poorly:

```powershell
# Windows
Remove-Item -Recurse -Force model
Copy-Item -Recurse model_backups/v1 model
```

```bash
# Linux/Mac
rm -rf model
cp -r model_backups/v1 model
```

### Training Tips

- **More data is better**: Aim for 100+ samples
- **Balanced categories**: Similar sample counts per category
- **Quality labels**: Accurate and consistent categorization
- **Monitor validation metrics**: Watch for overfitting
- **Auto-validation**: Training checks data quality before starting

---

## ⚙️ Configuration

### config.yaml

All settings configured in `config.yaml`:

```yaml
model:
  model_name_or_path: "xlm-roberta-base"
  model_dir: "./model/ukr_multilabel"
  base_model_name: "xlm-roberta-base"
  labels:
    - "Керівництво"
    - "Зарплата"
    - "Умови праці"
    - "Колектив"
    - "Самореалізація"
  confidence_threshold: 0.3 # Adjust threshold (0.0-1.0)
  max_length: 512 # Max token length
  device: "auto" # auto, cpu, cuda, or mps

training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-5
  validation_split: 0.2

paths:
  data_dir: "./data"
  log_dir: "./logs"
```

### Key Settings

**Confidence Threshold** (0.0-1.0):

- **0.3** (default): Balanced precision/recall
- **0.5**: Higher precision, fewer predictions
- **0.2**: Higher recall, more predictions

**Device Selection:**

- **auto**: Automatically uses GPU if available
- **cuda**: Force NVIDIA GPU
- **mps**: Force Apple Silicon GPU
- **cpu**: Force CPU (slower but always works)

**Batch Size:**

- **8**: Good for most GPUs
- **4**: If out of memory errors
- **16**: If you have powerful GPU

---

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

Use this tab to classify comments with an existing trained model.

#### Single File Mode

1. Select "Single File" processing mode
2. Upload an Excel file containing comments
3. (Optional) Specify output directory (default: Downloads)
4. Click "Run Classification"
5. Monitor progress and view processing summary
6. Results automatically saved with timestamped filename

#### Batch Processing Mode

Process multiple files sequentially with progress tracking:

1. Select "Batch Processing" mode
2. Upload multiple Excel/CSV files simultaneously
3. Configure options:
   - **Output Directory**: Choose where to save results (default: Downloads)
   - **Output Prefix**: Customize filename prefix (default: "classified\_")
   - **Create combined output**: Optionally merge all results into one file
4. Click "Process Batch"
5. View processing summary (success rate, total comments, time)
6. Results saved to your chosen directory with timestamped filenames

**Batch Processing Benefits:**

- Sequential processing optimized for typical 2-3 file workflows
- Timestamped output files: `classified_OriginalName_YYYYMMDD_HHMMSS.xlsx`
- Automatic file validation before processing
- Detailed error reporting per file
- Optional combined output with source file tracking
- Choose your own output directory

### Tab 2: Training

Use this tab to train a new model or refine an existing one:

#### Training Modes

**🆕 Train from Base Model (New Model)**

- Start fresh with pre-trained model (xlm-roberta-base)
- Best for: First-time training, new categories, poor existing model
- Parameters: 3-5 epochs, learning rate 2e-5

**🔄 Continue Training Existing Model (Refinement)**

- Load and improve your existing trained model
- Best for: Adding new samples, fixing mistakes, expanding categories
- Parameters: 1-2 epochs, learning rate 1e-5 (auto-adjusted by UI)
- Previous model auto-backed up to `model_backups/v#/`

#### Steps

1. **Upload Training Data**: Excel/CSV file with labeled comments

   - Required columns: `text` (comments) and `labels` (categories)
   - Labels format: comma-separated or list format
   - Minimum 10 samples required

2. **Select Training Mode**:

   - **Train from base model**: Choose xlm-roberta-base, bert-base-multilingual-cased, or xlm-roberta-large
   - **Continue training**: Enter path to existing model (default: `./model/ukr_multilabel`)

3. **Configure Training Parameters** (auto-adjusted for refinement):

   - Text/Labels column names
   - Number of epochs (3-5 for new, 1-2 for refinement)
   - Batch size (4-32, based on RAM)
   - Learning rate (2e-5 for new, 1e-5 for refinement)
   - Test/validation split percentages (10% recommended)
   - Output directory for trained model

4. **Start Training**:

   - Click "Start Training"
   - Monitor progress bar and status
   - View training results (loss, F1 scores, precision, recall)

5. **Use Trained Model**:
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

## 🔒 Security

### Data Handling and Privacy

- **Local Processing**: All classification and training happens on your machine
- **No Data Transmission**: Comments never leave your system
- **No Cloud Dependencies**: Models run locally, no external APIs
- **PII Responsibilities**: You control data access and deletion

### What We Do

- ✅ Process data locally with PyTorch/Transformers
- ✅ Validate input files (size, format, columns)
- ✅ Sanitize file paths and text inputs
- ✅ Log errors without sensitive data

### What We Don't Do

- ❌ Send data to external servers
- ❌ Store uploaded files permanently
- ❌ Log full comment text
- ❌ **Access internet during classification** (enforced with `local_files_only=True`)

### Internet Access Policy

**Classification (Inference)**: **100% Offline** ✅

- Model loading uses `local_files_only=True` flag
- All predictions run on local model files
- No network requests during processing
- Guaranteed air-gapped operation

**Training**: **Internet Required ONLY for Initial Setup** ⚠️

- First-time training downloads base model (e.g., `xlm-roberta-base` ~500MB)
- Downloaded once from Hugging Face Hub
- Cached locally for future use
- Subsequent training/refinement uses local cache

**Portable .exe**: **Fully Offline After Setup** ✅

- Includes pre-trained model in distribution
- No internet needed for classification
- Optional: Training requires internet only for base model download

### Logging Policy

**What IS logged:**

- Processing errors and exceptions
- File names and row counts
- Skip reasons and data quality metrics
- Model predictions (categories only, not full text)
- System performance metrics

**What is NOT logged:**

- Full comment text content
- Personal information from comments
- User identity or authentication details
- File contents beyond metadata

### Input Validation (3 Layers)

1. **File Upload**: Extension, size (<100MB), format validation
2. **Data Loading**: Column presence, text column validation
3. **Processing**: Length, type, encoding checks

### Deployment Security

**For Portable .exe Users:**

- Run in isolated directory (no admin needed)
- Logs stored locally in `logs/`
- Config in `config.yaml` (review before sharing)

**For Developers:**

- Use virtual environment: `python -m venv venv`
- Dependency scanning: `pip-audit`
- Update packages: `pip install --upgrade -r requirements.txt`

**For Organizations:**

- Review `config.yaml` before deployment
- Configure logging levels appropriately
- Restrict file upload directories
- Use model versioning to rollback if needed

### Vulnerability Reporting

Report security issues to: **volodymyrdunaiev@gmail.com**

---

## 📚 Documentation

**Quick Links:**

- **Version History**: See [CHANGELOG.md](CHANGELOG.md)
- **Issues/Support**: [GitHub Issues](https://github.com/Volodymyr-Dunaiev/employee_comment_analyzer/issues)

---

---

## 🛠️ Troubleshooting

| Issue                         | Solution                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------- |
| **"Model not found"**         | Run training first: `Train_Model.exe --data data\train.xlsx`                 |
| **"Out of memory"**           | Reduce batch_size to 4 in config.yaml                                        |
| **"CUDA not available"**      | Set `device: "cpu"` in config.yaml                                           |
| **"Invalid file format"**     | Ensure Excel has 'text' column with comments                                 |
| **"No labels column"**        | For training: add 'labels' column with categories                            |
| **Low F1 score (<0.5)**       | Need more training data (100+ samples recommended)                           |
| **Slow processing**           | Enable GPU: set `device: "cuda"` (NVIDIA) or `device: "mps"` (Apple Silicon) |
| **Import errors**             | Reinstall: `pip install -e .`                                                |
| **Permission denied**         | Run app from user directory, not system folders                              |
| **Excel won't open**          | Check file isn't already open in Excel                                       |
| **Model predictions wrong**   | Retrain with more diverse examples or adjust threshold                       |
| **Portable .exe won't start** | Extract ZIP fully, disable antivirus briefly, check Windows Defender         |

### Common Training Issues

- **"Validation split error"**: Need at least 5 samples per category
- **"NaN in loss"**: Lower learning rate (1e-5) or reduce batch size
- **"Overfitting"**: Reduce epochs or add more training data
- **"Unbalanced categories"**: Aim for similar sample counts per category

### Performance Tips

- **GPU acceleration**: Set `device: "cuda"` for 10-50× speedup
- **Batch size**: Larger = faster but needs more memory
- **Confidence threshold**: Lower (0.2) = more predictions, Higher (0.5) = fewer but more confident
- **Model choice**: xlm-roberta-base best for Ukrainian

---

## 📄 License

MIT License - see LICENSE file for details.

---

The `skip_reason` column provides detailed categorization of why texts are skipped during classification.

**Skip Reason Values:**

| Value        | Meaning                             | Example            |
| ------------ | ----------------------------------- | ------------------ |
| `none`       | Text processed successfully         | "Great management" |
| `empty`      | Text is empty string                | `""`               |
| `whitespace` | Text contains only whitespace       | `"   \n\t  "`      |
| `nan`        | Text is pandas NaN or string "nan"  | `NaN` or `"nan"`   |
| `non_text`   | Text is not a string (number, bool) | `123` or `True`    |

**Output Format:**

```python
# API output
result = classifier.classify_batch(["Good", "", "Bad"])
# {
#     'Category1': [0.8, 0.0, 0.3],
#     'skip_reason': ['none', 'empty', 'none'],
#     '_metadata': {
#         'skipped_indices': [1],
#         'skip_reason_counts': {'none': 2, 'empty': 1}
#     }
# }
```

**Analyzing Skip Reasons:**

```python
# Read output file
df = pd.read_excel('classified_output.xlsx')

# Filter processed rows
processed_df = df[df['skip_reason'] == 'none']

# Count skip reasons
skip_counts = df['skip_reason'].value_counts()
print(skip_counts)
```

## Support

**Troubleshooting:** See [QUICKSTART.md](QUICKSTART.md#-troubleshooting-guide) for common issues and solutions.

**Issues:** For bugs and feature requests, use the [GitHub issue tracker](https://github.com/yourusername/comments-classifier/issues).

**Security:** For security vulnerabilities, see [SECURITY.md](SECURITY.md#vulnerability-reporting).
