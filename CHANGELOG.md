# Changelog

All notable changes to this project will be documented in this file.

## [2.1.0] - 2025-10-24

### Critical Production Fixes

- **Browser Auto-Launch Fix** (`launcher.py`):

  - Complete rewrite from blocking `stcli.main()` to subprocess approach
  - Threading for delayed browser opening (3s delay)
  - Proper process management with terminate() on exit
  - CREATE_NO_WINDOW flag for Windows clean execution
  - **Impact**: Browser now auto-opens correctly in frozen .exe

- **Config Path Resolution** (`src/utils/config.py`):

  - Detects exe vs script execution mode using `sys.frozen`
  - Resolves config.yaml relative to exe location, not working directory
  - Works from desktop shortcuts, any launch location
  - **Impact**: Config loads regardless of launch location

- **Auto-Memory Management** (`src/utils/memory.py`):

  - Automatically detects available RAM using psutil
  - Tunes batch size to prevent OOM crashes
  - Conservative: 30MB per sample, reserves 1GB for OS + 500MB for model
  - Caps batch size between 1-32 for safety
  - Integration: `classifier.py` auto-tunes on initialization
  - **Impact**: Prevents crashes on 4GB laptops, maximizes performance on 16GB

- **Training Data Validation** (`src/utils/training_validation.py`):

  - Validates minimum 50 samples total, 5 per category
  - Detects empty text, duplicates (>10%), very short texts (<10 chars)
  - Warns about class imbalance (>20x ratio)
  - Provides actionable quality recommendations
  - Integration: `train.py` validates before training starts
  - **Impact**: Prevents wasting hours on bad training data

- **Model Versioning** (`src/core/train.py`):
  - Automatically backs up existing models to `model_backups/v{N}/`
  - JSON metadata with timestamps for each version
  - Easy rollback if new model performs poorly
  - See `model_backups/README.md` for restore guide
  - **Impact**: Never lose a good model again

### Important Improvements

- **PyInstaller Build Script** (`scripts/build_exe.bat`):

  - Fixed --add-data syntax for Windows (semicolon separator)
  - Added missing hidden imports: PIL, openpyxl.cell.\_writer, psutil
  - Changed to --collect-submodules for transformers, torch, datasets
  - Removed optional --icon flag (see APP_ICON_NOTE.md)
  - **Impact**: Builds complete and functional

- **Modern Packaging** (`pyproject.toml`):
  - Migrated from deprecated setup.py to pyproject.toml
  - Standard PEP 621 metadata format
  - setup.py now minimal shim for backward compatibility
  - **Impact**: No more pip deprecation warnings

### Dependencies

- Added `psutil>=5.8.0` for memory monitoring

### Documentation

- Updated `DEPLOYMENT_GUIDE.md` with new features and workflows
- Added `model_backups/README.md` for version management
- Added `APP_ICON_NOTE.md` explaining icon is optional
- Enhanced this CHANGELOG with detailed impact notes

### Performance Improvements

- **Lazy-Loading Architecture**: Implemented deferred imports for significant startup performance gains
  - Pipeline module import time reduced from 10.07s to 1.29s (87% faster)
  - ML libraries (torch, transformers) now load only when needed
  - Memory footprint reduced by 500MB-2GB until classification is executed
  - `CommentClassifier` import moved inside `_get_classifier()` function

### Code Optimization

- **Removed Redundant Code**:

  - Deleted `src/core/train_utils.py` (duplicate file, functions inlined)
  - Removed 9+ unused imports from `pipeline.py`
  - Deleted unused `process_batch_async` function (47 lines)
  - Removed duplicate `setup_logger` from `config.py`
  - Removed unused `load_cached_config` function

- **Consolidated Functions**:
  - Inlined `save_label_encoder`, `safe_save_model_and_tokenizer`, and `export_onnx` in `train.py`
  - Single source of truth for all helper functions
  - Reduced codebase by 16% (10 files vs 12 files)

### Documentation

- Added `CLEANUP_SUMMARY.md` with detailed performance optimization documentation
- Added `CLEANUP_COMPLETE.md` as quick reference guide
- Created `test_performance.py` for performance verification
- Updated README with performance metrics and optimization details

### Technical Details

- Lazy-loading prevents expensive model initialization at import time
- Global classifier instance creation deferred until `run_inference()` is called
- No breaking changes - all functionality preserved
- All tests passing after optimization

## [2.0.1] - 2025-10-24

### Added

- **Multi-Column Label Support**: Training data can now have labels in multiple columns
  - Support for `label_*`, `category_*`, or custom column patterns
  - Automatic detection and combination of multi-column labels
  - UI option to select label format (single vs. multiple columns)
  - Updated documentation with examples for both formats

### Fixed

- Replaced deprecated `load_metric` with `evaluate.load()` for metrics computation
- Removed unused `train_model` import that was causing test failures
- All docstrings converted to `#` comments for proper syntax highlighting

## [2.0.0] - 2025-10-24

### Added

- **Training Interface**: Complete UI for training custom models
  - Upload training data (CSV/Excel)
  - Configure training parameters (epochs, batch size, learning rate)
  - Real-time progress tracking
  - Training metrics display (F1, precision, recall)
  - Model saving and metadata export
- **New Modules**:
  - `src/core/train_interface.py`: Training UI wrapper
  - `src/core/model_utils.py`: Model loading utilities
  - `src/core/errors.py`: Custom exception classes
  - `src/core/classifier.py`: CommentClassifier class
- **Enhanced Testing**:
  - `tests/test_training_ui.py`: Training interface tests
  - Data validation tests
  - Parameter validation tests
  - Integration tests for training workflow
- **Documentation**:
  - `QUICKSTART.md`: Quick start guide
  - Enhanced README with training instructions
  - Commented configuration file
  - Training data format examples

### Changed

- **UI Improvements**:
  - Added tabbed interface (Classification + Training)
  - Enhanced error messages
  - Better progress tracking
  - Improved metrics display
- **Code Organization**:
  - Split classifier logic into separate module
  - Improved error handling with custom exceptions
  - Better separation of concerns
  - Added type hints throughout
- **Configuration**:
  - Added CSV support to allowed file types
  - Enhanced comments in config.yaml
  - Better validation for configuration values
- **Requirements**:
  - Added `scikit-learn` for training
  - Added `datasets` for data processing
  - Added `joblib` for model serialization
  - Organized requirements with comments

### Fixed

- Circular import issues between modules
- Missing import statements
- Validation for empty DataFrames
- Token dictionary `.to()` method issues in tests
- Progress callback handling

### Security

- Enhanced file validation
- Input sanitization for training data
- Size limits enforcement
- Safe file handling in training pipeline

## [1.0.0] - Previous Version

### Initial Features

- Multi-label classification
- Streamlit UI
- Excel file processing
- Batch processing
- Configuration management
- Logging system
- Basic testing suite
