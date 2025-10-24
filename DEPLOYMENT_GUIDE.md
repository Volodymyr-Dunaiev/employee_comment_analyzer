# Comments Classifier - Portable Desktop Application

## For End Users (Any Windows Laptop)

### Quick Start

1. Extract `Comments_Classifier_Portable.zip`
2. Double-click `Comments_Classifier.exe` or `START_APP.bat`
3. App opens automatically in your browser at http://localhost:8501
4. Upload Excel → Classify → Download results

**No Python installation required!**

### New Features (v2.1.0)

✅ **Auto-Memory Tuning**: Batch size automatically adjusts to available RAM  
✅ **Config Auto-Detection**: Works from desktop shortcuts, any folder  
✅ **Training Validation**: Checks data quality before wasting time on bad training  
✅ **Model Versioning**: Previous models auto-backed up before overwrite  
✅ **Modern Packaging**: Uses pyproject.toml (no more pip deprecation warnings)

---

## Initial Model Training

### Option 1: Train on the target laptop

1. Prepare your training data (Excel file):

   - Column `text` with comments
   - Column `labels` with categories (comma-separated)
   - OR separate columns: `Category 1`, `Category 2`, etc.
   - **Minimum**: 50 total samples, 5 samples per category

2. Save training file in `data\` folder

3. Open Command Prompt in the app folder

4. Run training:

   ```cmd
   Train_Model.exe --data data\train.xlsx --epochs 3
   ```

   **Training validation checks:**

   - Minimum sample requirements
   - Empty/duplicate text detection
   - Class imbalance warnings
   - Quality recommendations

5. Previous model auto-backed up to `model_backups/v1/`

6. Update `config.yaml` if needed:

   ```yaml
   model:
     path: model/ukr_multilabel
   ```

7. Restart the app

### Option 2: Train on your dev machine, transfer model

1. Train on your development laptop (with Python):

   ```cmd
   python -m src.core.train --data data\train.xlsx --epochs 3
   ```

2. Copy the entire `model\ukr_multilabel` folder

3. Paste into the portable app's `model\` folder on target laptop

4. Update `config.yaml` on target laptop to point to the model

---

## Model Refinement/Updates

### On any laptop (even without Python):

1. Collect new training samples (same Excel format, minimum 50 samples)

2. Add to `data\new_samples.xlsx`

3. Refine the existing model:

   ```cmd
   Train_Model.exe --data data\new_samples.xlsx --model_name_or_path model\ukr_multilabel --epochs 2
   ```

4. **Previous model auto-saved** to `model_backups/v2/` (version increments)

5. Restart the app to use updated model

### Restoring a previous model version:

If new model performs poorly:

```powershell
# Windows
Remove-Item -Recurse -Force model
Copy-Item -Recurse model_backups/v1 model
```

See `model_backups/README.md` for details.

### Tips for refinement:

- Use fewer epochs (1-2) when refining
- Can train on smaller batches: `--batch_size 4`
- Model updates incrementally without starting from scratch
- Validation runs automatically (checks quality before training)
- Memory auto-tunes based on available RAM (4GB minimum recommended)

---

## For Developers

### Build the portable package:

1. **One-time setup** (on your dev machine):

   ```cmd
   scripts\build_complete.bat
   ```

   This creates `Comments_Classifier_Portable.zip` containing:

   - `Comments_Classifier.exe` (main app)
   - `Train_Model.exe` (training utility)
   - `config.yaml` (settings)
   - `data\`, `model\`, `logs\` folders
   - Documentation and quick-start scripts

2. **Transfer to any laptop**:

   - Copy the `.zip` file
   - Extract anywhere
   - No installation needed

3. **Update and rebuild**:
   ```cmd
   REM After code changes:
   scripts\build_exe.bat
   scripts\package_distribution.bat
   ```

### Development workflow:

- **Local testing**: `streamlit run src/ui/app_ui.py`
- **Build portable**: `scripts\build_complete.bat`
- **Distribute**: Share `Comments_Classifier_Portable.zip`

---

## Training Data Formats

The app supports three formats:

### Format 1: Single labels column

```
| text                    | labels                    |
|------------------------|---------------------------|
| Sample comment 1       | Зарплата,Колектив        |
| Sample comment 2       | Умови праці              |
```

### Format 2: Category columns (recommended)

```
| text              | Category 1  | Category 2    | Category 3 |
|------------------|-------------|---------------|------------|
| Sample comment 1 | Зарплата    | Колектив      |            |
| Sample comment 2 | Умові праці |               |            |
```

### Format 3: Pattern columns

```
| text              | label_1     | label_2       |
|------------------|-------------|---------------|
| Sample comment 1 | Зарплата    | Колектив      |
```

All formats work with both the UI training tab and `Train_Model.exe`.

---

## System Requirements

### End users:

- Windows 10/11
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Developers:

- Python 3.11+
- 8GB RAM minimum
- Git (optional)

---

## Troubleshooting

### App won't start:

- Check if port 8501 is in use
- Check `logs\app.log` for errors
- Try running from Command Prompt to see output

### Training fails:

- Ensure data file has `text` column
- Reduce `--batch_size` if out of memory
- Check training data format matches examples

### Model not found:

- Verify `config.yaml` `model.path` points to correct folder
- Ensure model folder has `config.json` and `pytorch_model.bin`

---

## File Structure

```
Comments_Classifier_Portable/
├── Comments_Classifier.exe    # Main app
├── Train_Model.exe            # Training utility
├── config.yaml                # Settings
├── START_APP.bat              # Quick launcher
├── TRAIN_MODEL.bat            # Training wizard
├── README.txt                 # User guide
├── data/                      # Training data
│   └── training_template.csv
├── model/                     # Trained models
│   └── ukr_multilabel/
├── logs/                      # Application logs
```

---

## Advanced Options

### Custom App Icon (Optional)

The built executables use the default Python icon. To add a custom icon:

1. **Create/download an `.ico` file** (16x16, 32x32, 48x48, 256x256 pixels)

   - Free tools: https://www.favicon-generator.org/ or GIMP

2. **Place `app_icon.ico` in project root**

3. **Edit `scripts/build_exe.bat`** and add `--icon=app_icon.ico` flag:

   ```bat
   pyinstaller --clean ^
     --name "Comments_Classifier" ^
     --icon=app_icon.ico ^
     ...
   ```

4. **Rebuild**: `.\scripts\build_exe.bat`

### Training parameters:

```cmd
Train_Model.exe --help

Options:
  --data              Training file path (required)
  --epochs            Number of epochs (default: 3)
  --batch_size        Batch size (default: 8)
  --lr                Learning rate (default: 2e-5)
  --model_dir         Output directory (default: model\ukr_multilabel)
  --text_column       Text column name (default: text)
  --labels_column     Labels column name (default: labels)
```

### Config.yaml options:

```yaml
model:
  path: model/ukr_multilabel # Model location
  device: cpu # cpu or cuda
  batch_size: 16 # Inference batch size (auto-tuned by memory)

data:
  chunk_size: 1000 # Rows per chunk
  input_validation:
    max_length: 5000 # Max text length
```
