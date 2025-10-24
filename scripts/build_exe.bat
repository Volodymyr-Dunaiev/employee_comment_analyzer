@echo off
REM Build standalone executable using PyInstaller
REM This creates a distributable app that doesn't require Python installation

setlocal
echo ========================================
echo Building Comments Classifier Standalone
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate

REM Install PyInstaller if not present
pip install pyinstaller

REM Install UPX for better compression (optional but recommended)
echo Installing UPX for compression...
pip install pyinstaller[encryption]

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo Building main application...
pyinstaller --clean ^
  --name "Comments_Classifier" ^
  --onedir ^
  --windowed ^
  --add-data "config.yaml;." ^
  --add-data "src;src" ^
  --hidden-import "streamlit" ^
  --hidden-import "torch" ^
  --hidden-import "transformers" ^
  --hidden-import "sklearn" ^
  --hidden-import "pandas" ^
  --hidden-import "openpyxl" ^
  --hidden-import "openpyxl.cell._writer" ^
  --hidden-import "PIL" ^
  --hidden-import "PIL._imagingtk" ^
  --hidden-import "PIL.Image" ^
  --hidden-import "psutil" ^
  --collect-submodules "streamlit" ^
  --collect-submodules "transformers" ^
  --collect-submodules "torch" ^
  --collect-all "altair" ^
  --collect-all "plotly" ^
  --exclude-module "matplotlib" ^
  --exclude-module "IPython" ^
  --exclude-module "notebook" ^
  --exclude-module "jupyter" ^
  --exclude-module "scipy" ^
  --exclude-module "pytest" ^
  --exclude-module "PIL.ImageQt" ^
  --exclude-module "PyQt5" ^
  --exclude-module "PySide2" ^
  --exclude-module "tkinter" ^
  launcher.py

echo.
echo Building training utility...
pyinstaller --clean ^
  --name "Train_Model" ^
  --onedir ^
  --console ^
  --add-data "config.yaml;." ^
  --add-data "src;src" ^
  --hidden-import "transformers" ^
  --hidden-import "torch" ^
  --hidden-import "sklearn" ^
  --hidden-import "datasets" ^
  --hidden-import "evaluate" ^
  --hidden-import "psutil" ^
  --collect-submodules "transformers" ^
  --collect-submodules "torch" ^
  --collect-submodules "datasets" ^
  --exclude-module "matplotlib" ^
  --exclude-module "IPython" ^
  --exclude-module "notebook" ^
  --exclude-module "jupyter" ^
  --exclude-module "scipy" ^
  --exclude-module "pytest" ^
  --exclude-module "streamlit" ^
  --exclude-module "altair" ^
  --exclude-module "plotly" ^
  train_launcher.py

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Executables created in dist/:
echo   - Comments_Classifier.exe (Main App)
echo   - Train_Model.exe (Training Utility)
echo.
echo Next: Run scripts\package_distribution.bat to create final package
pause
endlocal
