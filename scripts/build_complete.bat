@echo off
REM Setup development environment and build the portable app
REM Run this once on your development machine

setlocal
echo ================================================================
echo Comments Classifier - Complete Build Setup
echo ================================================================
echo.

REM Step 1: Create virtual environment
echo Step 1: Setting up Python environment...
if not exist ".venv" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
)

call .venv\Scripts\activate

REM Step 2: Install dependencies
echo.
echo Step 2: Installing dependencies...
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install pyinstaller

REM Step 3: Build executables
echo.
echo Step 3: Building standalone executables...
call scripts\build_exe.bat

REM Step 4: Package distribution
echo.
echo Step 4: Creating distribution package...
call scripts\package_distribution.bat

echo.
echo ================================================================
echo Build Complete!
echo ================================================================
echo.
echo You can now:
echo   1. Test locally: Comments_Classifier_Portable\Comments_Classifier.exe
echo   2. Transfer: Share Comments_Classifier_Portable.zip
echo   3. Deploy: Copy to any Windows laptop and run
echo.
echo The portable package includes:
echo   - Main app (no Python needed)
echo   - Training utility
echo   - Documentation
echo   - Sample data templates
echo.
pause
endlocal
