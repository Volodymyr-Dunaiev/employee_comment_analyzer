@echo off
REM Package the complete application for distribution
REM Creates a zip file ready to transfer to any Windows laptop

setlocal
echo ========================================
echo Creating Distribution Package
echo ========================================
echo.

REM Check if executables exist
if not exist "dist\Comments_Classifier\Comments_Classifier.exe" (
    echo ERROR: Comments_Classifier.exe not found!
    echo Please run scripts\build_exe.bat first
    pause
    exit /b 1
)

if not exist "dist\Train_Model\Train_Model.exe" (
    echo ERROR: Train_Model.exe not found!
    echo Please run scripts\build_exe.bat first
    pause
    exit /b 1
)

REM Create distribution folder
set DIST_NAME=Comments_Classifier_Portable
if exist "%DIST_NAME%" rmdir /s /q "%DIST_NAME%"
mkdir "%DIST_NAME%"

echo Copying executables and dependencies...
xcopy /E /I /Y "dist\Comments_Classifier" "%DIST_NAME%\Comments_Classifier"
xcopy /E /I /Y "dist\Train_Model" "%DIST_NAME%\Train_Model"

echo Copying configuration...
copy config.yaml "%DIST_NAME%\"

echo Creating data folders...
mkdir "%DIST_NAME%\data"
mkdir "%DIST_NAME%\model"
mkdir "%DIST_NAME%\logs"

REM Copy model if it exists
if exist "model\ukr_multilabel" (
    echo Copying trained model...
    xcopy /E /I /Y "model\ukr_multilabel" "%DIST_NAME%\model\ukr_multilabel"
)

REM Create quick start guide
echo Creating user guide...
(
echo ================================================================
echo Ukrainian Comments Classifier - Portable Edition
echo ================================================================
echo.
echo QUICK START:
echo   1. Double-click "START_APP.bat" to start the application
echo   2. OR run Comments_Classifier\Comments_Classifier.exe directly
echo   3. App will open in your browser at http://localhost:8501
echo   4. Upload Excel file with comments and classify them
echo.
echo TRAINING A MODEL:
echo   1. Prepare training data ^(Excel file with 'text' and 'labels' columns^)
echo   2. Save it in the 'data' folder
echo   3. Double-click "TRAIN_MODEL.bat" OR
echo   4. Open Command Prompt and run: Train_Model\Train_Model.exe --data data\your_file.xlsx --epochs 3
echo   5. After training, update config.yaml to point to your model
echo.
echo SYSTEM REQUIREMENTS:
echo   - Windows 10/11
echo   - 4GB RAM minimum ^(8GB recommended^)
echo   - No Python installation needed!
echo.
echo FOLDER STRUCTURE:
echo   START_APP.bat            - Quick launcher for main app
echo   TRAIN_MODEL.bat          - Training wizard
echo   Comments_Classifier\     - Main application folder
echo   Train_Model\             - Training utility folder
echo   config.yaml              - Application settings
echo   data\                    - Place training files here
echo   model\                   - Trained models stored here
echo   logs\                    - Application logs
echo.
echo TRAINING DATA FORMAT:
echo   Your Excel file should have:
echo   - A 'text' column with comments
echo   - Either:
echo     * A 'labels' column with categories ^(comma-separated^)
echo     * OR separate columns: 'Category 1', 'Category 2', etc.
echo.
echo EXAMPLES:
echo   Train new model:
echo     Train_Model\Train_Model.exe --data data\train.xlsx --epochs 3
echo.
echo   Quick training ^(1 epoch^):
echo     Train_Model\Train_Model.exe --data data\train.xlsx --epochs 1 --batch_size 4
echo.
echo   Refine existing model:
echo     Train_Model\Train_Model.exe --data data\more_data.xlsx --model_name_or_path model\ukr_multilabel --epochs 2
echo.
echo TROUBLESHOOTING:
echo   - Port in use: Close other apps using port 8501
echo   - Slow performance: Reduce batch_size when training
echo   - Out of memory: Use smaller batch_size or fewer epochs
echo.
echo NEED HELP?
echo   Check logs\ folder for error details
echo.
echo ================================================================
) > "%DIST_NAME%\README.txt"

REM Create training examples
echo Creating example training template...
(
echo text,Category 1,Category 2,Category 3
echo "Sample comment 1","Зарплата","",""
echo "Sample comment 2","Колектив","Умови праці",""
echo "Sample comment 3","Керівництво","","
) > "%DIST_NAME%\data\training_template.csv"

REM Create sample config
echo Creating sample configuration...
copy config.yaml "%DIST_NAME%\config.yaml.sample"

REM Create quick launch scripts
echo Creating quick launch scripts...
(
echo @echo off
echo start /B Comments_Classifier\Comments_Classifier.exe
echo echo Application started! Opening in browser...
echo timeout /t 3 /nobreak ^>nul
echo start http://localhost:8501
) > "%DIST_NAME%\START_APP.bat"

(
echo @echo off
echo echo ========================================
echo echo Model Training Wizard
echo echo ========================================
echo echo.
echo set /p DATAFILE="Enter training file path (e.g., data\train.xlsx): "
echo set /p EPOCHS="Number of epochs (default 3): "
echo if "%%EPOCHS%%"=="" set EPOCHS=3
echo.
echo Train_Model\Train_Model.exe --data "%%DATAFILE%%" --epochs %%EPOCHS%%
echo pause
) > "%DIST_NAME%\TRAIN_MODEL.bat"

echo.
echo Creating ZIP archive...
powershell Compress-Archive -Path "%DIST_NAME%\*" -DestinationPath "%DIST_NAME%.zip" -Force

echo.
echo ========================================
echo Package Complete!
echo ========================================
echo.
echo Distribution package created:
echo   Folder: %DIST_NAME%\
echo   Archive: %DIST_NAME%.zip
echo.
echo Total size:
dir "%DIST_NAME%.zip" | find ".zip"
echo.
echo Transfer %DIST_NAME%.zip to any Windows laptop
echo Extract and run Comments_Classifier.exe
echo No Python or dependencies needed!
echo.
pause
endlocal
