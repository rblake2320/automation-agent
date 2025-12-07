@echo off
REM Continue Automation Agent Training Pipeline
REM Run this after data collection is complete

cd /d C:\Users\techai\automation-agent

echo ============================================================
echo  Automation Agent - Continue Training Pipeline
echo ============================================================
echo.

echo Step 1: Processing collected data...
python src/processors/data_processor.py
if %errorlevel% neq 0 (
    echo ERROR: Data processing failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Starting LoRA training with Axolotl...
echo This will take 1-3 hours on RTX 5090
python src/training/train.py train
if %errorlevel% neq 0 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Merging LoRA adapter...
python src/training/train.py merge

echo.
echo Step 4: Exporting to GGUF...
python src/training/train.py export

echo.
echo Step 5: Creating Ollama model...
python src/training/train.py ollama

echo.
echo ============================================================
echo  COMPLETE! Run: ollama run automation-agent
echo ============================================================
pause
