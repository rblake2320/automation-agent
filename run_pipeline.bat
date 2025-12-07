@echo off
REM VBScript/Selenium Automation Agent - Full Training Pipeline
REM Optimized for RTX 5090 (32GB VRAM)

echo ============================================================
echo  VBScript/Selenium Automation Agent Training Pipeline
echo  RTX 5090 Optimized
echo ============================================================
echo.

REM Check if conda environment exists
call conda activate automation-agent 2>nul
if %errorlevel% neq 0 (
    echo Creating conda environment...
    call conda create -n automation-agent python=3.11 -y
    call conda activate automation-agent
    echo Installing requirements...
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation
)

echo.
echo Step 1: Collecting GitHub Data
echo ============================================================
python src/scrapers/github_scraper.py --max-repos 30

echo.
echo Step 2: Collecting Stack Overflow Data
echo ============================================================
python src/scrapers/stackoverflow_scraper.py --max-per-category 300

echo.
echo Step 3: Processing Data
echo ============================================================
python src/processors/data_processor.py

echo.
echo Step 4: Starting Training
echo ============================================================
python src/training/train.py train

echo.
echo Step 5: Merging LoRA
echo ============================================================
python src/training/train.py merge

echo.
echo Step 6: Exporting to GGUF
echo ============================================================
python src/training/train.py export

echo.
echo Step 7: Creating Ollama Model
echo ============================================================
python src/training/train.py ollama

echo.
echo ============================================================
echo  Pipeline Complete!
echo  Run: ollama run automation-agent
echo ============================================================
pause
