@echo off
setlocal enabledelayedexpansion

echo ===============================================================================
echo     AI-Based Network Attack Forecasting System (SIH26153 - NTRO)
echo     World Model & SOC Decision Support Dashboard
echo ===============================================================================
echo.

:: Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [*] Checking Python environment...
python --version

:: Virtual environment setup (optional, uses existing if present)
if not exist "venv" (
    echo [*] Virtual environment not found. Setting up 'venv'...
    python -m venv venv
)

if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Verify dependencies
echo [*] Verifying required packages...
pip install -r requirements.txt --quiet

:: Ensure base dataset and model weights exist
if not exist "data\processed\fused_state.csv" (
    echo [*] Initializing sample multi-stage traffic datasets...
    python src\utils\synthetic_generator.py
    python src\feature_pipeline\flow_extractor.py --input data\raw\synthetic_flows.csv --output data\processed\flow_features.csv
    python src\feature_pipeline\fusion.py --flow data\processed\flow_features.csv --output data\processed\fused_state.csv
)

if not exist "model\weights\world_model_final.pt" (
    echo [*] Training initial World Model dynamics weights...
    python src\training\train.py --config configs\lstm_attention.yaml
)

echo.
echo ===============================================================================
echo  [SUCCESS] All systems verified and ready.
echo  Starting Streamlit SOC Dashboard on http://localhost:8501
echo ===============================================================================
echo.

:: Auto-open browser in background
start "" http://localhost:8501

:: Run Streamlit app
streamlit run src\app.py --server.port=8501 --server.headless=false

pause
