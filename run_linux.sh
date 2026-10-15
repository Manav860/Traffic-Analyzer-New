#!/usr/bin/env bash
# ===============================================================================
# AI-Based Network Attack Forecasting System (SIH26153 - NTRO)
# One-Click Linux/macOS Launcher
# ===============================================================================

set -e

echo "==============================================================================="
echo "    AI-Based Network Attack Forecasting System (SIH26153 - NTRO)"
echo "    World Model & SOC Decision Support Dashboard"
echo "==============================================================================="
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "[*] Checking Python environment..."
python3 --version

# Virtual environment setup
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "[*] Activating virtual environment..."
source venv/bin/activate

# Verify dependencies
echo "[*] Verifying required packages..."
pip install -r requirements.txt --quiet

# Ensure base dataset and model weights exist
if [ ! -f "data/processed/fused_state.csv" ]; then
    echo "[*] Initializing sample multi-stage traffic datasets..."
    python3 src/utils/synthetic_generator.py
    python3 src/feature_pipeline/flow_extractor.py --input data/raw/synthetic_flows.csv --output data/processed/flow_features.csv
    python3 src/feature_pipeline/fusion.py --flow data/processed/flow_features.csv --output data/processed/fused_state.csv
fi

if [ ! -f "model/weights/world_model_final.pt" ]; then
    echo "[*] Training initial World Model dynamics weights..."
    python3 src/training/train.py --config configs/lstm_attention.yaml
fi

echo ""
echo "==============================================================================="
echo " [SUCCESS] All systems verified and ready."
echo " Starting Streamlit SOC Dashboard on http://localhost:8501"
echo "==============================================================================="
echo ""

# Attempt to open browser automatically
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:8501 &
elif command -v open &> /dev/null; then
    open http://localhost:8501 &
fi

# Run Streamlit
streamlit run src/app.py --server.port=8501
