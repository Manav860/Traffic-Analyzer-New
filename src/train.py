"""
CLI Entrypoint for Model Training.
Forwards to src/training/train.py
"""

import sys
from pathlib import Path

# Append project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.training.train import train_world_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Model & Baseline")
    parser.add_argument("--config", default="configs/lstm_attention.yaml", help="Path to config yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    train_world_model(config_path=args.config, force_cpu=args.cpu)

