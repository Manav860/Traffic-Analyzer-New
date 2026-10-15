"""
Training Pipeline for Network World Model and Baseline.
Implements multi-task dynamics learning P(S_t+1 | S_t) with class imbalance handling,
early stopping, and persistent checkpointing to model/weights/.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seeding import set_seed
from src.utils.synthetic_generator import generate_synthetic_flows
from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.fusion import DualLevelFusionPipeline, STATE_FEATURE_NAMES
from src.feature_pipeline.dataset_loader import StateDatasetManager, NetworkStateSequenceDataset
from src.model.world_model import WorldModelLSTM
from src.model.transformer_world_model import TemporalTransformerWorldModel
from src.model.baseline import LogisticRegressionBaseline


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataset_and_baseline(
    config: Dict[str, Any]
) -> Tuple[StateDatasetManager, Dict[str, Any], LogisticRegressionBaseline]:
    """Ensure data is ready, compute splits, and fit baseline."""
    data_path = Path(config["paths"]["train_data"])
    
    # Auto-generate if missing
    if not data_path.exists():
        print(f"[+] Dataset not found at {data_path}. Auto-generating realistic synthetic traffic dataset...")
        raw_flows = generate_synthetic_flows(num_samples=25000, random_seed=config["experiment"]["random_seed"])
        extractor = FlowFeatureExtractor(window_size_sec=config["dataset"]["window_size_sec"])
        flow_feat = extractor.extract_windowed_features(raw_flows)
        fusion = DualLevelFusionPipeline()
        fused_df = fusion.fuse_dataframes(df_flow=flow_feat)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        fused_df.to_csv(data_path, index=False)
        print(f"[+] Saved {len(fused_df)} fused state windows to {data_path}")
    else:
        fused_df = pd.read_csv(data_path)

    # Initialize Dataset Manager
    manager = StateDatasetManager(
        feature_cols=STATE_FEATURE_NAMES,
        seq_len=config["dataset"]["sequence_length"],
        horizon_k=config["dataset"]["rollout_horizon_k"],
        stride=config["dataset"]["stride"],
        scaler_path=config["paths"].get("scaler_path", "model/weights/scaler.pkl")
    )

    splits = manager.prepare_train_val_test(
        df=fused_df,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_ratio=config["dataset"]["test_split"],
        held_out_category=config["dataset"].get("held_out_attack_category", "Lateral Movement")
    )

    # Train Baseline Model
    print("[+] Training Logistic Regression Baseline...")
    x_tr, _, y_inf_tr, y_sg_tr = splits["in_distribution"]["train"]
    baseline = LogisticRegressionBaseline(random_state=config["experiment"]["random_seed"])
    baseline.fit(x_tr, y_inf_tr, y_sg_tr)
    
    baseline_path = config["paths"].get("baseline_checkpoint", "model/weights/baseline_final.pkl")
    baseline.save(baseline_path)
    print(f"[+] Baseline model saved to: {baseline_path}")

    return manager, splits, baseline


def train_world_model(
    config_path: str = "configs/lstm_attention.yaml",
    force_cpu: bool = False
) -> Dict[str, Any]:
    """Train the World Model dynamics learner."""
    config = load_config(config_path)
    set_seed(config["experiment"]["random_seed"])

    # Device selection
    if force_cpu or config["training"].get("device") == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available() and config["training"].get("device") != "cpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[+] Training on device: {device}")

    # Prepare datasets
    manager, splits, baseline = prepare_dataset_and_baseline(config)

    x_tr, y_st_tr, y_inf_tr, y_sg_tr = splits["in_distribution"]["train"]
    x_va, y_st_va, y_inf_va, y_sg_va = splits["in_distribution"]["val"]

    train_ds = NetworkStateSequenceDataset(x_tr, y_st_tr, y_inf_tr, y_sg_tr)
    val_ds = NetworkStateSequenceDataset(x_va, y_st_va, y_inf_va, y_sg_va)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Instantiate Model
    model_type = config["model"].get("type", "lstm_attention")
    if model_type == "temporal_transformer":
        model = TemporalTransformerWorldModel(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_heads=config["model"].get("num_heads", 4),
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"].get("dim_feedforward", 256),
            dropout=config["model"]["dropout"],
            num_stages=config["model"]["num_stages"]
        )
    else:
        model = WorldModelLSTM(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            attention_dim=config["model"].get("attention_dim", 64),
            dropout=config["model"]["dropout"],
            num_stages=config["model"]["num_stages"]
        )

    model.to(device)

    # Loss functions
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    ce_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    alpha_dyn = config["training"].get("loss_alpha_dynamics", 0.5)
    beta_cls = config["training"].get("loss_beta_classification", 0.5)
    epochs = config["training"]["epochs"]
    patience = config["training"].get("patience", 5)

    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = Path(config["paths"]["model_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting World Model Training ({model_type.upper()}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_dyn_loss = 0.0
        train_infil_loss = 0.0
        train_stage_loss = 0.0

        for x_b, y_st_b, y_inf_b, y_sg_b in train_loader:
            x_b = x_b.to(device)
            y_next_st = y_st_b[:, 0, :].to(device) # immediate next state
            y_next_inf = y_inf_b[:, 0].unsqueeze(1).to(device)
            y_next_sg = y_sg_b[:, 0].to(device)

            optimizer.zero_grad()
            out = model(x_b)

            l_dyn = mse_criterion(out["next_state"], y_next_st)
            l_inf = bce_criterion(out["infil_prob"], y_next_inf)
            l_sg = ce_criterion(out["stage_logits"], y_next_sg)

            loss = (alpha_dyn * l_dyn) + (beta_cls * (0.5 * l_inf + 0.5 * l_sg))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item() * len(x_b)
            train_dyn_loss += l_dyn.item() * len(x_b)
            train_infil_loss += l_inf.item() * len(x_b)
            train_stage_loss += l_sg.item() * len(x_b)

        n_train = len(train_ds)
        train_loss_total /= max(n_train, 1)

        # Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x_b, y_st_b, y_inf_b, y_sg_b in val_loader:
                x_b = x_b.to(device)
                y_next_st = y_st_b[:, 0, :].to(device)
                y_next_inf = y_inf_b[:, 0].unsqueeze(1).to(device)
                y_next_sg = y_sg_b[:, 0].to(device)

                out = model(x_b)
                l_dyn = mse_criterion(out["next_state"], y_next_st)
                l_inf = bce_criterion(out["infil_prob"], y_next_inf)
                l_sg = ce_criterion(out["stage_logits"], y_next_sg)

                loss = (alpha_dyn * l_dyn) + (beta_cls * (0.5 * l_inf + 0.5 * l_sg))
                val_loss_total += loss.item() * len(x_b)

        n_val = len(val_ds)
        val_loss_total /= max(n_val, 1)
        scheduler.step(val_loss_total)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss_total:.4f} (Dyn: {train_dyn_loss/max(n_train, 1):.4f}, Infil: {train_infil_loss/max(n_train, 1):.4f}) "
            f"| Val Loss: {val_loss_total:.4f}"
        )

        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_loss": best_val_loss
            }, checkpoint_path)
            print(f"  [*] Checkpoint saved -> {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}")
                break

    print(f"\n[+] Training Complete. Best Checkpoint: {checkpoint_path}")
    return {"status": "success", "best_val_loss": best_val_loss, "checkpoint": str(checkpoint_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Model & Baseline")
    parser.add_argument("--config", default="configs/lstm_attention.yaml", help="Path to config yaml")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    train_world_model(config_path=args.config, force_cpu=args.cpu)
