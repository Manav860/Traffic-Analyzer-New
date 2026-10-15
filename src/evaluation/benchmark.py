"""
Benchmarking & Evaluation Suite.
Evaluates World Model vs. Logistic Regression Baseline on:
1. Standard In-Distribution Test Split (F1, Precision, Recall, FPR)
2. Dedicated Held-Out Unseen Attack Category Split (Zero-Shot Generalization)
3. Multi-Step Forward Horizon Rollout Accuracy (t+1 through t+K)
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from src.utils.seeding import set_seed
from src.feature_pipeline.fusion import STATE_FEATURE_NAMES
from src.feature_pipeline.dataset_loader import StateDatasetManager
from src.model.baseline import LogisticRegressionBaseline
from src.model.world_model import WorldModelLSTM
from src.model.transformer_world_model import TemporalTransformerWorldModel
from src.inference.rollout import RolloutEngine
from src.mitre_mapping.stix_mapper import MitreMapper


def compute_metrics(y_true: np.ndarray, y_pred_binary: np.ndarray) -> Dict[str, float]:
    """Compute F1, Precision, Recall, and False Positive Rate."""
    y_t = y_true.astype(int).flatten()
    y_p = y_pred_binary.astype(int).flatten()

    f1 = float(f1_score(y_t, y_p, zero_division=0))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = float(fp / max(fp + tn, 1))
    else:
        fpr = 0.0

    return {
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "fpr": fpr
    }


def run_benchmark(config_path: str = "configs/lstm_attention.yaml") -> Dict[str, Any]:
    """Run full benchmarking suite and print comparative evaluation tables."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["experiment"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and splits
    data_path = Path(config["paths"]["train_data"])
    if not data_path.exists():
        from src.training.train import train_world_model
        print("[!] Dataset or model not found. Running training first...")
        train_world_model(config_path=config_path)

    fused_df = pd.read_csv(data_path)
    manager = StateDatasetManager(
        feature_cols=STATE_FEATURE_NAMES,
        seq_len=config["dataset"]["sequence_length"],
        horizon_k=config["dataset"]["rollout_horizon_k"],
        stride=config["dataset"]["stride"],
        scaler_path=config["paths"]["scaler_path"]
    )

    held_out_cat = config["dataset"].get("held_out_attack_category", "Lateral Movement")
    splits = manager.prepare_train_val_test(
        df=fused_df,
        train_ratio=config["dataset"]["train_split"],
        val_ratio=config["dataset"]["val_split"],
        test_ratio=config["dataset"]["test_split"],
        held_out_category=held_out_cat
    )

    # Load Baseline Model
    baseline_path = config["paths"]["baseline_checkpoint"]
    if Path(baseline_path).exists():
        baseline = LogisticRegressionBaseline.load(baseline_path)
    else:
        baseline = LogisticRegressionBaseline()
        x_tr, _, y_inf_tr, y_sg_tr = splits["in_distribution"]["train"]
        baseline.fit(x_tr, y_inf_tr, y_sg_tr)
        baseline.save(baseline_path)

    # Load World Model
    model_path = Path(config["paths"]["model_checkpoint"])
    model_type = config["model"].get("type", "lstm_attention")
    if model_type == "temporal_transformer":
        world_model = TemporalTransformerWorldModel(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_heads=config["model"].get("num_heads", 4),
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"].get("dim_feedforward", 256),
            num_stages=config["model"]["num_stages"]
        )
    else:
        world_model = WorldModelLSTM(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            attention_dim=config["model"].get("attention_dim", 64),
            num_stages=config["model"]["num_stages"]
        )

    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        if "model_state_dict" in ckpt:
            world_model.load_state_dict(ckpt["model_state_dict"])
        else:
            world_model.load_state_dict(ckpt)
    world_model.to(device)
    world_model.eval()

    # 1. In-Distribution Evaluation
    x_test, y_st_test, y_inf_test, y_sg_test = splits["in_distribution"]["test"]
    
    # Baseline next-step prediction
    base_infil_probs = baseline.predict_infil_proba(x_test)
    base_infil_preds = (base_infil_probs >= 0.5).astype(int)
    base_metrics_in_dist = compute_metrics(y_inf_test[:, 0], base_infil_preds)

    # World Model rollout evaluation
    with torch.no_grad():
        x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
        rollout_res = world_model.rollout(x_tensor, horizon_k=config["dataset"]["rollout_horizon_k"])
        wm_infil_probs = rollout_res["future_infil_probs"].cpu().numpy() # [N, K]
        wm_infil_preds = (wm_infil_probs >= 0.5).astype(int)

    wm_metrics_in_dist = compute_metrics(y_inf_test[:, 0], wm_infil_preds[:, 0])

    # 2. Held-Out Category Evaluation (Zero-Shot Generalization Test)
    zs_x_tr, _, zs_y_inf_tr, zs_y_sg_tr = splits["held_out"]["train"]
    zs_x_te, _, zs_y_inf_te, _ = splits["held_out"]["test"]

    # Fit baseline on zero-shot train
    if len(zs_x_te) > 0 and len(zs_x_tr) > 0:
        zs_baseline = LogisticRegressionBaseline(random_state=config["experiment"]["random_seed"])
        zs_baseline.fit(zs_x_tr, zs_y_inf_tr, zs_y_sg_tr)
        zs_base_probs = zs_baseline.predict_infil_proba(zs_x_te)
        zs_base_preds = (zs_base_probs >= 0.5).astype(int)
        base_metrics_held_out = compute_metrics(zs_y_inf_te[:, 0], zs_base_preds)

        with torch.no_grad():
            zs_x_tensor = torch.tensor(zs_x_te, dtype=torch.float32).to(device)
            zs_rollout = world_model.rollout(zs_x_tensor, horizon_k=config["dataset"]["rollout_horizon_k"])
            zs_wm_probs = zs_rollout["future_infil_probs"].cpu().numpy()
            zs_wm_preds = (zs_wm_probs >= 0.5).astype(int)
        wm_metrics_held_out = compute_metrics(zs_y_inf_te[:, 0], zs_wm_preds[:, 0])
    else:
        base_metrics_held_out = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0}
        wm_metrics_held_out = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0}

    # Multi-step Rollout metrics (t+1 through t+K)
    k_step_metrics = []
    k_horizon = config["dataset"]["rollout_horizon_k"]
    for k in range(k_horizon):
        step_metrics = compute_metrics(y_inf_test[:, k], wm_infil_preds[:, k])
        step_metrics["step"] = k + 1
        step_metrics["time_sec"] = (k + 1) * config["dataset"]["window_size_sec"]
        k_step_metrics.append(step_metrics)

    # Print Formatted Benchmark Tables
    print("\n" + "="*80)
    print("                BENCHMARK EVALUATION RESULTS (SIH26153)")
    print("="*80)
    print("\nTABLE 1: In-Distribution Performance Comparison")
    print("-" * 75)
    print(f"{'Model Architecture':<35} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8} | {'FPR':<8}")
    print("-" * 75)
    print(f"{'Logistic Regression Baseline (Static)':<35} | {base_metrics_in_dist['f1']:<8.4f} | {base_metrics_in_dist['precision']:<9.4f} | {base_metrics_in_dist['recall']:<8.4f} | {base_metrics_in_dist['fpr']:<8.4f}")
    print(f"{'World Model (LSTM + Attention)':<35} | {wm_metrics_in_dist['f1']:<8.4f} | {wm_metrics_in_dist['precision']:<9.4f} | {wm_metrics_in_dist['recall']:<8.4f} | {wm_metrics_in_dist['fpr']:<8.4f}")
    print("-" * 75)

    print(f"\nTABLE 2: Held-Out Unseen Category Evaluation (Category: '{held_out_cat}')")
    print("-" * 75)
    print(f"{'Model Architecture':<35} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8} | {'FPR':<8}")
    print("-" * 75)
    print(f"{'Logistic Regression Baseline (Static)':<35} | {base_metrics_held_out['f1']:<8.4f} | {base_metrics_held_out['precision']:<9.4f} | {base_metrics_held_out['recall']:<8.4f} | {base_metrics_held_out['fpr']:<8.4f}")
    print(f"{'World Model (LSTM + Attention)':<35} | {wm_metrics_held_out['f1']:<8.4f} | {wm_metrics_held_out['precision']:<9.4f} | {wm_metrics_held_out['recall']:<8.4f} | {wm_metrics_held_out['fpr']:<8.4f}")
    print("-" * 75)

    print("\nTABLE 3: World Model Multi-Step Forward Simulation Horizon")
    print("-" * 65)
    print(f"{'Horizon Step':<15} | {'Forecast Time':<15} | {'F1-Score':<10} | {'Recall':<10}")
    print("-" * 65)
    for row in k_step_metrics:
        step_label = f"Step t+{row['step']}"
        time_label = f"+{row['time_sec']:.1f}s"
        print(f"{step_label:<15} | {time_label:<15} | {row['f1']:<10.4f} | {row['recall']:<10.4f}")
    print("-" * 65)

    results = {
        "in_distribution": {
            "baseline": base_metrics_in_dist,
            "world_model": wm_metrics_in_dist
        },
        "held_out_category": {
            "category": held_out_cat,
            "baseline": base_metrics_held_out,
            "world_model": wm_metrics_held_out
        },
        "rollout_horizon_metrics": k_step_metrics
    }

    # Save summary JSON
    out_json = Path("model/benchmark_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Benchmark metrics saved to: {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate World Model and Baseline")
    parser.add_argument("--config", default="configs/lstm_attention.yaml", help="Path to config yaml")
    args = parser.parse_args()

    run_benchmark(config_path=args.config)
