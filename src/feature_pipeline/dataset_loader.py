"""
Dataset Loader & Preprocessing Pipeline for World Model Dynamics.
Constructs sliding window sequences for P(S_t+1 | S_t) dynamics learning,
multi-step forward targets, and provides held-out unseen attack category splits.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from src.feature_pipeline.fusion import STATE_FEATURE_NAMES
from src.utils.seeding import set_seed


class NetworkStateSequenceDataset(Dataset):
    """PyTorch Dataset for multi-step network state sequence rollouts."""

    def __init__(
        self,
        x_sequences: np.ndarray,
        y_future_states: np.ndarray,
        y_future_infil: np.ndarray,
        y_future_stages: np.ndarray
    ):
        self.x = torch.tensor(x_sequences, dtype=torch.float32)
        self.y_states = torch.tensor(y_future_states, dtype=torch.float32)
        self.y_infil = torch.tensor(y_future_infil, dtype=torch.float32)
        self.y_stages = torch.tensor(y_future_stages, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_states[idx], self.y_infil[idx], self.y_stages[idx]


class StateDatasetManager:
    """Manages sliding window generation, normalization, and held-out partitioning."""

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        seq_len: int = 10,
        horizon_k: int = 5,
        stride: int = 1,
        scaler_path: Optional[str] = "model/weights/scaler.pkl"
    ):
        self.feature_cols = feature_cols or STATE_FEATURE_NAMES
        self.seq_len = seq_len
        self.horizon_k = horizon_k
        self.stride = stride
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.scaler: Optional[StandardScaler] = None

    def fit_scaler(self, df: pd.DataFrame) -> StandardScaler:
        """Fit StandardScaler on feature matrix and save."""
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.feature_cols].values)
        if self.scaler_path:
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
        return self.scaler

    def load_scaler(self) -> StandardScaler:
        """Load fitted StandardScaler from disk."""
        if self.scaler_path and self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            return self.scaler
        else:
            self.scaler = StandardScaler()
            return self.scaler

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Normalize feature columns."""
        if len(df) == 0:
            return np.zeros((0, len(self.feature_cols)), dtype=np.float32)
        if self.scaler is None:
            self.load_scaler()
        return self.scaler.transform(df[self.feature_cols].values)

    def create_sequences(
        self,
        df: pd.DataFrame,
        is_normalized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Slice continuous dataframe into (X, Y_states, Y_infil, Y_stages).
        X: [N, seq_len, D]
        Y_states: [N, horizon_k, D]
        Y_infil: [N, horizon_k]
        Y_stages: [N, horizon_k]
        """
        d = len(self.feature_cols)
        required_len = self.seq_len + self.horizon_k

        if len(df) < required_len:
            return (
                np.zeros((0, self.seq_len, d), dtype=np.float32),
                np.zeros((0, self.horizon_k, d), dtype=np.float32),
                np.zeros((0, self.horizon_k), dtype=np.float32),
                np.zeros((0, self.horizon_k), dtype=np.int64)
            )

        if not is_normalized:
            feature_matrix = self.transform_dataframe(df)
        else:
            feature_matrix = df[self.feature_cols].values

        infil_series = df["is_attack"].values
        stage_series = df["stage_index"].values

        total_windows = len(df)
        required_len = self.seq_len + self.horizon_k

        x_list, y_states_list, y_infil_list, y_stages_list = [], [], [], []

        for i in range(0, total_windows - required_len + 1, self.stride):
            # History: [i ... i + seq_len)
            x_seq = feature_matrix[i : i + self.seq_len]
            # Future: [i + seq_len ... i + seq_len + horizon_k)
            y_state_seq = feature_matrix[i + self.seq_len : i + self.seq_len + self.horizon_k]
            y_infil_seq = infil_series[i + self.seq_len : i + self.seq_len + self.horizon_k]
            y_stage_seq = stage_series[i + self.seq_len : i + self.seq_len + self.horizon_k]

            x_list.append(x_seq)
            y_states_list.append(y_state_seq)
            y_infil_list.append(y_infil_seq)
            y_stages_list.append(y_stage_seq)

        if len(x_list) == 0:
            d = len(self.feature_cols)
            return (
                np.zeros((0, self.seq_len, d)),
                np.zeros((0, self.horizon_k, d)),
                np.zeros((0, self.horizon_k)),
                np.zeros((0, self.horizon_k), dtype=int)
            )

        return (
            np.array(x_list, dtype=np.float32),
            np.array(y_states_list, dtype=np.float32),
            np.array(y_infil_list, dtype=np.float32),
            np.array(y_stages_list, dtype=np.int64)
        )

    def prepare_train_val_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        held_out_category: Optional[str] = "Lateral Movement"
    ) -> Dict[str, Any]:
        """
        Split into Standard in-distribution splits (Train/Val/Test)
        AND a dedicated Held-Out Unseen Category dataset.
        """
        # 1. Fit scaler on entire dataset or train slice
        self.fit_scaler(df)

        # 2. In-distribution Chronological Split
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        df_train_raw = df.iloc[:n_train].copy()
        df_val_raw = df.iloc[n_train : n_train + n_val].copy()
        df_test_raw = df.iloc[n_train + n_val:].copy()

        # Generate in-distribution sequences
        x_tr, y_st_tr, y_inf_tr, y_sg_tr = self.create_sequences(df_train_raw)
        x_va, y_st_va, y_inf_va, y_sg_va = self.create_sequences(df_val_raw)
        x_te, y_st_te, y_inf_te, y_sg_te = self.create_sequences(df_test_raw)

        # 3. Held-Out Category Split (Zero-Shot Generalization Test)
        # Train on non-held-out data only; evaluate strictly on held-out category
        if held_out_category:
            df_zero_shot_train = df[df["attack_category"].str.lower() != held_out_category.lower()].copy()
            df_zero_shot_test = df[df["attack_category"].str.lower() == held_out_category.lower()].copy()

            zs_x_tr, zs_y_st_tr, zs_y_inf_tr, zs_y_sg_tr = self.create_sequences(df_zero_shot_train)
            zs_x_te, zs_y_st_te, zs_y_inf_te, zs_y_sg_te = self.create_sequences(df_zero_shot_test)
        else:
            zs_x_tr, zs_y_st_tr, zs_y_inf_tr, zs_y_sg_tr = x_tr, y_st_tr, y_inf_tr, y_sg_tr
            zs_x_te, zs_y_st_te, zs_y_inf_te, zs_y_sg_te = x_te, y_st_te, y_inf_te, y_sg_te

        return {
            "in_distribution": {
                "train": (x_tr, y_st_tr, y_inf_tr, y_sg_tr),
                "val": (x_va, y_st_va, y_inf_va, y_sg_va),
                "test": (x_te, y_st_te, y_inf_te, y_sg_te),
            },
            "held_out": {
                "category": held_out_category,
                "train": (zs_x_tr, zs_y_st_tr, zs_y_inf_tr, zs_y_sg_tr),
                "test": (zs_x_te, zs_y_st_te, zs_y_inf_te, zs_y_sg_te)
            }
        }


if __name__ == "__main__":
    from src.utils.synthetic_generator import generate_synthetic_flows
    from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
    from src.feature_pipeline.fusion import DualLevelFusionPipeline

    print("Testing Dataset Loader...")
    raw_df = generate_synthetic_flows(num_samples=1000)
    flow_ext = FlowFeatureExtractor(window_size_sec=5.0)
    win_df = flow_ext.extract_windowed_features(raw_df)
    fusion = DualLevelFusionPipeline()
    fused_df = fusion.fuse_dataframes(df_flow=win_df)

    manager = StateDatasetManager(seq_len=10, horizon_k=5)
    splits = manager.prepare_train_val_test(fused_df, held_out_category="Lateral Movement")
    x_tr, y_st_tr, y_inf_tr, y_sg_tr = splits["in_distribution"]["train"]
    print(f"Dataset generated: Train X shape: {x_tr.shape}, Target State shape: {y_st_tr.shape}")
