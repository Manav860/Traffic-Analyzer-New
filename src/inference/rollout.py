"""
K-Step Forward Rollout Engine.
Performs recursive forward simulation of the World Model to forecast multi-step attack progression,
infiltration probability trajectories, and MITRE ATT&CK kill-chain transitions.
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import torch

from src.feature_pipeline.fusion import STATE_FEATURE_NAMES
from src.model.world_model import WorldModelLSTM
from src.model.transformer_world_model import TemporalTransformerWorldModel
from src.mitre_mapping.stix_mapper import MitreMapper


class RolloutEngine:
    """
    K-step forward simulation engine.
    Given historical sequence (t-L+1 ... t), simulates network state dynamics for (t+1 ... t+K).
    """

    def __init__(
        self,
        model_checkpoint: str = "model/weights/world_model_final.pt",
        scaler_path: str = "model/weights/scaler.pkl",
        config_path: str = "configs/lstm_attention.yaml",
        device: str = "auto"
    ):
        self.model_checkpoint = Path(model_checkpoint)
        self.scaler_path = Path(scaler_path)
        self.config_path = Path(config_path)
        self.mapper = MitreMapper()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[torch.nn.Module] = None
        self.scaler: Optional[Any] = None
        self.config: Dict[str, Any] = {}

        self._load_components()

    def _load_components(self) -> None:
        """Load configuration, fitted scaler, and model checkpoint."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        # Load Scaler
        if self.scaler_path.exists():
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # Instantiate & Load Model
        input_dim = self.config.get("model", {}).get("input_dim", 28)
        hidden_dim = self.config.get("model", {}).get("hidden_dim", 128)
        num_layers = self.config.get("model", {}).get("num_layers", 2)
        num_stages = self.config.get("model", {}).get("num_stages", 6)
        model_type = self.config.get("model", {}).get("type", "lstm_attention")

        if model_type == "temporal_transformer":
            self.model = TemporalTransformerWorldModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=self.config.get("model", {}).get("num_heads", 4),
                num_layers=num_layers,
                dim_feedforward=self.config.get("model", {}).get("dim_feedforward", 256),
                num_stages=num_stages
            )
        else:
            self.model = WorldModelLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                attention_dim=self.config.get("model", {}).get("attention_dim", 64),
                num_stages=num_stages
            )

        if self.model_checkpoint.exists():
            checkpoint = torch.load(self.model_checkpoint, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def simulate_from_raw_sequence(
        self,
        raw_df_history: pd.DataFrame,
        horizon_k: int = 5,
        window_size_sec: float = 2.0
    ) -> Dict[str, Any]:
        """
        Takes raw/fused dataframe of historical windows, normalizes, and rolls forward K steps.
        """
        if self.scaler is not None:
            norm_matrix = self.scaler.transform(raw_df_history[STATE_FEATURE_NAMES].values)
        else:
            norm_matrix = raw_df_history[STATE_FEATURE_NAMES].values

        seq_tensor = torch.tensor(norm_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        last_timestamp = float(raw_df_history["timestamp_start"].iloc[-1]) if "timestamp_start" in raw_df_history else 0.0

        return self.simulate(seq_tensor, horizon_k=horizon_k, base_timestamp=last_timestamp, window_size_sec=window_size_sec)

    def simulate(
        self,
        history_tensor: torch.Tensor,
        horizon_k: int = 5,
        base_timestamp: float = 0.0,
        window_size_sec: float = 2.0
    ) -> Dict[str, Any]:
        """
        Execute forward simulation.
        history_tensor: [1, Seq_Len, D]
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        self.model.eval()
        with torch.no_grad():
            rollout_output = self.model.rollout(history_tensor.to(self.device), horizon_k=horizon_k)

        # Unpack predictions
        fut_states_norm = rollout_output["future_states"].squeeze(0).cpu().numpy()     # [K, D]
        fut_infil_probs = rollout_output["future_infil_probs"].squeeze(0).cpu().numpy() # [K]
        fut_stage_probs = rollout_output["future_stage_probs"].squeeze(0).cpu().numpy() # [K, Num_Stages]
        fut_stage_preds = rollout_output["future_stage_preds"].squeeze(0).cpu().numpy() # [K]
        step_attns = [a.squeeze(0).cpu().numpy() for a in rollout_output["step_attentions"]]

        # Unnormalize predicted states for human inspection
        if self.scaler is not None:
            fut_states_unnorm = self.scaler.inverse_transform(fut_states_norm)
        else:
            fut_states_unnorm = fut_states_norm

        # Format timeline and stage information
        timestamps = [base_timestamp + (i + 1) * window_size_sec for i in range(horizon_k)]
        stage_names = [self.mapper.get_stage_name_from_index(int(idx)) for idx in fut_stage_preds]

        # Trajectory risk acceleration
        risk_deltas = np.diff(fut_infil_probs, prepend=fut_infil_probs[0])
        is_escalating = bool(np.mean(risk_deltas) > 0.05 or fut_infil_probs[-1] > 0.70)

        # Build trajectory items
        timeline_items = []
        for i in range(horizon_k):
            timeline_items.append({
                "step": i + 1,
                "relative_time_sec": (i + 1) * window_size_sec,
                "timestamp": timestamps[i],
                "infiltration_probability": float(fut_infil_probs[i]),
                "predicted_stage": stage_names[i],
                "stage_index": int(fut_stage_preds[i]),
                "stage_distribution": {
                    self.mapper.get_stage_name_from_index(s_idx): float(fut_stage_probs[i, s_idx])
                    for s_idx in range(len(fut_stage_probs[i]))
                },
                "key_predicted_features": {
                    "syn_ratio": float(fut_states_unnorm[i, STATE_FEATURE_NAMES.index("syn_ratio")]),
                    "flow_rate": float(fut_states_unnorm[i, STATE_FEATURE_NAMES.index("flow_rate_per_sec")]),
                    "port_entropy": float(fut_states_unnorm[i, STATE_FEATURE_NAMES.index("port_entropy")]),
                    "payload_entropy": float(fut_states_unnorm[i, STATE_FEATURE_NAMES.index("payload_entropy_mean")])
                }
            })

        # Select forecasted attack stage:
        # If an attack stage is forecasted along the trajectory, pick the highest attack stage
        # rather than letting a decaying tail (step 5 reverting to benign baseline) report "Benign"
        peak_risk = float(np.max(fut_infil_probs))
        non_benign_stages = [
            (int(fut_stage_preds[i]), stage_names[i], float(fut_infil_probs[i]))
            for i in range(horizon_k)
            if int(fut_stage_preds[i]) > 0
        ]

        if non_benign_stages:
            highest_stage_idx, highest_stage_name, _ = max(non_benign_stages, key=lambda x: x[0])
            terminal_stage = highest_stage_name
            terminal_stage_idx = highest_stage_idx
        elif peak_risk >= 0.35:
            # Fallback: check stage distributions across steps for the most probable attack stage
            best_attack_stage = "Reconnaissance"
            best_attack_idx = 1
            max_attack_prob = 0.0
            for i in range(horizon_k):
                for s_idx in range(1, len(fut_stage_probs[i])):
                    if fut_stage_probs[i, s_idx] > max_attack_prob:
                        max_attack_prob = float(fut_stage_probs[i, s_idx])
                        best_attack_idx = s_idx
                        best_attack_stage = self.mapper.get_stage_name_from_index(s_idx)
            if max_attack_prob > 0.10:
                terminal_stage = best_attack_stage
                terminal_stage_idx = best_attack_idx
            else:
                terminal_stage = "Reconnaissance" if peak_risk >= 0.50 else stage_names[-1]
                terminal_stage_idx = 1 if peak_risk >= 0.50 else int(fut_stage_preds[-1])
        else:
            terminal_stage = stage_names[-1]
            terminal_stage_idx = int(fut_stage_preds[-1])

        return {
            "horizon_k": horizon_k,
            "base_timestamp": base_timestamp,
            "is_escalating": is_escalating,
            "peak_risk": peak_risk,
            "terminal_stage": terminal_stage,
            "terminal_stage_idx": terminal_stage_idx,
            "timeline": timeline_items,
            "predicted_states_raw": fut_states_unnorm,
            "attention_weights_initial": step_attns[0] if step_attns else np.array([])
        }
