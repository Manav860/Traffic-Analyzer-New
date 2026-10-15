"""
Temporal Transformer World Model.
Swappable sequence architecture leveraging Multi-Head Self-Attention for long-range dependency modeling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence timestamps."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Seq_Len, D_Model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TemporalTransformerWorldModel(nn.Module):
    """
    Transformer-Encoder based network dynamics learner.
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_stages: int = 6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages

        # Linear projection to model dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dynamics Head
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Infiltration Probability Head
        self.infil_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # MITRE Stage Head
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_stages)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [Batch, Seq_Len, Input_Dim]
        proj = self.input_proj(x)
        encoded = self.pos_encoder(proj)
        out = self.transformer_encoder(encoded) # [Batch, Seq_Len, Hidden_Dim]

        # Use pooled / last state representation
        last_rep = self.layer_norm(out[:, -1, :])

        next_state = self.dynamics_head(last_rep)
        infil_prob = self.infil_head(last_rep)
        stage_logits = self.stage_head(last_rep)

        # Uniform or dummy attention weights for interface consistency
        dummy_weights = torch.ones(x.size(0), x.size(1), device=x.device) / x.size(1)

        return {
            "next_state": next_state,
            "infil_prob": infil_prob,
            "stage_logits": stage_logits,
            "attention_weights": dummy_weights,
            "latent_rep": last_rep
        }

    def rollout(self, initial_sequence: torch.Tensor, horizon_k: int = 5) -> Dict[str, Any]:
        """Recursive forward rollout matching WorldModelLSTM interface."""
        self.eval()
        curr_seq = initial_sequence.clone()

        future_states, future_infil_probs, future_stage_probs, future_stage_preds = [], [], [], []
        step_attentions = []

        with torch.no_grad():
            for _ in range(horizon_k):
                out = self.forward(curr_seq)
                pred_state = out["next_state"]
                pred_infil = out["infil_prob"]
                stage_log = out["stage_logits"]
                attn_w = out["attention_weights"]

                stage_probs = F.softmax(stage_log, dim=-1)
                stage_pred = torch.argmax(stage_probs, dim=-1)

                future_states.append(pred_state.unsqueeze(1))
                future_infil_probs.append(pred_infil.squeeze(-1).unsqueeze(1))
                future_stage_probs.append(stage_probs.unsqueeze(1))
                future_stage_preds.append(stage_pred.unsqueeze(1))
                step_attentions.append(attn_w)

                curr_seq = torch.cat([curr_seq[:, 1:, :], pred_state.unsqueeze(1)], dim=1)

        return {
            "future_states": torch.cat(future_states, dim=1),
            "future_infil_probs": torch.cat(future_infil_probs, dim=1),
            "future_stage_probs": torch.cat(future_stage_probs, dim=1),
            "future_stage_preds": torch.cat(future_stage_preds, dim=1),
            "step_attentions": step_attentions
        }
