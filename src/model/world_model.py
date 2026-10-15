"""
World Model Architecture: Network State Dynamics & Attack Progression Forecaster.
Learns P(S_t+1 | S_t) transition dynamics using LSTM with Additive Temporal Attention,
enabling multi-step forward simulation (rollout) and MITRE ATT&CK stage forecasting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


class TemporalAdditiveAttention(nn.Module):
    """
    Additive (Bahdanau-style) Temporal Attention mechanism.
    Computes importance weights across historical time windows.
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 64):
        super().__init__()
        self.w_h = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.w_q = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_seq: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_seq: [Batch, Seq_Len, Hidden_Dim]
        query: [Batch, Hidden_Dim] (typically the last hidden state)
        Returns:
            context: [Batch, Hidden_Dim]
            weights: [Batch, Seq_Len]
        """
        # [Batch, Seq_Len, Attn_Dim]
        proj_h = self.w_h(hidden_seq)
        # [Batch, 1, Attn_Dim]
        proj_q = self.w_q(query).unsqueeze(1)

        # [Batch, Seq_Len, 1]
        energy = self.v(torch.tanh(proj_h + proj_q))
        weights = F.softmax(energy.squeeze(-1), dim=-1) # [Batch, Seq_Len]

        # Context vector = weighted sum over hidden sequence
        context = torch.bmm(weights.unsqueeze(1), hidden_seq).squeeze(1) # [Batch, Hidden_Dim]
        return context, weights


class WorldModelLSTM(nn.Module):
    """
    Full World Model dynamics learner:
    1. Historical Sequence Processing (LSTM)
    2. Temporal Attention Layer
    3. Dynamics Head: P(S_t+1 | S_t) future state reconstruction
    4. Infiltration Risk Head: Probability of active/pending breach
    5. MITRE ATT&CK Stage Head: Tactical kill-chain phase classification
    """

    def __init__(
        self,
        input_dim: int = 28,
        hidden_dim: int = 128,
        num_layers: int = 2,
        attention_dim: int = 64,
        dropout: float = 0.2,
        num_stages: int = 6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stages = num_stages

        # Sequence Encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention Mechanism
        self.attention = TemporalAdditiveAttention(hidden_dim, attention_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Head 1: State Transition Dynamics P(S_t+1 | S_t)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Head 2: Infiltration Risk Probability Head
        self.infil_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Head 3: MITRE ATT&CK Kill-Chain Stage Head
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_stages)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Single-step forward pass.
        x: [Batch, Seq_Len, Input_Dim]
        Returns dict containing:
            - next_state: [Batch, Input_Dim] (Predicted S_t+1)
            - infil_prob: [Batch, 1] (Probability of infiltration at t+1)
            - stage_logits: [Batch, Num_Stages] (Logits over MITRE stages at t+1)
            - attention_weights: [Batch, Seq_Len]
            - latent_rep: [Batch, Hidden_Dim]
        """
        # lstm_out: [Batch, Seq_Len, Hidden_Dim]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Last layer hidden state as query
        query = h_n[-1] # [Batch, Hidden_Dim]

        # Temporal attention
        context, attn_weights = self.attention(lstm_out, query)
        
        # Residual merge + normalization
        fused = self.layer_norm(context + lstm_out[:, -1, :])
        fused = self.dropout(fused)

        # Compute predictions
        next_state = self.dynamics_head(fused)
        infil_prob = self.infil_head(fused)
        stage_logits = self.stage_head(fused)

        return {
            "next_state": next_state,
            "infil_prob": infil_prob,
            "stage_logits": stage_logits,
            "attention_weights": attn_weights,
            "latent_rep": fused
        }

    def rollout(self, initial_sequence: torch.Tensor, horizon_k: int = 5) -> Dict[str, Any]:
        """
        Recursive K-step forward simulation (Auto-regressive rollout).
        Feeds model's own predicted state back in as input for K consecutive steps.
        
        initial_sequence: [Batch, Seq_Len, Input_Dim]
        horizon_k: int
        
        Returns:
            future_states: [Batch, K, Input_Dim]
            future_infil_probs: [Batch, K]
            future_stage_probs: [Batch, K, Num_Stages]
            future_stage_preds: [Batch, K]
            step_attentions: list of [Batch, Seq_Len] for each step
        """
        self.eval()
        batch_size = initial_sequence.size(0)
        curr_seq = initial_sequence.clone()

        future_states = []
        future_infil_probs = []
        future_stage_probs = []
        future_stage_preds = []
        step_attentions = []

        with torch.no_grad():
            for step in range(horizon_k):
                out = self.forward(curr_seq)
                pred_state = out["next_state"]        # [Batch, Input_Dim]
                pred_infil = out["infil_prob"]        # [Batch, 1]
                stage_log = out["stage_logits"]       # [Batch, Num_Stages]
                attn_w = out["attention_weights"]     # [Batch, Seq_Len]

                stage_probs = F.softmax(stage_log, dim=-1)
                stage_pred = torch.argmax(stage_probs, dim=-1)

                future_states.append(pred_state.unsqueeze(1))
                future_infil_probs.append(pred_infil.squeeze(-1).unsqueeze(1))
                future_stage_probs.append(stage_probs.unsqueeze(1))
                future_stage_preds.append(stage_pred.unsqueeze(1))
                step_attentions.append(attn_w)

                # Slide the sequence window: append predicted state, drop oldest state
                # curr_seq: [Batch, Seq_Len, Input_Dim]
                curr_seq = torch.cat([curr_seq[:, 1:, :], pred_state.unsqueeze(1)], dim=1)

        return {
            "future_states": torch.cat(future_states, dim=1),
            "future_infil_probs": torch.cat(future_infil_probs, dim=1),
            "future_stage_probs": torch.cat(future_stage_probs, dim=1),
            "future_stage_preds": torch.cat(future_stage_preds, dim=1),
            "step_attentions": step_attentions
        }


if __name__ == "__main__":
    model = WorldModelLSTM(input_dim=28, hidden_dim=64, num_layers=2)
    dummy_x = torch.randn(4, 10, 28)
    out = model(dummy_x)
    print("Single step output shapes:")
    print("Next State:", out["next_state"].shape)
    print("Infil Prob:", out["infil_prob"].shape)
    print("Stage Logits:", out["stage_logits"].shape)
    print("Attention Weights:", out["attention_weights"].shape)

    rollout_res = model.rollout(dummy_x, horizon_k=5)
    print("\nRollout results:")
    print("Future states:", rollout_res["future_states"].shape)
    print("Future infil probs:", rollout_res["future_infil_probs"].shape)
