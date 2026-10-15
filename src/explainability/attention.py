"""
Temporal Attention Explainability Module.
Extracts and formats temporal attention weights across historical sequence windows,
identifying the specific time frames that drove the model's forward attack prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class AttentionExplainer:
    """Extracts and visualizes temporal attention weights for World Model predictions."""

    @staticmethod
    def format_attention_timeline(
        attention_weights: np.ndarray,
        base_timestamp: float = 0.0,
        window_size_sec: float = 5.0,
        seq_len: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Formats 1D attention weights array into structured timeline points.
        attention_weights: array of length seq_len (e.g. 10)
        """
        if len(attention_weights) != seq_len:
            # Pad or truncate if needed
            if len(attention_weights) < seq_len:
                attention_weights = np.pad(attention_weights, (seq_len - len(attention_weights), 0))
            else:
                attention_weights = attention_weights[-seq_len:]

        # Normalize to sum to 1.0
        weights_sum = np.sum(attention_weights)
        if weights_sum > 0:
            norm_weights = attention_weights / weights_sum
        else:
            norm_weights = np.ones(seq_len) / seq_len

        items = []
        for i in range(seq_len):
            offset_steps = i - (seq_len - 1)
            t_rel = offset_steps * window_size_sec
            t_abs = base_timestamp + t_rel
            
            items.append({
                "window_index": i,
                "relative_label": f"t{offset_steps:+d} ({t_rel:.0f}s)" if offset_steps != 0 else "t (Current)",
                "relative_sec": t_rel,
                "timestamp": t_abs,
                "attention_weight": float(norm_weights[i]),
                "is_current_window": (i == seq_len - 1)
            })

        return items

    @staticmethod
    def get_most_critical_window(formatted_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the historical time window with the highest attention weight."""
        if not formatted_timeline:
            return {}
        return max(formatted_timeline, key=lambda x: x["attention_weight"])
