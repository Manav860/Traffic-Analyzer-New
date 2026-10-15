"""
SHAP Explainability Module.
Computes feature attributions using SHAP (SHapley Additive exPlanations)
to explain the key traffic features driving infiltration predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.feature_pipeline.fusion import STATE_FEATURE_NAMES


class FeatureExplainer:
    """SHAP and feature attribution explainer for network state predictions."""

    def __init__(self, baseline_model: Optional[Any] = None, feature_names: Optional[List[str]] = None):
        self.baseline_model = baseline_model
        self.feature_names = feature_names or STATE_FEATURE_NAMES
        self.explainer: Optional[Any] = None

    def initialize_linear_explainer(self, background_data: np.ndarray) -> None:
        """Initialize LinearExplainer for fast real-time SHAP computations."""
        if not SHAP_AVAILABLE or self.baseline_model is None:
            return

        try:
            if hasattr(self.baseline_model, "infil_model"):
                self.explainer = shap.LinearExplainer(
                    self.baseline_model.infil_model,
                    background_data[:100] if len(background_data) > 100 else background_data
                )
        except Exception as e:
            print(f"[!] SHAP LinearExplainer init note: {e}")
            self.explainer = None

    def explain_instance(
        self,
        instance_features: np.ndarray,
        top_k: int = 8
    ) -> Dict[str, Any]:
        """
        Explain a single feature vector instance S_t.
        Returns top positive and negative contributing features.
        instance_features: 1D array of length D
        """
        if instance_features.ndim > 1:
            instance_1d = instance_features.flatten()
        else:
            instance_1d = instance_features

        # If SHAP linear explainer is available, compute exact Shapley values
        shap_values = None
        if self.explainer is not None:
            try:
                sv = self.explainer.shap_values(instance_1d.reshape(1, -1))
                if isinstance(sv, list):
                    shap_values = sv[1][0] # class 1
                elif isinstance(sv, np.ndarray):
                    shap_values = sv[0]
            except Exception:
                shap_values = None

        # Fallback to model weights * normalized value attribution if SHAP call fails
        if shap_values is None:
            if self.baseline_model and hasattr(self.baseline_model, "infil_model") and hasattr(self.baseline_model.infil_model, "coef_"):
                coefs = self.baseline_model.infil_model.coef_[0]
                shap_values = coefs * instance_1d[:len(coefs)]
            else:
                # Direct magnitude attribution
                shap_values = instance_1d * 0.1

        # Format feature attributions
        attributions = []
        for name, val, sv in zip(self.feature_names, instance_1d, shap_values):
            attributions.append({
                "feature": name,
                "value": float(val),
                "importance": float(sv),
                "abs_importance": float(abs(sv)),
                "direction": "Risk Increase (+)" if sv > 0 else "Risk Decrease (-)"
            })

        # Sort by absolute importance
        attributions_sorted = sorted(attributions, key=lambda x: x["abs_importance"], reverse=True)
        top_drivers = attributions_sorted[:top_k]

        return {
            "top_drivers": top_drivers,
            "all_attributions": attributions_sorted,
            "shap_values": [float(x) for x in shap_values]
        }
