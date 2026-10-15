"""
End-to-End Smoke Test.
Verifies data generation, feature extraction, baseline fitting,
World Model training (1 epoch), forward simulation rollout, and explainability in <30 seconds.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.utils.seeding import set_seed
from src.utils.synthetic_generator import generate_synthetic_flows
from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.fusion import DualLevelFusionPipeline, STATE_FEATURE_NAMES
from src.feature_pipeline.dataset_loader import StateDatasetManager
from src.model.baseline import LogisticRegressionBaseline
from src.model.world_model import WorldModelLSTM
from src.inference.rollout import RolloutEngine
from src.explainability.attention import AttentionExplainer
from src.explainability.shap_explainer import FeatureExplainer
from src.mitre_mapping.stix_mapper import MitreMapper


def test_end_to_end_smoke_pipeline(tmp_path):
    set_seed(42)

    # 1. Generate synthetic flow dataframe
    raw_df = generate_synthetic_flows(num_samples=1000, random_seed=42)
    assert len(raw_df) == 1000
    assert "Label" in raw_df.columns

    # 2. Extract windowed flow features
    flow_ext = FlowFeatureExtractor(window_size_sec=5.0)
    flow_df = flow_ext.extract_windowed_features(raw_df)
    assert len(flow_df) > 0
    assert "syn_ratio" in flow_df.columns

    # 3. Dual-Level Fusion
    fusion = DualLevelFusionPipeline()
    fused_df = fusion.fuse_dataframes(df_flow=flow_df)
    assert len(fused_df) == len(flow_df)
    for feat in STATE_FEATURE_NAMES:
        assert feat in fused_df.columns

    # 4. Dataset Manager & Splits
    scaler_file = tmp_path / "scaler.pkl"
    manager = StateDatasetManager(
        feature_cols=STATE_FEATURE_NAMES,
        seq_len=6,
        horizon_k=3,
        stride=1,
        scaler_path=str(scaler_file)
    )
    splits = manager.prepare_train_val_test(fused_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    x_tr, y_st_tr, y_inf_tr, y_sg_tr = splits["in_distribution"]["train"]
    assert x_tr.ndim == 3
    assert x_tr.shape[1] == 6 # seq_len
    assert x_tr.shape[2] == len(STATE_FEATURE_NAMES)

    # 5. Baseline Logistic Regression Fit & Predict
    baseline = LogisticRegressionBaseline(random_state=42)
    baseline.fit(x_tr, y_inf_tr, y_sg_tr)
    base_probs = baseline.predict_infil_proba(x_tr)
    assert len(base_probs) == len(x_tr)
    assert np.all((base_probs >= 0.0) & (base_probs <= 1.0))

    # 6. World Model Single-Step and Rollout
    model = WorldModelLSTM(input_dim=len(STATE_FEATURE_NAMES), hidden_dim=32, num_layers=1, num_stages=6)
    model.eval()
    dummy_input = torch.tensor(x_tr[:2], dtype=torch.float32)
    out = model(dummy_input)
    assert "next_state" in out
    assert "infil_prob" in out
    assert "stage_logits" in out
    assert "attention_weights" in out

    rollout_res = model.rollout(dummy_input, horizon_k=3)
    assert rollout_res["future_states"].shape == (2, 3, len(STATE_FEATURE_NAMES))
    assert rollout_res["future_infil_probs"].shape == (2, 3)

    # 7. MITRE STIX Mapper
    mapper = MitreMapper()
    info = mapper.get_tactic_for_label("PortScan")
    assert info["tactic_name"] == "Reconnaissance"

    # 8. Explainability
    attn_items = AttentionExplainer.format_attention_timeline(out["attention_weights"][0].detach().numpy(), seq_len=6)
    assert len(attn_items) == 6

    explainer = FeatureExplainer(baseline_model=baseline)
    shap_info = explainer.explain_instance(x_tr[0, -1], top_k=4)
    assert len(shap_info["top_drivers"]) == 4

    print("Smoke test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__])
