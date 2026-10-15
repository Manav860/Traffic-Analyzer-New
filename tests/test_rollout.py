"""
Unit Tests for Forward Simulation Rollout and MITRE Kill-Chain Forecasting.
"""

import pytest
import numpy as np
import torch

from src.utils.seeding import set_seed
from src.model.world_model import WorldModelLSTM
from src.model.transformer_world_model import TemporalTransformerWorldModel
from src.mitre_mapping.stix_mapper import MitreMapper


def test_lstm_world_model_rollout_dimensions():
    """Verify LSTM World Model rollout output shapes and probability bounds."""
    set_seed(42)
    batch_size = 3
    seq_len = 10
    input_dim = 28
    horizon_k = 5
    num_stages = 6

    model = WorldModelLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, num_stages=num_stages)
    dummy_x = torch.randn(batch_size, seq_len, input_dim)

    rollout_res = model.rollout(dummy_x, horizon_k=horizon_k)

    assert rollout_res["future_states"].shape == (batch_size, horizon_k, input_dim)
    assert rollout_res["future_infil_probs"].shape == (batch_size, horizon_k)
    assert rollout_res["future_stage_probs"].shape == (batch_size, horizon_k, num_stages)
    assert rollout_res["future_stage_preds"].shape == (batch_size, horizon_k)

    probs = rollout_res["future_infil_probs"].numpy()
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_transformer_world_model_rollout_dimensions():
    """Verify Transformer World Model rollout output shapes."""
    set_seed(42)
    batch_size = 2
    seq_len = 8
    input_dim = 28
    horizon_k = 4

    model = TemporalTransformerWorldModel(input_dim=input_dim, hidden_dim=64, num_heads=2, num_layers=1)
    dummy_x = torch.randn(batch_size, seq_len, input_dim)

    rollout_res = model.rollout(dummy_x, horizon_k=horizon_k)

    assert rollout_res["future_states"].shape == (batch_size, horizon_k, input_dim)
    assert rollout_res["future_infil_probs"].shape == (batch_size, horizon_k)


def test_mitre_mapper_stage_resolution():
    """Verify STIX mapper translates attack labels to expected tactics and indices."""
    mapper = MitreMapper()

    recon_info = mapper.get_tactic_for_label("PortScan")
    assert recon_info["tactic_name"] == "Reconnaissance"
    assert recon_info["kill_chain_stage"] == 1

    access_info = mapper.get_tactic_for_label("SSH-Bruteforce")
    assert access_info["tactic_name"] == "Initial Access"
    assert access_info["kill_chain_stage"] == 2

    lateral_info = mapper.get_tactic_for_label("Infiltration")
    assert lateral_info["tactic_name"] == "Lateral Movement"
    assert lateral_info["kill_chain_stage"] == 3

    c2_info = mapper.get_tactic_for_label("Bot")
    assert c2_info["tactic_name"] == "Command and Control"
    assert c2_info["kill_chain_stage"] == 4


if __name__ == "__main__":
    pytest.main([__file__])
