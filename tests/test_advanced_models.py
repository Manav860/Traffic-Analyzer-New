"""
Unit tests for Graph State Extractor, Authentication Extractor, and GNN World Model.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from src.feature_pipeline.graph_extractor import GraphFeatureExtractor, NetworkGraphState
from src.feature_pipeline.auth_extractor import AuthFeatureExtractor, AUTH_FEATURE_NAMES
from src.model.gnn_world_model import GNNWorldModel, GraphConvolution


def test_graph_feature_extractor():
    flows = pd.DataFrame({
        "src_ip": ["192.168.1.10", "192.168.1.10", "192.168.1.20", "10.0.0.5"],
        "dst_ip": ["10.0.0.5", "10.0.0.6", "10.0.0.5", "192.168.1.10"],
        "duration": [0.5, 1.2, 0.1, 2.0],
        "bytes": [1024, 2048, 512, 4096]
    })
    extractor = GraphFeatureExtractor(top_k_nodes=20)
    graph_state = extractor.extract_graph_state(flows)

    assert isinstance(graph_state, NetworkGraphState)
    assert graph_state.num_nodes > 0
    assert graph_state.num_edges > 0
    assert 0.0 <= graph_state.density <= 1.0
    assert 0.0 <= graph_state.reciprocity <= 1.0
    summary_vec = graph_state.to_summary_vector()
    assert summary_vec.shape == (8,)
    assert not np.isnan(summary_vec).any()


def test_auth_feature_extractor():
    auth_logs = pd.DataFrame({
        "timestamp": [100.0, 101.5, 102.0, 103.2, 106.0, 108.0],
        "event_id": [4624, 4625, 4625, 4672, 4624, 4625],
        "username": ["admin", "alice", "alice", "admin", "bob", "bob"],
        "src_host": ["192.168.1.5", "192.168.1.10", "192.168.1.10", "192.168.1.5", "192.168.1.15", "192.168.1.15"],
        "dst_host": ["DC01", "DC01", "FILESRV", "DC01", "WEB01", "WEB01"],
        "status": ["success", "failure", "failure", "success", "success", "failure"]
    })
    extractor = AuthFeatureExtractor(window_size_sec=5.0)
    df_auth_feat = extractor.extract_windowed_features(auth_logs)

    assert len(df_auth_feat) >= 2
    for col in AUTH_FEATURE_NAMES:
        assert col in df_auth_feat.columns
        assert not df_auth_feat[col].isna().any()


def test_gnn_world_model_forward():
    batch_size = 2
    seq_len = 5
    num_nodes = 8
    node_in_dim = 3
    latent_dim = 64
    num_stages = 6

    model = GNNWorldModel(
        node_in_dim=node_in_dim,
        gnn_hidden_dim=32,
        latent_dim=latent_dim,
        num_stages=num_stages,
        dropout=0.1
    )

    node_seq = torch.randn(batch_size, seq_len, num_nodes, node_in_dim)
    adj_seq = torch.rand(batch_size, seq_len, num_nodes, num_nodes)
    # Binarize adjacency
    adj_seq = (adj_seq > 0.6).float()

    outputs = model(node_seq, adj_seq)

    assert "predicted_next_latent" in outputs
    assert "infiltration_probability" in outputs
    assert "stage_logits" in outputs

    assert outputs["predicted_next_latent"].shape == (batch_size, latent_dim)
    assert outputs["infiltration_probability"].shape == (batch_size, 1)
    assert outputs["stage_logits"].shape == (batch_size, num_stages)

    # Infiltration prob must be valid probability
    assert (outputs["infiltration_probability"] >= 0.0).all()
    assert (outputs["infiltration_probability"] <= 1.0).all()
