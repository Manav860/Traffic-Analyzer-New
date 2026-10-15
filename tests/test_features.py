"""
Unit Tests for Feature Extraction, Cleaning, and Fusion Pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.seeding import set_seed
from src.utils.synthetic_generator import generate_synthetic_flows
from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.packet_extractor import calculate_shannon_entropy, PacketFeatureExtractor
from src.feature_pipeline.fusion import DualLevelFusionPipeline, STATE_FEATURE_NAMES


def test_flow_extractor_nan_inf_handling():
    """Verify that NaNs and Infinities in raw flow CSV are cleaned without crashing."""
    df_dirty = pd.DataFrame({
        "Timestamp": [100.0, 101.0, 102.0],
        "Flow Duration": [0.0, np.inf, -np.inf],
        "Flow Byts/s": [np.nan, 1000.0, np.inf],
        "Tot Fwd Pkts": [1, 2, 3],
        "Tot Bwd Pkts": [0, 1, 1],
        "SYN Flag Cnt": [1, 0, 0],
        "Label": ["PortScan", "Benign", "Infiltration"]
    })

    extractor = FlowFeatureExtractor(window_size_sec=5.0)
    cleaned = extractor.clean_dataframe(df_dirty)

    assert not cleaned["Flow_Duration"].isin([np.inf, -np.inf]).any()
    assert not cleaned["Flow_Byts_s"].isna().any()
    assert not cleaned["Flow_Byts_s"].isin([np.inf, -np.inf]).any()

    windowed = extractor.extract_windowed_features(cleaned)
    assert len(windowed) > 0
    assert "syn_ratio" in windowed.columns
    assert np.isfinite(windowed["flow_rate_per_sec"]).all()


def test_shannon_entropy_calculation():
    """Verify Shannon entropy computation bounds [0.0, 8.0]."""
    assert calculate_shannon_entropy(b"") == 0.0
    assert calculate_shannon_entropy(b"AAAAAAA") == 0.0
    
    # 256 random distinct bytes should be near 8.0 bits of entropy
    distinct_bytes = bytes(range(256))
    entropy = calculate_shannon_entropy(distinct_bytes)
    assert 7.9 < entropy <= 8.0


def test_dual_level_fusion_column_integrity():
    """Verify that fused state dataframe contains all required state features with zero missing/NaN values."""
    raw_df = generate_synthetic_flows(num_samples=200, random_seed=42)
    flow_ext = FlowFeatureExtractor(window_size_sec=5.0)
    flow_df = flow_ext.extract_windowed_features(raw_df)

    fusion = DualLevelFusionPipeline()
    fused_df = fusion.fuse_dataframes(df_flow=flow_df)

    for feat in STATE_FEATURE_NAMES:
        assert feat in fused_df.columns, f"Missing feature: {feat}"
        assert not fused_df[feat].isna().any(), f"NaN in feature: {feat}"
        assert np.isfinite(fused_df[feat]).all(), f"Non-finite in feature: {feat}"

    assert "stage_index" in fused_df.columns
    assert "is_attack" in fused_df.columns


def test_packet_feature_extractor_attack_dimensions(tmp_path):
    """Verify that packet extractor calculates syn_ratio, active_connection_density, packet_rate_per_sec, and port_entropy."""
    from scapy.all import IP, TCP, wrpcap

    pkts = []
    # Create 100 SYN packets across 50 different target ports within a 2-second window
    for i in range(100):
        dport = 1000 + (i % 50)
        pkt = IP(src="192.168.1.50", dst="192.168.1.100") / TCP(sport=50000 + i, dport=dport, flags="S")
        pkt.time = 100.0 + (i * 0.01) # All within 1.0 second (< 2.0s window)
        pkts.append(pkt)

    pcap_file = tmp_path / "test_sweep.pcap"
    wrpcap(str(pcap_file), pkts)

    extractor = PacketFeatureExtractor(window_size_sec=2.0)
    df_pkt = extractor.extract_from_pcap(str(pcap_file))

    assert len(df_pkt) >= 1
    row = df_pkt.iloc[0]

    # Verify 4 attack dimensions
    assert row["syn_ratio"] == 1.0, f"Expected syn_ratio 1.0, got {row['syn_ratio']}"
    assert row["packet_rate_per_sec"] == 100.0 / 2.0, f"Expected packet rate 50.0, got {row['packet_rate_per_sec']}"
    assert row["active_connection_density"] == 50 / 100.0, f"Expected conn density 0.5, got {row['active_connection_density']}"
    assert row["port_entropy"] > 0.8, f"Expected high port entropy, got {row['port_entropy']}"

    # Verify fusion preserves these features in live capture mode
    fusion = DualLevelFusionPipeline()
    fused_df = fusion.fuse_dataframes(df_packet=df_pkt, is_live_capture=True)
    fused_row = fused_df.iloc[0]
    assert fused_row["syn_ratio"] == 1.0
    assert fused_row["packet_rate_per_sec"] == 50.0
    assert fused_row["active_connection_density"] == 0.5
    assert fused_row["port_entropy"] > 0.8


if __name__ == "__main__":
    pytest.main([__file__])
