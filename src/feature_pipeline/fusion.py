"""
Dual-Level Feature Fusion Pipeline.
Fuses flow-level (NetFlow/IPFIX) and packet-level (PCAP-derived) feature streams into
a synchronized, normalized state vector S_t representing the active network state at time window t.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

STATE_FEATURE_NAMES = [
    "flow_rate_per_sec",
    "byte_rate_per_sec",
    "packet_rate_per_sec",
    "flow_duration_mean",
    "syn_ratio",
    "ack_ratio",
    "rst_ratio",
    "psh_ratio",
    "fin_ratio",
    "iat_mean",
    "iat_std",
    "iat_max",
    "down_up_ratio",
    "pkt_size_avg",
    "port_entropy",
    "packet_count",
    "ttl_mean",
    "ttl_var",
    "tcp_win_mean",
    "tcp_win_var",
    "ip_frag_ratio",
    "rst_burst_ratio",
    "payload_entropy_mean",
    "payload_entropy_max",
    "small_pkt_ratio",
    "large_pkt_ratio",
    "port_seq_randomness",
    "active_connection_density"
]

STAGE_INDEX_MAP = {
    "benign": 0,
    "reconnaissance": 1,
    "initial access": 2,
    "lateral movement": 3,
    "command and control": 4,
    "command & control": 4,
    "c2": 4,
    "exfiltration": 5,
    "impact": 5
}


class DualLevelFusionPipeline:
    """Combines flow and packet feature streams into a unified state matrix."""

    def __init__(self, state_features: Optional[List[str]] = None):
        self.state_features = state_features or STATE_FEATURE_NAMES

    def fuse_dataframes(
        self,
        df_flow: Optional[pd.DataFrame] = None,
        df_packet: Optional[pd.DataFrame] = None,
        is_live_capture: bool = False
    ) -> pd.DataFrame:
        """
        Merge flow and packet dataframes on window_id / timestamp.
        Provides realistic imputation if one stream is absent.
        """
        if df_flow is None and df_packet is None:
            raise ValueError("At least one of df_flow or df_packet must be provided.")

        if df_flow is not None and df_packet is not None:
            # Merge on window_id
            df_fused = pd.merge(df_flow, df_packet, on="window_id", how="outer", suffixes=("_flow", "_pkt"))
            if "timestamp_start_flow" in df_fused.columns:
                df_fused["timestamp_start"] = df_fused["timestamp_start_flow"].fillna(df_fused.get("timestamp_start_pkt", 0.0))
            elif "timestamp_start_pkt" in df_fused.columns:
                df_fused["timestamp_start"] = df_fused["timestamp_start_pkt"]
            else:
                df_fused["timestamp_start"] = df_fused["window_id"] * 2.0

        elif df_flow is not None:
            df_fused = df_flow.copy()
            # Synthesize missing packet-level columns based on flow features
            if "packet_count" not in df_fused.columns:
                df_fused["packet_count"] = (df_fused["packet_rate_per_sec"] * 2.0).astype(int)
            if "ttl_mean" not in df_fused.columns:
                df_fused["ttl_mean"] = 64.0
            if "ttl_var" not in df_fused.columns:
                df_fused["ttl_var"] = 0.2
            if "tcp_win_mean" not in df_fused.columns:
                df_fused["tcp_win_mean"] = 29200.0
            if "tcp_win_var" not in df_fused.columns:
                df_fused["tcp_win_var"] = 100.0
            if "ip_frag_ratio" not in df_fused.columns:
                df_fused["ip_frag_ratio"] = 0.0
            if "rst_burst_ratio" not in df_fused.columns:
                df_fused["rst_burst_ratio"] = df_fused.get("rst_ratio", 0.0)
            if "payload_entropy_mean" not in df_fused.columns:
                df_fused["payload_entropy_mean"] = 3.5
            if "payload_entropy_max" not in df_fused.columns:
                df_fused["payload_entropy_max"] = 5.0
            if "small_pkt_ratio" not in df_fused.columns:
                df_fused["small_pkt_ratio"] = df_fused.get("syn_ratio", 0.1)
            if "large_pkt_ratio" not in df_fused.columns:
                df_fused["large_pkt_ratio"] = (df_fused["pkt_size_avg"] > 800).astype(float) * 0.5
            if "port_seq_randomness" not in df_fused.columns:
                df_fused["port_seq_randomness"] = df_fused.get("port_entropy", 0.0) * 1.5

        else: # df_packet is not None
            df_fused = df_packet.copy()

            if is_live_capture:
                # ── Live Wi-Fi capture path ────────────────────────────────────
                # Scapy PCAP captures lack CICFlowMeter flow metadata, so
                # synthesizing from packet_count produces extreme outliers that
                # fall far outside the CIC-IDS-2017 training distribution.
                # Instead we pin missing flow-level features to the CIC-IDS-2017
                # benign/mean values (what the scaler was fitted on), so the
                # normalised state vector stays close to zero and does NOT trigger
                # false-positive attack predictions.
                # Scaler fitted means (from model/weights/scaler.pkl):
                #   flow_rate_per_sec   mean=10.0
                #   packet_rate_per_sec mean=204.9
                #   byte_rate_per_sec   mean=210_000
                #   flow_duration_mean  mean=28_000_000 µs
                if "flow_rate_per_sec" not in df_fused.columns:
                    df_fused["flow_rate_per_sec"] = 10.0
                if "byte_rate_per_sec" not in df_fused.columns:
                    df_fused["byte_rate_per_sec"] = 210_000.0
                if "packet_rate_per_sec" not in df_fused.columns:
                    df_fused["packet_rate_per_sec"] = 204.9
                if "flow_duration_mean" not in df_fused.columns:
                    df_fused["flow_duration_mean"] = 28_000_000.0
                if "syn_ratio" not in df_fused.columns:
                    # HTTPS traffic is ACK-dominated; 0.05 is typical for benign TLS
                    df_fused["syn_ratio"] = 0.05
                if "ack_ratio" not in df_fused.columns:
                    df_fused["ack_ratio"] = 0.85
                if "rst_ratio" not in df_fused.columns:
                    df_fused["rst_ratio"] = 0.01
                if "psh_ratio" not in df_fused.columns:
                    df_fused["psh_ratio"] = 0.3
                if "fin_ratio" not in df_fused.columns:
                    df_fused["fin_ratio"] = 0.1
                if "iat_mean" not in df_fused.columns:
                    df_fused["iat_mean"] = 0.05
                if "iat_std" not in df_fused.columns:
                    df_fused["iat_std"] = 0.02
                if "iat_max" not in df_fused.columns:
                    df_fused["iat_max"] = 0.5
                if "down_up_ratio" not in df_fused.columns:
                    df_fused["down_up_ratio"] = 1.5
                if "pkt_size_avg" not in df_fused.columns:
                    df_fused["pkt_size_avg"] = 500.0
                if "port_entropy" in df_fused.columns:
                    df_fused["port_entropy"] = df_fused["port_entropy"].clip(0.0, 1.0)
                else:
                    df_fused["port_entropy"] = 0.25

                # Force label to benign so MITRE resolver doesn't assign an
                # attack stage from an empty or misinterpreted "label" column.
                df_fused["label"] = "Live_Capture"

            else:
                # ── Offline / CSV-based path (original behaviour) ──────────────
                if "flow_rate_per_sec" not in df_fused.columns:
                    df_fused["flow_rate_per_sec"] = df_fused["packet_count"] / 10.0
                if "byte_rate_per_sec" not in df_fused.columns:
                    df_fused["byte_rate_per_sec"] = df_fused["packet_count"] * 300.0
                if "packet_rate_per_sec" not in df_fused.columns:
                    df_fused["packet_rate_per_sec"] = df_fused["packet_count"] / 2.0
                if "flow_duration_mean" not in df_fused.columns:
                    df_fused["flow_duration_mean"] = 1.0
                if "syn_ratio" not in df_fused.columns:
                    df_fused["syn_ratio"] = df_fused["small_pkt_ratio"]
                if "ack_ratio" not in df_fused.columns:
                    df_fused["ack_ratio"] = 0.8
                if "rst_ratio" not in df_fused.columns:
                    df_fused["rst_ratio"] = df_fused["rst_burst_ratio"]
                if "psh_ratio" not in df_fused.columns:
                    df_fused["psh_ratio"] = 0.2
                if "fin_ratio" not in df_fused.columns:
                    df_fused["fin_ratio"] = 0.1
                if "iat_mean" not in df_fused.columns:
                    df_fused["iat_mean"] = 0.05
                if "iat_std" not in df_fused.columns:
                    df_fused["iat_std"] = 0.01
                if "iat_max" not in df_fused.columns:
                    df_fused["iat_max"] = 0.2
                if "down_up_ratio" not in df_fused.columns:
                    df_fused["down_up_ratio"] = 1.2
                if "pkt_size_avg" not in df_fused.columns:
                    df_fused["pkt_size_avg"] = 450.0
                if "port_entropy" not in df_fused.columns:
                    df_fused["port_entropy"] = df_fused["port_seq_randomness"]

        # Calculate interaction density
        if "active_connection_density" not in df_fused.columns:
            flow_r = df_fused.get("flow_rate_per_sec", 1.0)
            pkt_r = df_fused.get("packet_rate_per_sec", 1.0)
            df_fused["active_connection_density"] = flow_r / (pkt_r + 1.0)

        # Standardize metadata labels
        if "label" not in df_fused.columns:
            # Check if attack can be inferred from signatures
            is_scan = (df_fused.get("port_entropy", 0) > 0.6) | (df_fused.get("syn_ratio", 0) > 0.8)
            df_fused["label"] = np.where(is_scan, "PortScan", "Benign")
        
        # Standardize metadata labels & map to 5-stage MITRE kill chain
        def resolve_attack_category(row_label: str) -> str:
            l = str(row_label).lower()
            if "scan" in l:
                return "Reconnaissance"
            if any(k in l for k in ["patator", "brute", "sql", "xss", "web attack", "heartbleed"]):
                return "Initial Access"
            if "infil" in l:
                return "Lateral Movement"
            if any(k in l for k in ["bot", "c2", "ares", "heartbeat"]):
                return "Command and Control"
            if any(k in l for k in ["dos", "ddos", "hulk", "slow", "golden", "exfil"]):
                return "Impact"
            return "Benign"

        df_fused["attack_category"] = df_fused["label"].apply(resolve_attack_category)
        df_fused["is_attack"] = (df_fused["attack_category"].str.lower() != "benign").astype(int)
        df_fused["stage_index"] = df_fused["attack_category"].str.lower().map(
            lambda cat: STAGE_INDEX_MAP.get(str(cat), 0)
        ).fillna(0).astype(int)

        # Ensure all 28 feature columns exist and are finite numbers
        for feat in self.state_features:
            if feat not in df_fused.columns:
                df_fused[feat] = 0.0
            df_fused[feat] = df_fused[feat].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        df_fused = df_fused.sort_values("timestamp_start").reset_index(drop=True)
        return df_fused

    def fuse_files(
        self,
        flow_csv: Optional[str] = None,
        packet_csv: Optional[str] = None,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """Read CSV files, fuse streams, and save."""
        df_flow = pd.read_csv(flow_csv) if flow_csv and Path(flow_csv).exists() else None
        df_packet = pd.read_csv(packet_csv) if packet_csv and Path(packet_csv).exists() else None

        df_fused = self.fuse_dataframes(df_flow=df_flow, df_packet=df_packet)
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df_fused.to_csv(output_csv, index=False)
        return df_fused


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fuse flow and packet feature streams")
    parser.add_argument("--flow", default=None, help="Flow features CSV")
    parser.add_argument("--packet", default=None, help="Packet features CSV")
    parser.add_argument("--output", default="data/processed/fused_state.csv", help="Output fused CSV")
    args = parser.parse_args()

    fusion = DualLevelFusionPipeline()
    result = fusion.fuse_files(args.flow, args.packet, args.output)
    print(f"Fused state matrix generated: {result.shape} -> {args.output}")
