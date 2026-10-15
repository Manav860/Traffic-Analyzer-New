"""
Flow-Level Feature Extractor.
Parses NetFlow / IPFIX / CIC-IDS-2018 CSV flow records into standardized time-windowed aggregated flow vectors.
Explicitly detects and cleans NaNs, Infinities, and missing fields.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class FlowFeatureExtractor:
    """Extracts and cleans aggregate flow statistics across fixed time windows."""

    # Standard feature columns expected or mapped
    RAW_COLUMN_MAPPINGS = {
        # CIC-IDS-2018 variants -> Standard Name
        "timestamp": "Timestamp",
        "dst port": "Dst_Port",
        "src port": "Src_Port",
        "destination port": "Dst_Port",
        "flow duration": "Flow_Duration",
        "tot fwd pkts": "Tot_Fwd_Pkts",
        "tot bwd pkts": "Tot_Bwd_Pkts",
        "total fwd packets": "Tot_Fwd_Pkts",
        "total backward packets": "Tot_Bwd_Pkts",
        "totlen fwd pkts": "TotLen_Fwd_Pkts",
        "totlen bwd pkts": "TotLen_Bwd_Pkts",
        "total length of fwd packets": "TotLen_Fwd_Pkts",
        "total length of bwd packets": "TotLen_Bwd_Pkts",
        "flow byts/s": "Flow_Byts_s",
        "flow pkts/s": "Flow_Pkts_s",
        "flow bytes/s": "Flow_Byts_s",
        "flow packets/s": "Flow_Pkts_s",
        "flow iat mean": "Flow_IAT_Mean",
        "flow iat std": "Flow_IAT_Std",
        "flow iat max": "Flow_IAT_Max",
        "flow iat min": "Flow_IAT_Min",
        "fwd iat tot": "Fwd_IAT_Tot",
        "bwd iat tot": "Bwd_IAT_Tot",
        "fwd iat total": "Fwd_IAT_Tot",
        "bwd iat total": "Bwd_IAT_Tot",
        "syn flag cnt": "SYN_Flag_Cnt",
        "ack flag cnt": "ACK_Flag_Cnt",
        "fin flag cnt": "FIN_Flag_Cnt",
        "rst flag cnt": "RST_Flag_Cnt",
        "psh flag cnt": "PSH_Flag_Cnt",
        "urg flag cnt": "URG_Flag_Cnt",
        "syn flag count": "SYN_Flag_Cnt",
        "ack flag count": "ACK_Flag_Cnt",
        "fin flag count": "FIN_Flag_Cnt",
        "rst flag count": "RST_Flag_Cnt",
        "psh flag count": "PSH_Flag_Cnt",
        "urg flag count": "URG_Flag_Cnt",
        "down/up ratio": "Down_Up_Ratio",
        "pkt size avg": "Pkt_Size_Avg",
        "average packet size": "Pkt_Size_Avg",
        "fwd seg size avg": "Fwd_Seg_Size_Avg",
        "bwd seg size avg": "Bwd_Seg_Size_Avg",
        "avg fwd segment size": "Fwd_Seg_Size_Avg",
        "avg bwd segment size": "Bwd_Seg_Size_Avg",
        "init win bytes forward": "Init_Win_Bytes_Fwd",
        "init win bytes backward": "Init_Win_Bytes_Bwd",
        "init_win_bytes_forward": "Init_Win_Bytes_Fwd",
        "init_win_bytes_backward": "Init_Win_Bytes_Bwd",
        "label": "Label",
        "protocol": "Protocol"
    }

    def __init__(self, window_size_sec: float = 5.0):
        self.window_size_sec = window_size_sec

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names, handle NaN / Inf values, and parse timestamps.
        """
        # Normalize column names (strip whitespace, lowercase matching)
        df_clean = df.copy()
        col_rename = {}
        for col in df_clean.columns:
            clean_col = col.strip().lower()
            if clean_col in self.RAW_COLUMN_MAPPINGS:
                col_rename[col] = self.RAW_COLUMN_MAPPINGS[clean_col]
        df_clean = df_clean.rename(columns=col_rename)

        # Replace Inf and -Inf with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Impute or drop NaNs in critical numerical fields
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(0.0 if np.isnan(median_val) else median_val)

        # Handle Timestamps
        if "Timestamp" in df_clean.columns:
            if not np.issubdtype(df_clean["Timestamp"].dtype, np.number):
                try:
                    df_clean["Timestamp"] = pd.to_datetime(df_clean["Timestamp"]).astype(np.int64) / 1e9
                except Exception:
                    # Fallback to sequential timestamps if parsing fails
                    df_clean["Timestamp"] = np.arange(len(df_clean)) * 0.1
        else:
            df_clean["Timestamp"] = np.arange(len(df_clean)) * 0.1

        # Sort chronologically
        df_clean = df_clean.sort_values("Timestamp").reset_index(drop=True)

        return df_clean

    def extract_windowed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate flow records into fixed time windows of `window_size_sec`.
        Computes flow counts, byte/packet rates, flag ratios, IAT statistics, and port entropy.
        """
        df_clean = self.clean_dataframe(df)

        min_t = df_clean["Timestamp"].min()
        max_t = df_clean["Timestamp"].max()

        if max_t - min_t < self.window_size_sec:
            # Single window fallback
            window_bins = [min_t, min_t + self.window_size_sec]
        else:
            window_bins = np.arange(min_t, max_t + self.window_size_sec, self.window_size_sec)

        df_clean["window_id"] = pd.cut(df_clean["Timestamp"], bins=window_bins, right=False, labels=False)
        df_clean["window_id"] = df_clean["window_id"].fillna(0).astype(int)

        window_records = []
        for win_id, group in df_clean.groupby("window_id"):
            num_flows = len(group)
            win_start = min_t + win_id * self.window_size_sec
            
            # Flow volume features
            tot_fwd = group.get("Tot_Fwd_Pkts", pd.Series([1])).sum()
            tot_bwd = group.get("Tot_Bwd_Pkts", pd.Series([0])).sum()
            tot_pkts = max(tot_fwd + tot_bwd, 1)
            
            tot_fwd_bytes = group.get("TotLen_Fwd_Pkts", pd.Series([60])).sum()
            tot_bwd_bytes = group.get("TotLen_Bwd_Pkts", pd.Series([0])).sum()
            tot_bytes = tot_fwd_bytes + tot_bwd_bytes

            # Flag statistics (ratios relative to total flows in window)
            syn_cnt = group.get("SYN_Flag_Cnt", pd.Series([0])).sum()
            ack_cnt = group.get("ACK_Flag_Cnt", pd.Series([0])).sum()
            fin_cnt = group.get("FIN_Flag_Cnt", pd.Series([0])).sum()
            rst_cnt = group.get("RST_Flag_Cnt", pd.Series([0])).sum()
            psh_cnt = group.get("PSH_Flag_Cnt", pd.Series([0])).sum()
            urg_cnt = group.get("URG_Flag_Cnt", pd.Series([0])).sum()

            syn_ratio = float(syn_cnt / max(num_flows, 1))
            ack_ratio = float(ack_cnt / max(num_flows, 1))
            rst_ratio = float(rst_cnt / max(num_flows, 1))
            psh_ratio = float(psh_cnt / max(num_flows, 1))
            fin_ratio = float(fin_cnt / max(num_flows, 1))

            # Inter-arrival time and duration stats
            flow_duration_mean = float(group.get("Flow_Duration", pd.Series([0.1])).mean())
            iat_mean = float(group.get("Flow_IAT_Mean", pd.Series([0.01])).mean())
            iat_std = float(group.get("Flow_IAT_Std", pd.Series([0.0])).mean())
            iat_max = float(group.get("Flow_IAT_Max", pd.Series([0.1])).max())

            # Packet size and ratios
            down_up_ratio = float(tot_bwd_bytes / max(tot_fwd_bytes, 1))
            pkt_size_avg = float(tot_bytes / tot_pkts)

            # Port entropy / dispersion (high during reconnaissance port scans)
            dst_ports = group.get("Dst_Port", pd.Series([80]))
            unique_dst_ports = len(dst_ports.unique())
            port_entropy = float(unique_dst_ports / max(num_flows, 1))

            # Ground truth label in this window
            if "Label" in group.columns:
                # Majority label or worst-case attack label
                labels = group["Label"].tolist()
                attack_labels = [l for l in labels if str(l).lower() != "benign"]
                primary_label = attack_labels[0] if attack_labels else "Benign"
            else:
                primary_label = "Benign"

            is_attack = 1 if primary_label.lower() != "benign" else 0
            attack_cat = group.get("Attack_Category", pd.Series([primary_label])).iloc[0]

            window_records.append({
                "window_id": int(win_id),
                "timestamp_start": float(win_start),
                "num_flows": int(num_flows),
                "flow_rate_per_sec": float(num_flows / self.window_size_sec),
                "byte_rate_per_sec": float(tot_bytes / self.window_size_sec),
                "packet_rate_per_sec": float(tot_pkts / self.window_size_sec),
                "flow_duration_mean": flow_duration_mean,
                "syn_ratio": syn_ratio,
                "ack_ratio": ack_ratio,
                "rst_ratio": rst_ratio,
                "psh_ratio": psh_ratio,
                "fin_ratio": fin_ratio,
                "iat_mean": iat_mean,
                "iat_std": iat_std,
                "iat_max": iat_max,
                "down_up_ratio": down_up_ratio,
                "pkt_size_avg": pkt_size_avg,
                "port_entropy": port_entropy,
                "label": primary_label,
                "attack_category": attack_cat,
                "is_attack": is_attack
            })

        return pd.DataFrame(window_records)

    def process_file(self, input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Read CSV, process windowed flow features, and optionally save."""
        df_raw = pd.read_csv(input_csv)
        df_features = self.extract_windowed_features(df_raw)
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df_features.to_csv(output_csv, index=False)
        return df_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract windowed flow features from CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="data/processed/flow_features.csv", help="Output CSV path")
    parser.add_argument("--window", type=float, default=5.0, help="Window size in seconds")
    args = parser.parse_args()

    extractor = FlowFeatureExtractor(window_size_sec=args.window)
    result = extractor.process_file(args.input, args.output)
    print(f"Extracted {len(result)} flow feature windows -> {args.output}")
