"""
Packet-Level Feature Extractor.
Parses raw PCAP captures using Scapy to extract fine-grained packet-level signals:
TTL distributions, TCP window dynamics, fragmentation, payload entropy, and port scan sequence randomness.
"""

import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

try:
    from scapy.all import rdpcap, IP, TCP, UDP, Raw
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def calculate_shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy (0 to 8.0) of byte payload."""
    if not data:
        return 0.0
    entropy = 0.0
    length = len(data)
    counts = {}
    for b in data:
        counts[b] = counts.get(b, 0) + 1
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return float(entropy)


class PacketFeatureExtractor:
    """Extracts fine-grained packet dynamics from PCAP files grouped by time windows."""

    def __init__(self, window_size_sec: float = 2.0):
        self.window_size_sec = window_size_sec

    def extract_from_pcap(self, pcap_path: str) -> pd.DataFrame:
        """
        Parse PCAP file and aggregate packet metrics into fixed time windows.
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required to parse PCAP files. Install via `pip install scapy`.")

        packets = rdpcap(pcap_path)
        if len(packets) == 0:
            return pd.DataFrame()

        # Extract per-packet raw attributes
        pkt_data = []
        for pkt in packets:
            if not pkt.haslayer(IP):
                continue
            
            t = float(pkt.time)
            ip_layer = pkt[IP]
            ttl = int(ip_layer.ttl)
            proto = int(ip_layer.proto)
            pkt_len = int(len(pkt))
            frag_flag = int(1 if (ip_layer.flags & 1 or ip_layer.frag > 0) else 0)

            # TCP specific
            win_size = 0
            is_syn = 0
            is_rst = 0
            is_ack = 0
            sport = 0
            dport = 0
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                sport = int(tcp.sport)
                dport = int(tcp.dport)
                win_size = int(tcp.window)
                flags = str(tcp.flags)
                is_syn = int(1 if "S" in flags else 0)
                is_rst = int(1 if "R" in flags else 0)
                is_ack = int(1 if "A" in flags else 0)
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                sport = int(udp.sport)
                dport = int(udp.dport)

            # Payload entropy
            payload_entropy = 0.0
            if pkt.haslayer(Raw):
                payload_entropy = calculate_shannon_entropy(pkt[Raw].load)

            pkt_data.append({
                "time": t,
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "proto": proto,
                "sport": sport,
                "dport": dport,
                "ttl": ttl,
                "win_size": win_size,
                "frag_flag": frag_flag,
                "pkt_len": pkt_len,
                "is_syn": is_syn,
                "is_rst": is_rst,
                "is_ack": is_ack,
                "payload_entropy": payload_entropy
            })

        df_pkts = pd.DataFrame(pkt_data)
        if df_pkts.empty:
            return pd.DataFrame()

        df_pkts = df_pkts.sort_values("time").reset_index(drop=True)
        min_t = df_pkts["time"].min()
        max_t = df_pkts["time"].max()

        if max_t - min_t < self.window_size_sec:
            bins = [min_t, min_t + self.window_size_sec]
        else:
            bins = np.arange(min_t, max_t + self.window_size_sec, self.window_size_sec)

        df_pkts["window_id"] = pd.cut(df_pkts["time"], bins=bins, right=False, labels=False)
        df_pkts["window_id"] = df_pkts["window_id"].fillna(0).astype(int)

        window_records = []
        for win_id, grp in df_pkts.groupby("window_id"):
            n_pkts = len(grp)
            win_start = min_t + win_id * self.window_size_sec

            # TTL statistics
            ttl_vals = grp["ttl"].values
            ttl_mean = float(np.mean(ttl_vals))
            ttl_var = float(np.var(ttl_vals)) if n_pkts > 1 else 0.0

            # TCP Window dynamics
            win_vals = grp[grp["win_size"] > 0]["win_size"].values
            win_mean = float(np.mean(win_vals)) if len(win_vals) > 0 else 0.0
            win_var = float(np.var(win_vals)) if len(win_vals) > 1 else 0.0

            # IP Fragmentations & RST bursts
            frag_ratio = float(grp["frag_flag"].sum() / n_pkts)
            rst_ratio = float(grp["is_rst"].sum() / n_pkts)

            # SYN and ACK ratios
            syn_ratio = float(grp["is_syn"].sum() / n_pkts) if n_pkts > 0 else 0.0
            ack_ratio = float(grp["is_ack"].sum() / n_pkts) if n_pkts > 0 else 0.0

            # Real packet throughput per second
            packet_rate_per_sec = float(n_pkts / self.window_size_sec)

            # Payload Entropy
            entropy_vals = grp["payload_entropy"].values
            entropy_mean = float(np.mean(entropy_vals))
            entropy_max = float(np.max(entropy_vals))

            # Small vs Large packet proportions
            small_pkt_ratio = float((grp["pkt_len"] < 100).sum() / n_pkts)
            large_pkt_ratio = float((grp["pkt_len"] > 1000).sum() / n_pkts)

            # Target ports dynamics, active connection density, and port entropy
            dports = grp[grp["dport"] > 0]["dport"].values
            unique_dports = len(np.unique(dports)) if len(dports) > 0 else 0
            active_connection_density = float(unique_dports / n_pkts) if n_pkts > 0 else 0.0

            if len(dports) > 2:
                port_diffs = np.abs(np.diff(dports))
                port_seq_randomness = float(np.std(port_diffs) / (np.mean(port_diffs) + 1.0))
                # True Shannon entropy
                _, counts = np.unique(dports, return_counts=True)
                p = counts / len(dports)
                shannon_raw = float(-np.sum(p * np.log2(p + 1e-9)))
                # Normalized by log2(n) so it stays within [0.0, 1.0]
                norm_denom = np.log2(max(len(counts), 2))
                port_entropy = float(np.clip(shannon_raw / norm_denom, 0.0, 1.0))
            else:
                port_seq_randomness = 0.0
                port_entropy = 0.0

            window_records.append({
                "window_id": int(win_id),
                "timestamp_start": float(win_start),
                "packet_count": int(n_pkts),
                "packet_rate_per_sec": packet_rate_per_sec,
                "syn_ratio": syn_ratio,
                "ack_ratio": ack_ratio,
                "active_connection_density": active_connection_density,
                "ttl_mean": ttl_mean,
                "ttl_var": ttl_var,
                "tcp_win_mean": win_mean,
                "tcp_win_var": win_var,
                "ip_frag_ratio": frag_ratio,
                "rst_burst_ratio": rst_ratio,
                "payload_entropy_mean": entropy_mean,
                "payload_entropy_max": entropy_max,
                "small_pkt_ratio": small_pkt_ratio,
                "large_pkt_ratio": large_pkt_ratio,
                "port_seq_randomness": port_seq_randomness,
                "port_entropy": port_entropy
            })

        return pd.DataFrame(window_records)

    def process_file(self, input_pcap: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """Process PCAP file and output packet-level feature CSV."""
        df_feat = self.extract_from_pcap(input_pcap)
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df_feat.to_csv(output_csv, index=False)
        return df_feat


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract windowed packet features from PCAP")
    parser.add_argument("--input", required=True, help="Input PCAP path")
    parser.add_argument("--output", default="data/processed/packet_features.csv", help="Output CSV path")
    parser.add_argument("--window", type=float, default=2.0, help="Window size in seconds")
    args = parser.parse_args()

    extractor = PacketFeatureExtractor(window_size_sec=args.window)
    result = extractor.process_file(args.input, args.output)
    print(f"Extracted {len(result)} packet feature windows -> {args.output}")
