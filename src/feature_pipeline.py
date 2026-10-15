"""
Unified Feature Pipeline CLI Entrypoint.
Accepts raw NetFlow CSV or PCAP file, extracts dual-level flow/packet features,
and outputs a normalized 28D state feature matrix for World Model dynamics.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Append project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.packet_extractor import PacketFeatureExtractor, SCAPY_AVAILABLE
from src.feature_pipeline.fusion import DualLevelFusionPipeline


def main():
    parser = argparse.ArgumentParser(description="Extract and fuse dual-level features from PCAP/CSV")
    parser.add_argument("--input", required=True, help="Input raw PCAP or NetFlow CSV file path")
    parser.add_argument("--output", default="data/processed/fused_state.csv", help="Output fused features CSV path")
    parser.add_argument("--window", type=float, default=5.0, help="Sliding temporal observation window in seconds")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"[*] Processing {input_path.name} with window={args.window}s...")
    flow_df = None
    pkt_df = None

    if input_path.suffix.lower() in [".pcap", ".pcapng", ".cap"]:
        if not SCAPY_AVAILABLE:
            print("[!] Scapy is required for PCAP parsing. Run `pip install scapy`.")
            sys.exit(1)
        pkt_ext = PacketFeatureExtractor(window_size_sec=args.window)
        pkt_df = pkt_ext.extract_from_pcap(str(input_path))
        print(f"[+] Extracted packet-level dynamics: {len(pkt_df)} windows")
    else: # CSV file
        raw_df = pd.read_csv(input_path)
        flow_ext = FlowFeatureExtractor(window_size_sec=args.window)
        flow_df = flow_ext.extract_windowed_features(raw_df)
        print(f"[+] Extracted flow-level aggregates: {len(flow_df)} windows")

    fusion = DualLevelFusionPipeline()
    fused_df = fusion.fuse_dataframes(df_flow=flow_df, df_packet=pkt_df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Fused 28D state representation saved: {fused_df.shape} -> {out_path}")


if __name__ == "__main__":
    main()
