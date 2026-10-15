"""
CIC-IDS-2017 Full MachineLearningCVE Dataset Processor.
Processes all 8 official CSV files from MachineLearningCVE:
- Friday PortScan (Reconnaissance)
- Tuesday FTP/SSH Patator (Initial Access)
- Thursday Infiltration (Lateral Movement)
- Friday Morning Botnet (Command & Control)
- Wednesday DoS / Friday DDoS (Impact & Exfiltration)
- Monday Pure Benign Baseline

Extracts all attack flows and a balanced sample of benign flows, cleans NaNs/Infs,
aggregates into 5-second observation windows, and fuses into the 28D state matrix.
"""

import os
import glob
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.fusion import DualLevelFusionPipeline


def process_machine_learning_cve(
    cve_dir: str = "MachineLearningCVE",
    output_fused_csv: str = "data/processed/fused_state.csv",
    benign_per_file: int = 8000,
    window_size_sec: float = 5.0
) -> pd.DataFrame:
    cve_path = Path(cve_dir)
    if not cve_path.exists():
        raise FileNotFoundError(f"MachineLearningCVE directory not found at {cve_path}")

    csv_files = sorted(glob.glob(str(cve_path / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cve_path}")

    print("=" * 80)
    print("   CIC-IDS-2017 (MachineLearningCVE) FULL DATASET PROCESSING ENGINE")
    print(f"   Found {len(csv_files)} official CSV files in {cve_path}")
    print("=" * 80)

    all_sampled_chunks = []
    total_attack_rows = 0
    total_benign_rows = 0

    # Ordered mapping to establish realistic multi-stage kill-chain progression
    # 1. PortScan -> 2. Patator/WebAttacks -> 3. Infiltration -> 4. Botnet -> 5. DoS/DDoS
    file_priority = {
        "Monday": 0,       # Benign baseline
        "PortScan": 1,     # Reconnaissance
        "Tuesday": 2,      # Initial Access (Patator)
        "WebAttacks": 2,   # Initial Access (Web)
        "Infilteration": 3,# Lateral Movement
        "Morning": 4,      # Command & Control (Botnet)
        "Wednesday": 5,    # DoS
        "DDos": 5          # DDoS
    }

    def get_priority(filename: str) -> int:
        for k, p in file_priority.items():
            if k.lower() in filename.lower():
                return p
        return 99

    sorted_csv_files = sorted(csv_files, key=lambda f: (get_priority(f), f))

    for idx, filepath in enumerate(sorted_csv_files):
        fname = os.path.basename(filepath)
        print(f"\n[{idx+1}/{len(sorted_csv_files)}] Ingesting {fname}...")
        t0 = time.time()

        file_attacks = []
        file_benign = []
        benign_collected = 0

        # Read in memory-efficient chunks of 50,000 rows
        chunk_iter = pd.read_csv(
            filepath,
            chunksize=50000,
            encoding="utf-8",
            low_memory=False
        )

        for chunk in chunk_iter:
            # Find the label column (case-insensitive and stripping whitespace)
            label_col = None
            for c in chunk.columns:
                if "label" in c.lower():
                    label_col = c
                    break

            if not label_col:
                continue

            # Identify attacks vs benign
            labels_clean = chunk[label_col].astype(str).str.strip()
            is_attack = labels_clean.str.upper() != "BENIGN"

            attacks = chunk[is_attack]
            if len(attacks) > 0:
                file_attacks.append(attacks)
                total_attack_rows += len(attacks)

            if benign_collected < benign_per_file:
                needed = benign_per_file - benign_collected
                benign = chunk[~is_attack].iloc[:needed]
                if len(benign) > 0:
                    file_benign.append(benign)
                    benign_collected += len(benign)
                    total_benign_rows += len(benign)

        # Concatenate this file's captured flows
        file_dfs = []
        if file_benign:
            file_dfs.extend(file_benign)
        if file_attacks:
            file_dfs.extend(file_attacks)

        if file_dfs:
            combined_file_df = pd.concat(file_dfs, ignore_index=True)
            all_sampled_chunks.append(combined_file_df)
            n_att = sum(len(a) for a in file_attacks)
            print(f"    -> Extracted {len(combined_file_df):,} flows ({n_att:,} attacks, {benign_collected:,} benign) in {time.time()-t0:.1f}s")

    print("\n" + "=" * 80)
    print(f"[+] Total raw flows extracted across all 8 files: {total_attack_rows + total_benign_rows:,}")
    print(f"    • Genuine Attack Flows: {total_attack_rows:,}")
    print(f"    • Baseline Benign Flows: {total_benign_rows:,}")
    print("=" * 80)

    print("\n[*] Assembling chronological kill-chain dataset...")
    master_df = pd.concat(all_sampled_chunks, ignore_index=True)

    # Standardize label column name
    for c in master_df.columns:
        if "label" in c.lower() and c != "Label":
            master_df = master_df.rename(columns={c: "Label"})
            break

    print(f"[*] Extracting 5-second windowed feature dynamics using FlowFeatureExtractor...")
    extractor = FlowFeatureExtractor(window_size_sec=window_size_sec)
    flow_features_df = extractor.extract_windowed_features(master_df)
    print(f"[+] Generated {len(flow_features_df):,} continuous observation windows!")

    print(f"[*] Fusing dual-level state representations into 28D state matrix S_t...")
    fusion = DualLevelFusionPipeline()
    fused_state_df = fusion.fuse_dataframes(df_flow=flow_features_df)

    # Save to data/processed
    out_path = Path(output_fused_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fused_state_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Master fused state matrix saved: {fused_state_df.shape} -> {out_path}")

    # Also save a dedicated copy named cicids2017_fused_state.csv
    cic_copy_path = out_path.parent / "cicids2017_fused_state.csv"
    fused_state_df.to_csv(cic_copy_path, index=False)
    print(f"[SUCCESS] Archive copy saved: {cic_copy_path}")

    # Print summary of attack stages present in the windows
    print("\n[+] Windowed Attack Distribution in Fused Dataset:")
    print(fused_state_df["label"].value_counts().to_string())

    return fused_state_df


if __name__ == "__main__":
    process_machine_learning_cve()
