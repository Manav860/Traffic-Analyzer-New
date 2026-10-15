"""
Synthetic Network Traffic Generator.
Generates realistic multi-stage kill-chain network flows (CSV) and raw PCAP captures
demonstrating the progression:
Normal Traffic -> Reconnaissance (PortScan) -> Initial Access (Brute Force) -> Lateral Movement (Infiltration) -> C2 / Exfiltration
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from src.utils.seeding import set_seed

# Scapy is optional for PCAP generation, CSV generation works with pure python/numpy/pandas
try:
    from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


def generate_synthetic_flows(
    num_samples: int = 5000,
    random_seed: int = 42,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate synthetic flow records matching the CIC-IDS-2018 / NetFlow schema.
    Includes temporal sequences of benign traffic and realistic multi-stage attacks.
    """
    set_seed(random_seed)
    
    # Attack stages and temporal ordering
    # 0: Benign, 1: PortScan (Recon), 2: BruteForce (Initial Access), 3: Infiltration (Lateral Movement), 4: Bot_Exfiltration (C2)
    records = []
    base_time = 1700000000.0  # Unix timestamp
    
    # Generate sequential time windows (e.g. 5-second windows)
    # Stage progression:
    # 0.0 - 0.4: Benign baseline
    # 0.4 - 0.6: Reconnaissance (PortScan)
    # 0.6 - 0.75: Initial Access (Brute Force)
    # 0.75 - 0.90: Lateral Movement (Infiltration)
    # 0.90 - 1.0: C2 / Data Exfiltration
    
    current_time = base_time
    for i in range(num_samples):
        progress = i / num_samples
        current_time += np.random.exponential(scale=0.1) # Realistic inter-flow timing
        
        # Determine active stage
        if progress < 0.40:
            stage = "Benign"
            label = "Benign"
            attack_cat = "Benign"
            is_attack = 0
        elif progress < 0.58:
            # Reconnaissance stage
            if np.random.rand() < 0.7:
                stage = "Reconnaissance"
                label = "PortScan"
                attack_cat = "Reconnaissance"
                is_attack = 1
            else:
                stage = "Benign"
                label = "Benign"
                attack_cat = "Benign"
                is_attack = 0
        elif progress < 0.75:
            # Initial Access stage
            if np.random.rand() < 0.75:
                stage = "Initial Access"
                label = "FTP-BruteForce" if np.random.rand() < 0.5 else "SSH-Bruteforce"
                attack_cat = "Initial Access"
                is_attack = 1
            else:
                stage = "Benign"
                label = "Benign"
                attack_cat = "Benign"
                is_attack = 0
        elif progress < 0.88:
            # Lateral Movement / Infiltration
            if np.random.rand() < 0.8:
                stage = "Lateral Movement"
                label = "Infiltration"
                attack_cat = "Lateral Movement"
                is_attack = 1
            else:
                stage = "Benign"
                label = "Benign"
                attack_cat = "Benign"
                is_attack = 0
        else:
            # C2 & Exfiltration
            if np.random.rand() < 0.85:
                stage = "Command and Control"
                label = "Bot"
                attack_cat = "Command and Control"
                is_attack = 1
            else:
                stage = "Benign"
                label = "Benign"
                attack_cat = "Benign"
                is_attack = 0
        
        # IP and Port generation based on attack stage
        if label == "Benign":
            src_ip = f"192.168.1.{np.random.randint(10, 50)}"
            dst_ip = f"10.0.0.{np.random.randint(1, 10)}"
            src_port = int(np.random.randint(1024, 65535))
            dst_port = int(np.random.choice([80, 443, 53, 8080, 22]))
            proto = int(np.random.choice([6, 17], p=[0.85, 0.15])) # TCP/UDP
            duration = float(np.random.uniform(0.1, 15.0))
            fwd_pkts = int(np.random.poisson(15) + 2)
            bwd_pkts = int(np.random.poisson(18) + 2)
            fwd_bytes = int(fwd_pkts * np.random.uniform(100, 1400))
            bwd_bytes = int(bwd_pkts * np.random.uniform(200, 1450))
            syn_flags = int(1) if proto == 6 else 0
            ack_flags = int(np.random.randint(1, 10)) if proto == 6 else 0
            fin_flags = int(1) if proto == 6 and np.random.rand() < 0.8 else 0
            rst_flags = int(0)
            psh_flags = int(np.random.randint(0, 4))
            urg_flags = int(0)
            iat_mean = float(np.random.uniform(0.01, 0.5))
            iat_std = float(iat_mean * np.random.uniform(0.1, 0.4))
            iat_max = float(iat_mean * np.random.uniform(1.5, 3.0))
            ttl_mean = float(np.random.choice([64.0, 128.0]))
            ttl_std = float(np.random.uniform(0.0, 0.5))
            win_size_mean = float(np.random.uniform(14000, 65535))
        
        elif label == "PortScan":
            src_ip = "192.168.1.105" # Attacker IP
            dst_ip = "10.0.0.5"      # Victim target
            src_port = int(np.random.randint(40000, 60000))
            dst_port = int(np.random.randint(1, 1024)) # Probing low ports
            proto = 6 # TCP SYN scan
            duration = float(np.random.uniform(0.0001, 0.05)) # Fast probe
            fwd_pkts = int(np.random.choice([1, 2]))
            bwd_pkts = int(np.random.choice([0, 1]))
            fwd_bytes = int(fwd_pkts * 60)
            bwd_bytes = int(bwd_pkts * 40)
            syn_flags = 1
            ack_flags = int(1 if bwd_pkts > 0 else 0)
            fin_flags = 0
            rst_flags = int(1 if bwd_pkts > 0 else 0) # RST received from closed ports
            psh_flags = 0
            urg_flags = 0
            iat_mean = float(np.random.uniform(0.0001, 0.01)) # High rate
            iat_std = float(0.00005)
            iat_max = float(0.02)
            ttl_mean = float(64.0)
            ttl_std = float(0.0) # Uniform TTL in scanner tool
            win_size_mean = float(1024.0) # Fixed window size from nmap
            
        elif "BruteForce" in label:
            src_ip = "192.168.1.105"
            dst_ip = "10.0.0.5"
            src_port = int(np.random.randint(30000, 65000))
            dst_port = int(22 if "SSH" in label else 21)
            proto = 6
            duration = float(np.random.uniform(0.05, 0.5))
            fwd_pkts = int(np.random.randint(8, 20))
            bwd_pkts = int(np.random.randint(6, 18))
            fwd_bytes = int(fwd_pkts * np.random.uniform(60, 150))
            bwd_bytes = int(bwd_pkts * np.random.uniform(60, 180))
            syn_flags = 1
            ack_flags = int(fwd_pkts - 1)
            fin_flags = 1
            rst_flags = int(np.random.choice([0, 1], p=[0.7, 0.3]))
            psh_flags = int(np.random.randint(2, 6)) # High PSH for payload delivery
            urg_flags = 0
            iat_mean = float(np.random.uniform(0.005, 0.05))
            iat_std = float(np.random.uniform(0.001, 0.01))
            iat_max = float(0.1)
            ttl_mean = float(64.0)
            ttl_std = float(np.random.uniform(0.0, 0.2))
            win_size_mean = float(29200.0)
            
        elif label == "Infiltration":
            # Pivoting / Lateral movement from compromised host 10.0.0.5 to internal 10.0.0.20
            src_ip = "10.0.0.5"
            dst_ip = "10.0.0.20"
            src_port = int(np.random.randint(40000, 65000))
            dst_port = int(np.random.choice([445, 139, 3389, 22])) # SMB / RDP / SSH
            proto = 6
            duration = float(np.random.uniform(0.5, 8.0))
            fwd_pkts = int(np.random.randint(20, 60))
            bwd_pkts = int(np.random.randint(15, 50))
            fwd_bytes = int(fwd_pkts * np.random.uniform(150, 800))
            bwd_bytes = int(bwd_pkts * np.random.uniform(100, 500))
            syn_flags = 1
            ack_flags = int(fwd_pkts - 2)
            fin_flags = 1
            rst_flags = 0
            psh_flags = int(np.random.randint(4, 12))
            urg_flags = 0
            iat_mean = float(np.random.uniform(0.02, 0.15))
            iat_std = float(np.random.uniform(0.01, 0.05))
            iat_max = float(0.8)
            ttl_mean = float(128.0) # Windows internal host
            ttl_std = float(0.1)
            win_size_mean = float(64240.0)
            
        else: # Bot / C2 / Exfiltration
            src_ip = "10.0.0.20"
            dst_ip = "203.0.113.88" # External rogue C2 server
            src_port = int(np.random.randint(50000, 65000))
            dst_port = int(np.random.choice([443, 8080, 8443, 53]))
            proto = 6
            duration = float(np.random.uniform(5.0, 30.0))
            fwd_pkts = int(np.random.randint(80, 300)) # Heavy data upload
            bwd_pkts = int(np.random.randint(10, 40))
            fwd_bytes = int(fwd_pkts * np.random.uniform(800, 1400)) # Large payloads
            bwd_bytes = int(bwd_pkts * 80)
            syn_flags = 1
            ack_flags = int(fwd_pkts)
            fin_flags = 1
            rst_flags = 0
            psh_flags = int(np.random.randint(15, 50))
            urg_flags = 0
            iat_mean = float(np.random.uniform(0.05, 0.2)) # Periodic beaconing
            iat_std = float(0.002) # Highly periodic (low variance)
            iat_max = float(0.5)
            ttl_mean = float(64.0)
            ttl_std = float(0.0)
            win_size_mean = float(65535.0)

        # Calculate rates and metrics
        safe_duration = max(duration, 0.0001)
        tot_bytes = fwd_bytes + bwd_bytes
        tot_pkts = fwd_pkts + bwd_pkts
        flow_byts_s = float(tot_bytes / safe_duration)
        flow_pkts_s = float(tot_pkts / safe_duration)
        down_up_ratio = float(bwd_bytes / max(fwd_bytes, 1))
        pkt_size_avg = float(tot_bytes / max(tot_pkts, 1))
        
        record = {
            "Timestamp": current_time,
            "Src_IP": src_ip,
            "Dst_IP": dst_ip,
            "Src_Port": src_port,
            "Dst_Port": dst_port,
            "Protocol": proto,
            "Flow_Duration": safe_duration,
            "Tot_Fwd_Pkts": fwd_pkts,
            "Tot_Bwd_Pkts": bwd_pkts,
            "TotLen_Fwd_Pkts": fwd_bytes,
            "TotLen_Bwd_Pkts": bwd_bytes,
            "Flow_Byts_s": flow_byts_s,
            "Flow_Pkts_s": flow_pkts_s,
            "Flow_IAT_Mean": iat_mean,
            "Flow_IAT_Std": iat_std,
            "Flow_IAT_Max": iat_max,
            "Flow_IAT_Min": max(0.0, iat_mean - 2 * iat_std),
            "Fwd_IAT_Tot": duration * 0.9,
            "Bwd_IAT_Tot": duration * 0.8 if bwd_pkts > 0 else 0.0,
            "SYN_Flag_Cnt": syn_flags,
            "ACK_Flag_Cnt": ack_flags,
            "FIN_Flag_Cnt": fin_flags,
            "RST_Flag_Cnt": rst_flags,
            "PSH_Flag_Cnt": psh_flags,
            "URG_Flag_Cnt": urg_flags,
            "Down_Up_Ratio": down_up_ratio,
            "Pkt_Size_Avg": pkt_size_avg,
            "Fwd_Seg_Size_Avg": float(fwd_bytes / max(fwd_pkts, 1)),
            "Bwd_Seg_Size_Avg": float(bwd_bytes / max(bwd_pkts, 1)),
            "Init_Win_Bytes_Fwd": win_size_mean,
            "Init_Win_Bytes_Bwd": win_size_mean if bwd_pkts > 0 else 0.0,
            "TTL_Mean": ttl_mean,
            "TTL_Std": ttl_std,
            "Label": label,
            "Attack_Category": attack_cat,
            "Is_Attack": is_attack
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
    return df


def generate_synthetic_pcap(
    output_path: str,
    duration_seconds: int = 60,
    random_seed: int = 42
) -> str:
    """
    Generate a synthetic PCAP trace with Scapy showcasing the multi-stage attack.
    Stage 1 (0-15s): Normal web traffic
    Stage 2 (15-30s): Port scan (Reconnaissance)
    Stage 3 (30-45s): SSH/HTTP brute force (Initial Access)
    Stage 4 (45-60s): Pivoting & Data Exfiltration (Lateral Movement & C2)
    """
    if not SCAPY_AVAILABLE:
        raise ImportError("Scapy is required for PCAP generation.")
        
    set_seed(random_seed)
    packets = []
    curr_time = 1700000000.0
    
    # 1. Normal traffic (0 - 15s)
    while curr_time < 1700000015.0:
        curr_time += np.random.exponential(0.05)
        src_ip = f"192.168.1.{np.random.randint(10, 30)}"
        dst_ip = "10.0.0.5"
        sport = int(np.random.randint(1024, 65535))
        dport = int(np.random.choice([80, 443]))
        
        # SYN -> SYN/ACK -> ACK -> HTTP GET -> ACK
        p_syn = Ether()/IP(src=src_ip, dst=dst_ip, ttl=64)/TCP(sport=sport, dport=dport, flags="S", seq=1000, window=64240)
        p_syn.time = curr_time
        packets.append(p_syn)
        
        p_ack = Ether()/IP(src=src_ip, dst=dst_ip, ttl=64)/TCP(sport=sport, dport=dport, flags="A", seq=1001, ack=2001, window=64240)/Raw(b"GET / HTTP/1.1\r\nHost: example.local\r\n\r\n")
        p_ack.time = curr_time + 0.01
        packets.append(p_ack)
        
    # 2. PortScan (15 - 30s)
    attacker_ip = "192.168.1.105"
    victim_ip = "10.0.0.5"
    for port in range(20, 100):
        curr_time += 0.15
        sport = int(np.random.randint(40000, 60000))
        # TCP SYN scan with fixed window and zero payload
        p_scan = Ether()/IP(src=attacker_ip, dst=victim_ip, ttl=64)/TCP(sport=sport, dport=port, flags="S", seq=5000, window=1024)
        p_scan.time = curr_time
        packets.append(p_scan)
        
    # 3. Brute Force (30 - 45s)
    for attempt in range(40):
        curr_time += 0.35
        sport = int(np.random.randint(30000, 60000))
        p_ssh = Ether()/IP(src=attacker_ip, dst=victim_ip, ttl=64)/TCP(sport=sport, dport=22, flags="PA", seq=6000 + attempt*100, window=29200)/Raw(f"SSH-2.0-OpenSSH_8.2p1 user=admin attempt={attempt}\r\n".encode())
        p_ssh.time = curr_time
        packets.append(p_ssh)
        
    # 4. Infiltration / Pivoting / Exfil (45 - 60s)
    pivot_ip = "10.0.0.20"
    c2_ip = "203.0.113.88"
    for burst in range(50):
        curr_time += 0.28
        # Lateral movement SMB
        p_smb = Ether()/IP(src=victim_ip, dst=pivot_ip, ttl=128)/TCP(sport=44500 + burst, dport=445, flags="PA", seq=7000 + burst*200, window=65535)/Raw(b"\x00\x00\x00\x45\xffSMB%lateral_pivot_payload%")
        p_smb.time = curr_time
        packets.append(p_smb)
        
        # C2 Exfil beacon
        p_exfil = Ether()/IP(src=pivot_ip, dst=c2_ip, ttl=64)/TCP(sport=55000 + burst, dport=8443, flags="PA", seq=8000 + burst*500, window=65535)/Raw(b"\x17\x03\x03\x00\x80" + b"A"*120)
        p_exfil.time = curr_time + 0.05
        packets.append(p_exfil)
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wrpcap(output_path, packets)
    return output_path


if __name__ == "__main__":
    print("Generating synthetic datasets...")
    flows_df = generate_synthetic_flows(num_samples=10000, output_path="data/raw/synthetic_flows.csv")
    print(f"Generated {len(flows_df)} flow records -> data/raw/synthetic_flows.csv")
    if SCAPY_AVAILABLE:
        pcap_file = generate_synthetic_pcap("data/raw/synthetic_traffic.pcap")
        print(f"Generated PCAP trace -> {pcap_file}")
    else:
        print("Scapy not installed yet. Skipping PCAP generation.")
