#!/bin/bash
# ==============================================================================
# PCAP Replay Script via tcpreplay (SIH26153 Sandbox Fallback)
# Replays pre-recorded or synthetic PCAP network traffic traces over the sandbox
# bridge network interface without requiring live exploit tools.
# ==============================================================================

set -e

INTERFACE="${1:-eth0}"
PCAP_FILE="${2:-../data/raw/synthetic_traffic.pcap}"
MBPS="${3:-10}"

echo "============================================================"
echo "   SIH26153: TCPSANDBOX PCAP REPLAY ENGINE"
echo "   Interface: ${INTERFACE} | File: ${PCAP_FILE}"
echo "============================================================"

if [ ! -f "${PCAP_FILE}" ]; then
    echo "[!] PCAP file not found at ${PCAP_FILE}. Generating synthetic PCAP trace..."
    python3 ../src/utils/synthetic_generator.py
fi

if command -v tcpreplay &> /dev/null; then
    echo "[*] Replaying ${PCAP_FILE} on ${INTERFACE} at ${MBPS} Mbps using tcpreplay..."
    tcpreplay --intf1="${INTERFACE}" --mbps="${MBPS}" "${PCAP_FILE}"
    echo "[+] tcpreplay replay complete."
else
    echo "[*] tcpreplay not installed. Running Python Scapy socket packet sender fallback..."
    python3 -c "
import sys
from scapy.all import rdpcap, sendp
pcap_path = '${PCAP_FILE}'
pkts = rdpcap(pcap_path)
print(f'Sending {len(pkts)} packets on interface ${INTERFACE}...')
sendp(pkts, iface='${INTERFACE}', verbose=False)
print('Replay complete.')
"
fi

echo "============================================================"
echo "   PCAP REPLAY COMPLETED"
echo "============================================================"
