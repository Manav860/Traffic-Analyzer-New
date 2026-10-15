#!/usr/bin/env python3
"""
Multi-Threaded Network Attack & Port Sweep Simulator
=====================================================
Performs a rapid SYN and UDP port sweep across 10,000 ports using standard
Python sockets. Requires no special tools (no nmap, no raw sockets, no root/admin).

Usage:
    python attack_sim.py <target_ip> [duration_seconds]

Example:
    python attack_sim.py 10.10.13.162 40
    python attack_sim.py 127.0.0.1 20
"""

import sys
import time
import socket
import random
import threading
import argparse
from typing import List


# Ensure stdout does not crash on Windows non-UTF8 terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class AttackSimulator:
    """Multi-threaded port sweep and traffic generation simulator."""

    def __init__(
        self,
        target_ip: str,
        duration_sec: int = 40,
        num_threads: int = 60,
        num_ports: int = 10000,
        base_port: int = 1024
    ):
        self.target_ip = target_ip
        self.duration_sec = duration_sec
        self.num_threads = num_threads
        self.num_ports = num_ports
        self.base_port = base_port

        # Build port list (10,000 ports spread across range)
        max_p = min(65535, base_port + num_ports)
        self.ports: List[int] = list(range(base_port, max_p))
        random.shuffle(self.ports)

        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.total_packets_sent = 0
        self.total_syn_sent = 0
        self.total_udp_sent = 0

    def _worker(self, worker_id: int):
        """Worker thread executing rapid SYN connect_ex attempts and UDP probes."""
        port_idx = worker_id * (len(self.ports) // self.num_threads)
        ports_len = len(self.ports)
        payload = b"PROBE_SWEEP_" + str(worker_id).encode()

        while not self.stop_event.is_set():
            port = self.ports[port_idx % ports_len]
            port_idx += 1

            # 1. TCP SYN probe via non-blocking connect_ex
            # connect_ex() causes OS network stack to transmit an initial TCP SYN packet
            syn_ok = False
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.04)
                s.connect_ex((self.target_ip, port))
                s.close()
                syn_ok = True
            except Exception:
                pass

            # 2. Interleaved UDP probe (1 in 10 to maintain ~90%+ SYN ratio)
            udp_ok = False
            if random.random() < 0.10:
                try:
                    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    u.sendto(payload, (self.target_ip, port))
                    u.close()
                    udp_ok = True
                except Exception:
                    pass

            with self.lock:
                if syn_ok:
                    self.total_syn_sent += 1
                    self.total_packets_sent += 1
                if udp_ok:
                    self.total_udp_sent += 1
                    self.total_packets_sent += 1

    def run(self):
        """Execute simulation across worker threads with live console telemetry."""
        print("=" * 68)
        print("[*] MULTI-THREADED NETWORK ATTACK SIMULATOR")
        print("=" * 68)
        print(f" Target IP         : {self.target_ip}")
        print(f" Duration          : {self.duration_sec} seconds")
        print(f" Concurrent Threads: {self.num_threads}")
        print(f" Target Port Pool  : {len(self.ports)} ports ({self.base_port} - {self.ports[-1]})")
        print(f" Traffic Profile   : High-density SYN sweep + UDP probes")
        print("=" * 68)
        print("[+] Launching attack threads...")

        threads = []
        for i in range(self.num_threads):
            t = threading.Thread(target=self._worker, args=(i,), daemon=True)
            threads.append(t)
            t.start()

        start_time = time.time()
        try:
            while not self.stop_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= self.duration_sec:
                    self.stop_event.set()
                    break

                with self.lock:
                    pkts = self.total_packets_sent
                    syns = self.total_syn_sent
                    udps = self.total_udp_sent

                rate = pkts / max(elapsed, 0.1)
                syn_pct = (syns / max(pkts, 1)) * 100.0
                remaining = max(0, self.duration_sec - int(elapsed))

                sys.stdout.write(
                    f"\r[{elapsed:04.1f}s / {self.duration_sec}s] "
                    f"Sent: {pkts:,} pkts | "
                    f"Rate: {rate:5.0f} pkt/s | "
                    f"SYN: {syn_pct:.1f}% | "
                    f"Remaining: {remaining}s   "
                )
                sys.stdout.flush()
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[!] Interrupted by user. Stopping threads...")
            self.stop_event.set()

        # Wait briefly for workers
        time.sleep(0.2)
        total_time = time.time() - start_time
        avg_rate = self.total_packets_sent / max(total_time, 0.1)

        print("\n" + "=" * 68)
        print("[+] SIMULATION COMPLETE")
        print(f" Total Elapsed   : {total_time:.2f} seconds")
        print(f" Total Packets   : {self.total_packets_sent:,}")
        print(f" TCP SYN Probes  : {self.total_syn_sent:,}")
        print(f" UDP Probes      : {self.total_udp_sent:,}")
        print(f" Average Rate    : {avg_rate:.1f} packets/second")
        print("=" * 68)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-threaded rapid SYN + UDP port sweep simulator across 10,000 ports."
    )
    parser.add_argument(
        "target_ip",
        type=str,
        nargs="?",
        default="127.0.0.1",
        help="Target IP address of the traffic analyzer machine (e.g. 10.10.13.162 or 127.0.0.1)"
    )
    parser.add_argument(
        "duration",
        type=int,
        nargs="?",
        default=40,
        help="Attack duration in seconds (default: 40)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=60,
        help="Number of concurrent worker threads (default: 60)"
    )
    parser.add_argument(
        "--ports",
        type=int,
        default=10000,
        help="Total number of target ports to sweep across (default: 10000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim = AttackSimulator(
        target_ip=args.target_ip,
        duration_sec=args.duration,
        num_threads=args.threads,
        num_ports=args.ports
    )
    sim.run()
