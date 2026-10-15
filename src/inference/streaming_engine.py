"""
Streaming Inference Engine.
Maintains a sliding-window temporal buffer of network states.
Supports:
1. Streaming Simulation Mode (Default / Safe): Replays PCAP or CSV at controlled playback rate (no elevation required).
2. Live Interface Capture Mode (Advanced / Elevated): Sniffs live packets on network adapter (requires sudo / Administrator).
3. Continuous Network Monitor Mode: Captures rolling 2-second windows indefinitely, forecasting attack progression in real time.
"""

import time
import queue
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Generator, Tuple, List

from src.feature_pipeline.fusion import DualLevelFusionPipeline, STATE_FEATURE_NAMES
from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.packet_extractor import PacketFeatureExtractor
from src.inference.rollout import RolloutEngine


class StreamingInferenceEngine:
    """Sliding-window stream processor feeding the World Model rollout engine."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        seq_len: int = 10,
        horizon_k: int = 5,
        window_size_sec: float = 2.0
    ):
        self.rollout_engine = rollout_engine
        self.seq_len = seq_len
        self.horizon_k = horizon_k
        self.window_size_sec = window_size_sec
        self.history_buffer: List[pd.Series] = []

    def reset(self) -> None:
        """Clear historical window buffer."""
        self.history_buffer.clear()

    def push_window(self, window_record: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Add a single 2s window state to the rolling buffer.
        When buffer reaches seq_len, executes K-step forward simulation.
        """
        self.history_buffer.append(window_record)
        if len(self.history_buffer) > self.seq_len:
            self.history_buffer.pop(0)

        if len(self.history_buffer) == self.seq_len:
            df_history = pd.DataFrame(self.history_buffer)
            forecast = self.rollout_engine.simulate_from_raw_sequence(
                df_history,
                horizon_k=self.horizon_k,
                window_size_sec=self.window_size_sec
            )
            return forecast
        return None

    def replay_dataset_generator(
        self,
        fused_df: pd.DataFrame,
        delay_sec: float = 0.5
    ) -> Generator[Tuple[pd.Series, Optional[Dict[str, Any]]], None, None]:
        """
        Generator for Streaming Simulation Mode.
        Streams fused state windows one by one to simulate real-time network traffic.
        """
        self.reset()
        for idx, row in fused_df.iterrows():
            forecast = self.push_window(row)
            yield row, forecast
            if delay_sec > 0:
                time.sleep(delay_sec)


class LiveCaptureSniffer:
    """
    Live Network Interface Sniffer (Advanced Mode).
    Captures live packets on a selected interface using Scapy sniff() with a
    fixed timeout — no threading race condition, works with Npcap on Windows.
    Requires root (Linux) or Administrator (Windows) privileges.

    DEPLOYMENT NOTE:
    ─────────────────────────────────────────────────────────────────────────
    On a home/office switched network, this captures traffic TO and FROM this
    machine only (your own flows). To monitor the WHOLE network (all hosts):
      • Enterprise: Configure a SPAN/mirror port on your switch/router so all
        traffic is copied to the monitoring server's port.
      • Lab/Demo: Use a hub (not switch) — all packets are broadcast to all ports.
      • Advanced: Run on the default gateway / router itself (captures all flows).
    ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, interface: Optional[str] = None, window_size_sec: float = 2.0):
        self.interface = interface
        self.window_size_sec = window_size_sec

    @staticmethod
    def get_interfaces() -> List[str]:
        """Return list of available network interface names."""
        try:
            from scapy.all import get_if_list
            return get_if_list()
        except Exception:
            return []

    @staticmethod
    def get_friendly_interfaces() -> List[Tuple[str, str]]:
        """
        Return list of (friendly_label, pcap_iface) tuples sorted with active internet
        adapters first. Uses Scapy's internal conf.ifaces dictionary directly for instant,
        reliable detection without slow or failing subprocess calls.
        """
        try:
            from scapy.all import conf
            result = []
            for pcap_name, iface in conf.ifaces.items():
                name = getattr(iface, "name", pcap_name)
                desc = getattr(iface, "description", "")
                ip = getattr(iface, "ip", "")

                # Check if interface has a real routable IP (not loopback, link-local, or empty)
                is_active_internet = bool(
                    ip
                    and not ip.startswith("127.")
                    and not ip.startswith("169.254")
                    and ip != "0.0.0.0"
                )

                if is_active_internet:
                    label = f"[ACTIVE INTERNET] {name} ({ip}) - {desc}"
                    prio = 0
                elif ip and not ip.startswith("127."):
                    label = f"[LOCAL/VIRTUAL] {name} ({ip}) - {desc}"
                    prio = 1
                elif ip.startswith("127."):
                    label = f"[LOOPBACK] {name} ({ip})"
                    prio = 3
                else:
                    label = f"[INACTIVE - NO IP] {name}"
                    prio = 2

                result.append((label, pcap_name, prio))

            # Sort so active internet adapters appear at the very top (index 0)
            result.sort(key=lambda x: x[2])
            return [(r[0], r[1]) for r in result]
        except Exception:
            try:
                from scapy.all import get_if_list
                return [(iface, iface) for iface in get_if_list()]
            except Exception:
                return []

    def capture(self, duration_sec: int = 30) -> List[Any]:
        """
        Blocking live capture for `duration_sec` seconds.
        Returns list of captured Scapy packets.
        Uses sniff(timeout=...) — the most reliable approach on Windows Npcap.
        promisc=False is used on Wi-Fi adapters to prevent NDIS driver miniport resets.
        """
        try:
            from scapy.all import sniff, conf
            conf.use_pcap = True
        except ImportError:
            raise ImportError("Scapy is not installed. Run: pip install scapy")

        import time
        last_err = None
        for attempt in range(3):
            try:
                packets = sniff(
                    iface=self.interface,
                    timeout=duration_sec,
                    store=True,
                    promisc=False,
                    prn=None
                )
                return list(packets)
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        if last_err:
            raise last_err
        return []

    def continuous_window_generator(
        self,
        window_sec: float = 2.0,
        max_windows: Optional[int] = None
    ) -> Generator[List[Any], None, None]:
        """
        Continuously captures live packets in rolling `window_sec` windows.
        Each yield returns the raw Scapy packet list for that window.
        Uses promisc=False and automatic retry to avoid Realtek Wi-Fi driver resets
        (error 2150891563: Network interface was not found).
        """
        try:
            from scapy.all import sniff, conf
            conf.use_pcap = True
        except ImportError:
            raise ImportError("Scapy is not installed.")

        import time
        window_count = 0
        while True:
            if max_windows is not None and window_count >= max_windows:
                break

            packets = []
            for attempt in range(3):
                try:
                    packets = sniff(
                        iface=self.interface,
                        timeout=window_sec,
                        store=True,
                        promisc=False,
                        prn=None
                    )
                    break
                except Exception as e:
                    time.sleep(0.3)
                    packets = []

            yield list(packets)
            window_count += 1
