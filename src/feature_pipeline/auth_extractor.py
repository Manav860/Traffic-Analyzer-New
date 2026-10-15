"""
Authentication & Host Telemetry Feature Extractor.
Parses authentication logs (Windows Security Events 4624/4625/4672, Linux auth.log, Kerberos TGT)
and extracts temporal host-level indicators of compromise (failed login bursts, credential stuffing, privilege escalation).
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


AUTH_FEATURE_NAMES = [
    "auth_attempts_rate",
    "failed_auth_ratio",
    "privilege_escalation_count",
    "unique_users_count",
    "unique_hosts_count",
    "admin_account_ratio",
    "lateral_auth_entropy"
]


class AuthFeatureExtractor:
    """
    Extracts security and authentication indicators from structured or tabular event logs
    over sliding temporal observation windows (default 5.0 seconds).
    """

    def __init__(self, window_size_sec: float = 5.0):
        self.window_size_sec = window_size_sec

    def extract_windowed_features(self, df_auth: pd.DataFrame) -> pd.DataFrame:
        """
        Processes raw authentication log records and computes windowed host-security features.
        Expected columns: [timestamp, event_id/action, username, src_host, dst_host, status, is_admin]
        """
        if df_auth is None or len(df_auth) == 0:
            return pd.DataFrame(columns=AUTH_FEATURE_NAMES + ["timestamp_window"])

        df = df_auth.copy()
        time_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)

        min_time = df[time_col].min()
        df["window_idx"] = ((df[time_col] - min_time) // self.window_size_sec).astype(int)

        window_records = []
        for win_idx, win_group in df.groupby("window_idx"):
            n_events = len(win_group)
            
            # Failed authentication ratio
            if "status" in win_group.columns:
                failed_cnt = (win_group["status"].str.lower().isin(["failed", "failure", "error", "rejected"])).sum()
            elif "event_id" in win_group.columns:
                failed_cnt = (win_group["event_id"].isin([4625, 4771, "4625", "4771"])).sum()
            else:
                failed_cnt = 0
            failed_ratio = failed_cnt / max(1, n_events)

            # Privilege escalation / special privileges
            if "event_id" in win_group.columns:
                priv_cnt = (win_group["event_id"].isin([4672, "4672", "sudo"])).sum()
            elif "action" in win_group.columns:
                priv_cnt = (win_group["action"].str.lower().str.contains("privilege|sudo|root|admin")).sum()
            else:
                priv_cnt = 0

            # Unique users and hosts
            u_col = "username" if "username" in win_group.columns else "user"
            unique_users = win_group[u_col].nunique() if u_col in win_group.columns else 1

            h_col = "dst_host" if "dst_host" in win_group.columns else "host"
            unique_hosts = win_group[h_col].nunique() if h_col in win_group.columns else 1

            # Admin account ratio
            if "is_admin" in win_group.columns:
                admin_ratio = (win_group["is_admin"].astype(bool)).sum() / max(1, n_events)
            elif u_col in win_group.columns:
                admin_cnt = win_group[u_col].str.lower().str.contains("admin|root|system|service").sum()
                admin_ratio = admin_cnt / max(1, n_events)
            else:
                admin_ratio = 0.0

            # Lateral authentication entropy across target hosts
            if h_col in win_group.columns and n_events > 1:
                host_counts = win_group[h_col].value_counts().values
                probs = host_counts / n_events
                lateral_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            else:
                lateral_entropy = 0.0

            window_records.append({
                "window_idx": win_idx,
                "timestamp_window": min_time + (win_idx * self.window_size_sec),
                "auth_attempts_rate": float(n_events / self.window_size_sec),
                "failed_auth_ratio": float(failed_ratio),
                "privilege_escalation_count": float(priv_cnt),
                "unique_users_count": float(unique_users),
                "unique_hosts_count": float(unique_hosts),
                "admin_account_ratio": float(admin_ratio),
                "lateral_auth_entropy": float(lateral_entropy)
            })

        return pd.DataFrame(window_records)
