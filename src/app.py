"""
AI-Based Network Attack Forecasting Dashboard (SIH26153).
Offline Decision Support Interface for SOC Analysts.
Runs 100% locally with zero cloud API dependency.
"""

import os
import sys
import time
import io
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seeding import set_seed
from src.utils.synthetic_generator import generate_synthetic_flows, generate_synthetic_pcap
from src.feature_pipeline.flow_extractor import FlowFeatureExtractor
from src.feature_pipeline.packet_extractor import PacketFeatureExtractor, SCAPY_AVAILABLE
import importlib
import src.feature_pipeline.fusion as _fusion_module
importlib.reload(_fusion_module)
DualLevelFusionPipeline = _fusion_module.DualLevelFusionPipeline
STATE_FEATURE_NAMES = _fusion_module.STATE_FEATURE_NAMES
from src.inference.rollout import RolloutEngine
import src.inference.streaming_engine as _streaming_module
importlib.reload(_streaming_module)
StreamingInferenceEngine = _streaming_module.StreamingInferenceEngine
LiveCaptureSniffer = _streaming_module.LiveCaptureSniffer
from src.mitre_mapping.stix_mapper import MitreMapper
from src.explainability.attention import AttentionExplainer
from src.explainability.shap_explainer import FeatureExplainer
from src.model.baseline import LogisticRegressionBaseline

# Streamlit Page Config
st.set_page_config(
    page_title="AI Network Attack Forecaster | SIH26153 (NTRO)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Presentation-focused CSS for the SIH SOC command center
st.markdown("""
<style>
    :root { --ink:#07111f; --panel:#0d1a2b; --line:#203b59; --cyan:#35d7ff; --blue:#4e8cff; --mint:#36e0a0; --amber:#ffbf5b; --red:#ff6275; }
    .stApp { background: radial-gradient(circle at 82% -8%, #153b61 0, transparent 28%), radial-gradient(circle at 12% 5%, #102a48 0, transparent 23%), #07111f; }
    header[data-testid="stHeader"], [data-testid="stHeader"] { background: transparent !important; border: 0 !important; box-shadow: none !important; }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { background: transparent !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1c30, #081321); border-right: 1px solid #1c3854; }
    [data-testid="stSidebar"] h1 { font-size: 1.45rem; }
    .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1500px; }
    .main-header {
        font-size: 2.55rem;
        font-weight: 800;
        letter-spacing: -0.045em;
        color: #edf8ff;
        margin-bottom: 0.22rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #a8c4dd;
        margin-bottom: 1.35rem;
    }
    .hero-shell { border: 1px solid #28557a; background: linear-gradient(105deg, rgba(14,48,77,.92), rgba(12,25,44,.92) 58%, rgba(15,81,93,.76)); border-radius: 18px; padding: 26px 30px; margin: 0 0 22px 0; box-shadow: 0 18px 42px rgba(0,0,0,.22); }
    .hero-shell { position:relative; overflow:hidden; border: 1px solid #2d7297; background: linear-gradient(112deg, rgba(20,50,110,.95), rgba(27,45,104,.94) 45%, rgba(7,113,110,.90)); border-radius: 18px; padding: 26px 30px; margin: 0 0 22px 0; box-shadow: 0 18px 42px rgba(0,0,0,.22); }
    .hero-shell:after { content:""; position:absolute; width:280px; height:280px; border-radius:50%; top:-175px; right:-55px; background:radial-gradient(circle, rgba(80,235,255,.42), transparent 67%); pointer-events:none; }
    .eyebrow { color:#73e4ff; text-transform:uppercase; letter-spacing:.14em; font-size:.72rem; font-weight:800; }
    .hero-title { color:#fff; font-size:1.5rem; font-weight:750; margin:.45rem 0; }
    .hero-copy { color:#b9d3e7; max-width:860px; line-height:1.55; font-size:.93rem; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
    .hero-pill { color:#cdeeff; background:rgba(63,155,214,.14); border:1px solid #2b668e; border-radius:99px; padding:5px 10px; font-size:.74rem; font-weight:700; }
    .impact-card { background:rgba(12,30,48,.88); border:1px solid #214562; border-radius:12px; padding:14px 15px; min-height:105px; }
    .impact-number { color:#75e5ff; font-size:1.28rem; font-weight:800; margin-top:3px; }
    .impact-card { background:rgba(12,30,48,.88); border:1px solid #214562; border-radius:12px; padding:14px 15px; min-height:105px; box-shadow:0 9px 22px rgba(0,0,0,.15); transition:transform .18s ease, box-shadow .18s ease; }
    .impact-card:hover { transform:translateY(-3px); box-shadow:0 14px 28px rgba(0,0,0,.28); }
    .impact-number { color:#75e5ff; font-size:1.38rem; font-weight:800; margin-top:3px; }
    .impact-label { color:#8faec6; font-size:.74rem; text-transform:uppercase; letter-spacing:.07em; font-weight:700; }
    .briefing { background:linear-gradient(90deg, rgba(41,99,127,.22), rgba(20,39,61,.25)); border:1px solid #2a5875; border-radius:12px; padding:17px 19px; margin:12px 0 18px; }
    .briefing h4 { color:#effaff; margin:0 0 6px; font-size:1rem; }
    .briefing p { color:#b7d0e3; margin:0; line-height:1.5; font-size:.9rem; }
    [data-testid="stMetric"] { background:rgba(13,30,48,.86); border:1px solid #21445f; border-radius:11px; padding:14px 15px; }
    [data-testid="stMetricLabel"] { color:#9fbed5; font-size:.78rem; text-transform:uppercase; letter-spacing:.04em; }
    [data-testid="stMetricValue"] { color:#edf8ff; }
    [data-testid="stExpander"] { border:1px solid #23435f; border-radius:10px; background:rgba(9,21,36,.58); }
    .stButton > button { border-radius:9px; border:1px solid #3476a1; font-weight:700; }
    .stButton > button[kind="primary"] { background:linear-gradient(100deg, #2768e6, #00a9b7); border-color:#58d5ed; color:#fff; box-shadow:0 8px 18px rgba(20,120,220,.2); }
    .metric-container {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }
    .alert-critical {
        background-color: #3b1111;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        color: #FCA5A5;
        border: 1px solid #7F1D1D;
        margin-bottom: 1rem;
    }
    .alert-warning {
        background-color: #362007;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        color: #FDE68A;
        border: 1px solid #78350F;
        margin-bottom: 1rem;
    }
    .alert-safe {
        background-color: #06281e;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        color: #6EE7B7;
        border: 1px solid #065F46;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def render_command_center_hero() -> None:
    """Compact, judge-facing framing for the offline SOC demonstration."""
    st.markdown("""
    <div class="hero-shell">
      <div class="eyebrow">SIH 2026 · NTRO · Offline Decision Intelligence</div>
      <div class="hero-title">Forecast the next move. Contain the attack before impact.</div>
      <div class="hero-copy">A temporal world model fuses NetFlow and packet telemetry into a 28-dimensional network state, then performs autoregressive K-step rollout to anticipate MITRE ATTACK kill-chain progression.</div>
      <div class="hero-pills"><span class="hero-pill">AIR-GAPPED READY</span><span class="hero-pill">28-D STATE FUSION</span><span class="hero-pill">LSTM + TEMPORAL ATTENTION</span><span class="hero-pill">MITRE ATT&amp;CK MAPPING</span><span class="hero-pill">EXPLAINABLE AI</span></div>
    </div>
    """, unsafe_allow_html=True)
    cards = st.columns(4)
    content = [
        ("28", "Telemetry features fused", "#33d5ff", "rgba(16, 91, 133, .52)"),
        ("5", "Attack stages forecast", "#b99cff", "rgba(73, 50, 135, .48)"),
        ("K × 2 s", "Forward decision horizon", "#ffca70", "rgba(122, 76, 29, .48)"),
        ("0", "Cloud telemetry egress", "#51e5ad", "rgba(17, 104, 76, .46)"),
    ]
    for col, (number, label, accent, background) in zip(cards, content):
        with col:
            st.markdown(f'<div class="impact-card" style="border-color:{accent}66; background:linear-gradient(135deg, {background}, rgba(10,25,42,.94));"><div class="impact-label">{label}</div><div class="impact-number" style="color:{accent};">{number}</div></div>', unsafe_allow_html=True)


@st.cache_resource
def load_cached_engine() -> Tuple[RolloutEngine, MitreMapper, FeatureExplainer]:
    """Initialize models, mapper, and explainers with cached weights."""
    set_seed(42)
    config_path = PROJECT_ROOT / "configs" / "lstm_attention.yaml"
    model_ckpt = PROJECT_ROOT / "model" / "weights" / "world_model_final.pt"
    scaler_path = PROJECT_ROOT / "model" / "weights" / "scaler.pkl"
    baseline_path = PROJECT_ROOT / "model" / "weights" / "baseline_final.pkl"

    # Auto-train if weights don't exist yet
    if not model_ckpt.exists():
        from src.training.train import train_world_model
        with st.spinner("Initializing World Model & Pretraining weights..."):
            train_world_model(config_path=str(config_path))

    engine = RolloutEngine(
        model_checkpoint=str(model_ckpt),
        scaler_path=str(scaler_path),
        config_path=str(config_path)
    )
    mapper = MitreMapper()

    # Load Baseline for SHAP
    baseline = LogisticRegressionBaseline.load(str(baseline_path)) if baseline_path.exists() else None
    explainer = FeatureExplainer(baseline_model=baseline)
    
    return engine, mapper, explainer


def render_killchain_stepper(predicted_stage: str, stage_index: int, peak_risk: float, alert_threshold: float) -> str:
    """
    Renders a horizontal 5-stage stepper representing the MITRE ATT&CK kill chain.
    Highlights the currently predicted stage with a green -> yellow -> red color progression
    tied to the alert_threshold slider.
    """
    stages = [
        ("Reconnaissance", "TA0043", 1),
        ("Initial Access", "TA0001", 2),
        ("Lateral Movement", "TA0008", 3),
        ("Command & Control", "TA0011", 4),
        ("Exfiltration", "TA0010", 5),
    ]

    # Color progression tied directly to alert_threshold
    if peak_risk >= alert_threshold:
        theme_color = "#EF4444"      # Red (Critical alert)
        glow_color = "rgba(239, 68, 68, 0.45)"
        text_color = "#FCA5A5"
        badge_bg = "rgba(239, 68, 68, 0.18)"
        status_label = "HIGH THREAT ESCALATION"
    elif peak_risk >= alert_threshold * 0.7:
        theme_color = "#F59E0B"      # Yellow/Amber (Elevated risk)
        glow_color = "rgba(245, 158, 11, 0.4)"
        text_color = "#FDE68A"
        badge_bg = "rgba(245, 158, 11, 0.18)"
        status_label = "ELEVATED RISK TRAJECTORY"
    else:
        theme_color = "#10B981"      # Green (Nominal baseline)
        glow_color = "rgba(16, 185, 129, 0.35)"
        text_color = "#6EE7B7"
        badge_bg = "rgba(16, 185, 129, 0.18)"
        status_label = "NOMINAL PATTERN"

    stepper_html = f"""
    <div style="background: #111827; border: 1px solid #1F2937; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.35);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8;">
                🛡️ MITRE ATT&CK Kill-Chain Progression Stepper
            </div>
            <div style="background: {badge_bg}; color: {text_color}; border: 1px solid {theme_color}; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">
                {status_label} — Target Stage: {predicted_stage}
            </div>
        </div>
        <div style="display: flex; align-items: center; justify-content: space-between; position: relative;">
    """

    for i, (name, tactic_id, idx) in enumerate(stages):
        is_active = (stage_index == idx) or (name.lower() in predicted_stage.lower()) or (predicted_stage.lower() in name.lower() and predicted_stage.lower() != "benign")
        is_past = (stage_index > idx) and (stage_index > 0)

        if is_active:
            circle_style = f"background: {theme_color}; color: #ffffff; border: 2px solid {theme_color}; box-shadow: 0 0 14px {glow_color}; font-weight: 800;"
            title_style = f"color: {theme_color}; font-weight: 700;"
        elif is_past:
            circle_style = f"background: #1E293B; color: #94A3B8; border: 2px solid #334155;"
            title_style = "color: #94A3B8; font-weight: 600;"
        else:
            circle_style = "background: #0B0F19; color: #64748B; border: 2px solid #1F2937;"
            title_style = "color: #64748B; font-weight: 400;"

        stepper_html += f"""
        <div style="display: flex; flex-direction: column; align-items: center; text-align: center; flex: 1; position: relative; z-index: 2;">
            <div style="width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.82rem; {circle_style} margin-bottom: 5px;">
                {idx}
            </div>
            <div style="font-size: 0.78rem; line-height: 1.15; {title_style}">{name}</div>
            <div style="font-size: 0.68rem; color: #64748B; margin-top: 1px;">{tactic_id}</div>
        </div>
        """

        if i < len(stages) - 1:
            line_color = theme_color if is_past or (is_active and stage_index > idx) else "#1F2937"
            stepper_html += f"""
            <div style="flex: 1; height: 3px; background: {line_color}; margin: 0 -6px 20px -6px; z-index: 1;"></div>
            """

    stepper_html += """
        </div>
    </div>
    """
    return "\n".join(line.strip() for line in stepper_html.splitlines())


def build_rollout_plotly_figure(timeline_df: pd.DataFrame, alert_threshold: float, peak_risk: float) -> go.Figure:
    """Creates a high-fidelity Plotly line chart for the forward rollout simulation."""
    fig = go.Figure()

    if peak_risk >= alert_threshold:
        line_color = "#EF4444"
        fill_color = "rgba(239, 68, 68, 0.10)"
    elif peak_risk >= alert_threshold * 0.7:
        line_color = "#F59E0B"
        fill_color = "rgba(245, 158, 11, 0.10)"
    else:
        line_color = "#10B981"
        fill_color = "rgba(16, 185, 129, 0.08)"

    x_labels = [f"+{row['relative_time_sec']}s (t+{row['step']})" for _, row in timeline_df.iterrows()]
    y_vals = timeline_df["infiltration_probability"] * 100
    stages = timeline_df["predicted_stage"].tolist()

    fig.add_trace(go.Scatter(
        x=x_labels,
        y=y_vals,
        mode="lines+markers+text",
        name="Infiltration Risk (%)",
        text=[f"{s}<br>{v:.0f}%" for s, v in zip(stages, y_vals)],
        textposition="top center",
        line=dict(color=line_color, width=3, shape="spline", smoothing=1.1),
        marker=dict(size=9, symbol="diamond", color=line_color, line=dict(width=1.5, color="#ffffff")),
        fill="tozeroy",
        fillcolor=fill_color,
        customdata=stages,
        hovertemplate="<b>Horizon %{x}</b><br>Predicted Stage: <b>%{customdata}</b><br>Infiltration Risk: <b>%{y:.1f}%</b><extra></extra>"
    ))

    fig.add_hline(
        y=alert_threshold * 100,
        line_dash="dash",
        line_color="#F59E0B",
        line_width=2,
        annotation_text=f"Alert Threshold ({alert_threshold*100:.0f}%)",
        annotation_position="bottom right",
        annotation_font=dict(color="#FDE68A", size=10)
    )

    fig.update_layout(
        title=dict(text="📈 Multi-Step Forward Infiltration Risk Trajectory", font=dict(size=14, color="#F1F5F9")),
        yaxis=dict(title="Infiltration Risk Score (%)", range=[0, 115], gridcolor="#1F2937", zeroline=False, color="#E2E8F0"),
        xaxis=dict(title="Forward Rollout Horizon", gridcolor="#1F2937", color="#E2E8F0"),
        height=340,
        margin=dict(l=15, r=15, t=35, b=15),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        hovermode="x unified",
        showlegend=False
    )
    return fig


def render_terminal_console(log_entries: List[str]) -> str:
    """Renders a styled dark SOC terminal/console log panel."""
    formatted_lines = ""
    for entry in log_entries[-18:]:
        if "CRITICAL" in entry or "ALERT" in entry or "Exfiltration" in entry or "Command & Control" in entry:
            color = "#ff7b72"
        elif "Lateral" in entry or "Initial Access" in entry or "ELEVATED" in entry:
            color = "#d29922"
        elif "Benign" in entry or "Nominal" in entry:
            color = "#3fb950"
        else:
            color = "#58a6ff"
        formatted_lines += f"<div style='color: {color}; margin-bottom: 3px; font-family: monospace;'>{entry}</div>"

    if not formatted_lines:
        formatted_lines = "<div style='color: #8b949e; font-family: monospace;'>[System Initialized] Awaiting network traffic observation windows...</div>"

    console_html = f"""
    <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; font-family: 'Consolas', 'Courier New', monospace; font-size: 0.80rem; height: 340px; overflow-y: auto; color: #c9d1d9; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);">
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-bottom: 8px; color: #8b949e; font-size: 0.72rem; text-transform: uppercase;">
            <span>● SOC TELEMETRY & EVENT LOG</span>
            <span>LIVE BUFFER</span>
        </div>
        {formatted_lines}
    </div>
    """
    return "\n".join(line.strip() for line in console_html.splitlines())


def generate_incident_report_csv(
    session_metadata: Dict[str, Any],
    forecast: Dict[str, Any],
    shap_top_drivers: List[Dict[str, Any]],
    current_window: pd.Series
) -> str:
    """Generates a structured CSV incident report for forensic handoff."""
    output = io.StringIO()
    output.write("# AI-BASED NETWORK ATTACK FORECASTING - SOC INCIDENT BRIEFING REPORT\n")
    output.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
    output.write(f"# Analysis Window (t): {session_metadata.get('window_id', 0)}\n")
    output.write(f"# Target Predicted Stage: {forecast.get('terminal_stage', 'Unknown')}\n")
    output.write(f"# Peak Infiltration Risk: {forecast.get('peak_risk', 0.0) * 100:.2f}%\n")
    output.write(f"# Alert Threshold: {session_metadata.get('alert_threshold', 0.5) * 100:.2f}%\n")
    output.write("# ----------------------------------------------------------------\n\n")

    output.write("SECTION 1: FORWARD ROLLOUT HORIZON TIMELINE\n")
    output.write("Step,Relative_Time_Sec,Timestamp,Infiltration_Probability_Pct,Predicted_MITRE_Stage\n")
    for item in forecast.get("timeline", []):
        output.write(f"{item['step']},{item['relative_time_sec']},{item['timestamp']},{item['infiltration_probability']*100:.2f},{item['predicted_stage']}\n")

    output.write("\nSECTION 2: TOP SHAP FEATURE ATTRIBUTION DRIVERS\n")
    output.write("Feature_Name,Observed_Value,SHAP_Importance,Direction\n")
    for d in shap_top_drivers:
        output.write(f"{d['feature']},{d['value']:.4f},{d['importance']:.4f},{d['direction']}\n")

    output.write("\nSECTION 3: RECOMMENDED SOC MITIGATION ACTIONS\n")
    terminal_stage = forecast.get('terminal_stage', 'Benign')
    if "Recon" in terminal_stage:
        output.write("Action 1: Rate-limit external TCP SYN sweeps and port scans on firewall ingress.\n")
        output.write("Action 2: Blacklist aggressive source IP addresses on perimeter edge.\n")
    elif "Initial" in terminal_stage:
        output.write("Action 1: Enforce mandatory MFA on SSH (port 22) and administrative portals.\n")
        output.write("Action 2: Isolate targeted host and rotate compromised service credentials.\n")
    elif "Lateral" in terminal_stage:
        output.write("Action 1: Sever inter-VLAN routing towards internal pivot targets.\n")
        output.write("Action 2: Revoke active Kerberos and SSH tokens across internal subnets.\n")
    elif "Command" in terminal_stage or "Exfil" in terminal_stage:
        output.write("Action 1: Sinkhole external C2 IP/domain on corporate recursive DNS.\n")
        output.write("Action 2: Terminate active outbound proxy data tunnels immediately.\n")
    else:
        output.write("Action: Maintain nominal telemetry monitoring. No defensive intervention required.\n")

    return output.getvalue()


def render_soc_view(
    current_window: pd.Series,
    history_df: pd.DataFrame,
    forecast: Dict[str, Any],
    mapper: MitreMapper,
    explainer: FeatureExplainer,
    horizon_k: int,
    alert_threshold: float,
    current_idx: int,
    total_windows: int,
    prev_prob: Optional[float] = None,
    event_logs: Optional[List[str]] = None,
    full_trace_df: Optional[pd.DataFrame] = None
):
    """Renders the SOC metrics, 5-stage stepper, rollout trajectory, live console, explainability overlay, and MITRE panels."""
    if event_logs is None:
        event_logs = st.session_state.get("event_logs", [])

    current_label = current_window.get("label", "Benign")
    current_tactic_info = mapper.get_tactic_for_label(str(current_label))
    peak_risk = forecast["peak_risk"]
    terminal_stage = forecast["terminal_stage"]
    is_escalating = forecast["is_escalating"]

    # Calculate current infiltration probability & delta
    curr_prob = float(forecast["timeline"][0]["infiltration_probability"]) if forecast.get("timeline") else peak_risk
    if prev_prob is not None:
        delta_val = (curr_prob - prev_prob) * 100
        delta_str = f"{delta_val:+.1f}% vs prev"
    else:
        delta_str = "Baseline"

    # Threshold-crossing moment detection (fires toast once per upward crossing)
    if "is_in_alert" not in st.session_state:
        st.session_state["is_in_alert"] = False
    if "alerts_count" not in st.session_state:
        st.session_state["alerts_count"] = 0

    if peak_risk >= alert_threshold:
        if not st.session_state["is_in_alert"]:
            st.session_state["alerts_count"] += 1
            st.session_state["is_in_alert"] = True
            st.toast(f"🚨 ALERT: Infiltration risk crossed threshold ({peak_risk*100:.1f}%) — Projected {terminal_stage}!", icon="⚠️")
    else:
        st.session_state["is_in_alert"] = False

    # Top KPI Cards (st.columns(4))
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric(
            label="Current Infiltration Risk",
            value=f"{curr_prob * 100:.1f}%",
            delta=delta_str,
            delta_color="inverse"
        )
    with col_stat2:
        st.metric(
            label="Forecasted Terminal Stage",
            value=f"{terminal_stage}",
            delta=f"+{horizon_k * 2}s ahead"
        )
    with col_stat3:
        st.metric(
            label="Observation Window (t)",
            value=f"{current_idx} / {total_windows}",
            delta=f"Elapsed: +{current_idx * 2.0:.0f}s"
        )
    with col_stat4:
        st.metric(
            label="Alerts Escalated Session",
            value=f"{st.session_state.get('alerts_count', 0)} Alerts",
            delta="Active Monitor" if peak_risk >= alert_threshold else "Nominal",
            delta_color="inverse" if peak_risk >= alert_threshold else "normal"
        )

    # 1. Horizontal 5-Stage MITRE Kill-Chain Stepper
    terminal_stage_idx = forecast.get("terminal_stage_idx", forecast["timeline"][-1].get("stage_index", 0) if forecast.get("timeline") else 0)
    stepper_html = render_killchain_stepper(
        predicted_stage=terminal_stage,
        stage_index=terminal_stage_idx,
        peak_risk=peak_risk,
        alert_threshold=alert_threshold
    )
    st.markdown(stepper_html, unsafe_allow_html=True)

    # Threat Status Alert Banner
    if peak_risk >= alert_threshold:
        st.markdown(f"""
        <div class="alert-critical">
            <strong>🚨 ADVANCE THREAT WARNING (Risk: {peak_risk*100:.1f}%):</strong> High probability of multi-stage attack trajectory converging to 
            <strong>{terminal_stage}</strong> within the next {horizon_k * 2} seconds.
            <strong>Recommended Defender Action:</strong> Stage host isolation, rotate compromised credentials, rate-limit ingress probes.
        </div>
        """, unsafe_allow_html=True)
    elif peak_risk >= alert_threshold * 0.7:
        st.markdown(f"""
        <div class="alert-warning">
            <strong>⚠️ ELEVATED RISK TRAJECTORY (Risk: {peak_risk*100:.1f}%):</strong> Anomalous dynamics detected. Anticipated progression to 
            <strong>{terminal_stage}</strong> (+{horizon_k * 2}s). Heighten telemetry surveillance.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-safe">
            <strong>✅ NOMINAL OPERATIONS:</strong> Traffic dynamics remain consistent with benign baseline. Forward rollout shows zero kill-chain convergence.
        </div>
        """, unsafe_allow_html=True)

    # 2 & 6: Main Visualizations: Animated Plotly Trajectory Chart & Live Event Console Panel
    col_chart1, col_chart2 = st.columns([6, 4])

    with col_chart1:
        timeline_df = pd.DataFrame(forecast["timeline"])
        fig_traj = build_rollout_plotly_figure(timeline_df, alert_threshold, peak_risk)
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_chart2:
        st.subheader("🖥️ SOC Live Event Log")
        console_html = render_terminal_console(event_logs)
        st.markdown(console_html, unsafe_allow_html=True)

    # 3. Explainability Overlay on Data (Features shaded by SHAP attribution magnitude)
    st.markdown("---")
    curr_feat_vector = current_window[STATE_FEATURE_NAMES].values
    shap_res = explainer.explain_instance(curr_feat_vector, top_k=8)
    top_drivers = shap_res["top_drivers"]

    col_exp1, col_exp2 = st.columns([6, 4])

    with col_exp1:
        st.subheader("🔍 Dual-Level State Matrix & SHAP Attribution Overlay")
        st.caption("Color-coded feature table directly connecting observed network telemetry with attribution weights:")

        # Construct enriched dataframe
        all_attributions = {item["feature"]: item["importance"] for item in shap_res["all_attributions"]}
        table_rows = []
        disp_cols = [
            "flow_rate_per_sec", "byte_rate_per_sec", "syn_ratio", "rst_ratio",
            "port_entropy", "ttl_var", "tcp_win_mean", "payload_entropy_mean", "port_seq_randomness", "small_pkt_ratio"
        ]
        for col in disp_cols:
            val = float(current_window.get(col, 0.0))
            shap_val = float(all_attributions.get(col, 0.0))
            stream_lvl = "Packet-Level (PCAP)" if col in ["ttl_var", "tcp_win_mean", "payload_entropy_mean", "port_seq_randomness", "small_pkt_ratio"] else "Flow-Level (NetFlow)"
            impact = "🚨 Risk Escalator (+)" if shap_val > 0.01 else ("🛡️ Mitigating (-)" if shap_val < -0.01 else "Neutral (0)")
            table_rows.append({
                "Feature Attribute": col,
                "Observed Value (t)": f"{val:.4f}",
                "Telemetry Layer": stream_lvl,
                "SHAP Weight": shap_val,
                "Threat Impact Direction": impact
            })

        df_table = pd.DataFrame(table_rows)

        # Style with background gradient on SHAP Weight (Dark Theme)
        def highlight_threat(row):
            w = row["SHAP Weight"]
            if w > 0.05:
                return ["background-color: #3b1111; color: #fca5a5; font-weight: 600;"] * len(row)
            elif w > 0.01:
                return ["background-color: #362007; color: #fde68a;"] * len(row)
            elif w < -0.01:
                return ["background-color: #0c2340; color: #93c5fd;"] * len(row)
            else:
                return ["background-color: #111827; color: #9ca3af;"] * len(row)

        styled_df = df_table.style.apply(highlight_threat, axis=1).format({"SHAP Weight": "{:+.4f}"})
        st.dataframe(styled_df, use_container_width=True, height=280)

    with col_exp2:
        st.subheader("💡 Top Threat Drivers (SHAP)")
        st.caption("Key traffic attributes pushing threat probability upwards / downwards:")
        top_drivers_df = pd.DataFrame(top_drivers)

        fig_shap = px.bar(
            top_drivers_df,
            x="importance",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"Risk Increase (+)": "#EF4444", "Risk Decrease (-)": "#3B82F6"},
            labels={"importance": "SHAP Attribution", "feature": "Feature"}
        )
        fig_shap.update_layout(
            height=280,
            margin=dict(l=15, r=15, t=10, b=15),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(gridcolor="#1F2937"),
            yaxis=dict(gridcolor="#1F2937")
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Temporal Attention Attribution
    st.markdown("---")
    col_att1, col_att2 = st.columns([6, 4])

    with col_att1:
        st.subheader("🧠 Temporal Attention Alignment (\\(\\alpha_i\\))")
        st.caption("Historical sequence weights identifying the specific timeframes where malicious intent was first inferred:")

        raw_attn = forecast["attention_weights_initial"]
        attn_items = AttentionExplainer.format_attention_timeline(
            raw_attn,
            base_timestamp=float(current_window.get("timestamp_start", 0.0)),
            window_size_sec=2.0,
            seq_len=10
        )
        attn_df = pd.DataFrame(attn_items)

        fig_attn = px.bar(
            attn_df,
            x="relative_label",
            y="attention_weight",
            color="attention_weight",
            color_continuous_scale="Viridis",
            labels={"relative_label": "Time Window", "attention_weight": "Attention Weight"}
        )
        fig_attn.update_layout(
            height=260,
            margin=dict(l=15, r=15, t=10, b=15),
            coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E2E8F0"),
            xaxis=dict(gridcolor="#1F2937"),
            yaxis=dict(gridcolor="#1F2937")
        )
        st.plotly_chart(fig_attn, use_container_width=True)

    with col_att2:
        st.subheader("📥 Export SOC Forensic Report & Traffic")
        st.caption("Download forensic incident briefing or raw traffic trace:")
        
        session_meta = {
            "window_id": current_idx,
            "alert_threshold": alert_threshold,
            "horizon_k": horizon_k
        }
        report_content = generate_incident_report_csv(session_meta, forecast, top_drivers, current_window)
        
        st.download_button(
            label="📄 Download Analyst Incident Briefing (.csv)",
            data=report_content,
            file_name=f"SOC_Incident_Report_SIH26153_Window_{current_idx}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Human-readable text report with forecast timeline, top SHAP drivers, and mitigation steps."
        )

        export_df = full_trace_df if (full_trace_df is not None and not full_trace_df.empty) else history_df
        if export_df is not None and not export_df.empty:
            num_wins = len(export_df)
            st.download_button(
                label=f"💾 Download Replayable Traffic Dataset ({num_wins} Windows .csv)",
                data=export_df.to_csv(index=False),
                file_name=f"captured_traffic_trace_session_{num_wins}windows.csv",
                mime="text/csv",
                use_container_width=True,
                help=f"Exports all {num_wins} captured network windows as a tabular CSV that you can replay in Mode A: Offline File Replay or Mode B: Streaming Simulation."
            )

        st.info("💡 **Tip**: The **Incident Briefing** is a human-readable report for ticketing. The **Replayable Traffic Dataset** is a tabular CSV that can be re-uploaded to **Mode A**.")

    # MITRE ATT&CK STIX Mapping Reference
    st.markdown("---")
    with st.expander("📚 MITRE ATT&CK STIX 2.1 Taxonomy & Kill-Chain Mapping Reference"):
        st.markdown("""
        **System Grounding & Ontological Transparency**:
        - Raw datasets (CIC-IDS-2018 / CTU-13) lack native MITRE tactic annotations.
        - Our model resolves dataset attack classes through the canonical `mitre_mapping/label_to_tactic.yaml` mapping specification.
        """)
        
        mappings_data = []
        for lbl, info in mapper.mappings.items():
            mappings_data.append({
                "Dataset Label": lbl,
                "MITRE Tactic": info["tactic_name"],
                "Tactic ID": info["tactic_id"],
                "Technique Name": info["technique_name"],
                "Technique ID": info["technique_id"],
                "Kill-Chain Stage": info["kill_chain_stage"]
            })
        st.dataframe(pd.DataFrame(mappings_data), use_container_width=True)


def main():
    st.markdown('<div class="main-header">🛡️ AI-Based Network Attack Forecasting</div>', unsafe_allow_html=True)
    st.markdown(r'<div class="sub-header">Transition Dynamics World Model \(P(S_{t+1} | S_t)\) — Multi-Step Kill-Chain Horizon Simulation (SIH26153 - NTRO)</div>', unsafe_allow_html=True)

    render_command_center_hero()
    engine, mapper, explainer = load_cached_engine()

    # Sidebar Navigation & Settings organized in clean Expanders (Item 4)
    st.sidebar.title("🛡️ SOC Controller Console")

    with st.sidebar.expander("📂 Telemetry Ingestion & Mode", expanded=True):
        mode = st.radio(
            "Select Ingestion Mode:",
            [
                "Mode A: Offline File Replay",
                "Mode B: Streaming Simulation",
                "Mode C: Live Socket Capture"
            ],
            index=0
        )

        sample_btn = st.button("🚀 Load Multi-Stage Trace", use_container_width=True)

    with st.sidebar.expander("⚙️ Forecasting Parameters", expanded=True):
        horizon_k = st.slider("Rollout Horizon K (Steps)", min_value=3, max_value=10, value=5)
        alert_threshold = st.slider("Infiltration Alert Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

    with st.sidebar.expander("🔒 Deployment & CII Safeguards", expanded=False):
        st.info("💡 **100% Offline Air-Gapped**: Zero cloud API dependencies, zero external network telemetry egress. Safe for Critical Information Infrastructure.")

    # Data Ingestion based on Mode
    fused_df = None

    if "Mode A" in mode:
        st.subheader("📁 Mode A: Offline Dataset / PCAP File Ingestion")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            uploaded_csv = st.file_uploader("Upload NetFlow / CIC-IDS Flow CSV", type=["csv"])
        with col_up2:
            uploaded_pcap = st.file_uploader("Upload Raw PCAP Packet Capture (Optional)", type=["pcap", "pcapng"])

        if uploaded_csv is not None or uploaded_pcap is not None:
            with st.spinner("Parsing and fusing dual-level traffic features..."):
                flow_df = None
                pkt_df = None
                if uploaded_csv is not None:
                    try:
                        # Check if the uploaded file is an exported incident briefing report
                        file_bytes = uploaded_csv.read(500)
                        uploaded_csv.seek(0)
                        if b"INCIDENT BRIEFING REPORT" in file_bytes or b"SECTION 1:" in file_bytes or b"AI-BASED NETWORK ATTACK" in file_bytes:
                            st.warning(
                                "⚠️ **Incident Briefing Report Uploaded:**  \n"
                                "You uploaded an exported **SOC Incident Briefing Report** (`SOC_Incident_Report_*.csv`). "
                                "This file is a human-readable forensic report with text sections, not a tabular network traffic stream.  \n\n"
                                "👉 **How to replay captured traffic:**  \n"
                                "• In **Mode C: Live Socket Capture**, click **'💾 Download Replayable Traffic Dataset (.csv)'**  \n"
                                "• Upload that dataset (`captured_traffic_trace_*.csv`) here to replay your attack!"
                            )
                            return

                        flow_raw = pd.read_csv(uploaded_csv)

                        # Check if this file is already an extracted/fused traffic dataset
                        state_cols = ["syn_ratio", "port_entropy", "packet_rate_per_sec"]
                        if all(c in flow_raw.columns for c in state_cols):
                            flow_df = flow_raw
                        else:
                            flow_ext = FlowFeatureExtractor(window_size_sec=2.0)
                            flow_df = flow_ext.extract_windowed_features(flow_raw)

                    except pd.errors.ParserError as pe:
                        st.error(
                            f"❌ **CSV Format Error:** Unable to parse file as a tabular network dataset.  \n"
                            f"`{pe}`  \n"
                            f"Ensure the CSV contains tabular NetFlow/CIC-IDS flow records with consistent columns."
                        )
                        return
                    except Exception as ex:
                        st.error(f"❌ Failed to parse CSV: {ex}")
                        return

                if uploaded_pcap is not None:
                    try:
                        temp_pcap = PROJECT_ROOT / "data" / "raw" / "temp_upload.pcap"
                        temp_pcap.parent.mkdir(parents=True, exist_ok=True)
                        with open(temp_pcap, "wb") as f:
                            f.write(uploaded_pcap.getbuffer())
                        pkt_ext = PacketFeatureExtractor(window_size_sec=2.0)
                        pkt_df = pkt_ext.extract_from_pcap(str(temp_pcap))
                    except Exception as pcap_ex:
                        st.error(f"❌ Failed to parse PCAP: {pcap_ex}")
                        return

                fusion = DualLevelFusionPipeline()
                fused_df = fusion.fuse_dataframes(df_flow=flow_df, df_packet=pkt_df)
                st.success(f"Successfully fused {len(fused_df)} time windows of network traffic!")

        elif sample_btn or "sample_loaded" in st.session_state:
            st.session_state["sample_loaded"] = True
            demo_path = PROJECT_ROOT / "data" / "processed" / "cicids2017_demo_trace.csv"
            full_path = PROJECT_ROOT / "data" / "processed" / "fused_state.csv"

            dataset_choice = st.radio(
                "Telemetry Dataset Selection (Real Canadian Institute for Cybersecurity Traffic):",
                [
                    "🎯 Real CIC-IDS-2017 Multi-Stage Kill-Chain (71 Windows)",
                    "🌐 Full CIC-IDS-2017 Entire Dataset (12,433 Windows)"
                ],
                index=0,
                horizontal=True
            )
            if "Multi-Stage" in dataset_choice and demo_path.exists() or not full_path.exists():
                fused_df = pd.read_csv(demo_path)
                st.success("✅ Loaded **Real CIC-IDS-2017 Kill-Chain Sequence** (71 observation windows: Reconnaissance → Initial Access → Lateral Movement → C2 → Impact)")
            else:
                fused_df = pd.read_csv(full_path)
                st.success(f"✅ Loaded **Full CIC-IDS-2017 Dataset** ({len(fused_df):,} observation windows across all 8 days)")

    elif "Mode B" in mode:
        st.subheader("⏱️ Mode B: Streaming Simulation Mode (Real-Time Playback)")
        st.write("Simulates continuous traffic streaming through a sliding temporal window buffer.")
        
        demo_path = PROJECT_ROOT / "data" / "processed" / "cicids2017_demo_trace.csv"
        full_path = PROJECT_ROOT / "data" / "processed" / "fused_state.csv"
        if demo_path.exists():
            fused_df = pd.read_csv(demo_path)
            st.info("📡 Streaming playback source: **Real CIC-IDS-2017 Multi-Stage Kill-Chain** (Reconnaissance → Initial Access → Lateral Movement → C2 → Impact)")
        elif full_path.exists():
            fused_df = pd.read_csv(full_path)
        else:
            raw_flows = generate_synthetic_flows(num_samples=5000, random_seed=42)
            flow_ext = FlowFeatureExtractor(window_size_sec=2.0)
            flow_df = flow_ext.extract_windowed_features(raw_flows)
            fusion = DualLevelFusionPipeline()
            fused_df = fusion.fuse_dataframes(df_flow=flow_df)

    else:
        # ── MODE C: LIVE INTERFACE CAPTURE ──────────────────────────────────────
        st.subheader("⚡ Mode C: Live Network Interface Capture (Administrator Mode)")

        st.warning(
            "⚠️ **Administrator / Root Privileges Required**  \n"
            "Launch Streamlit from an **Administrator Command Prompt** for this mode to work."
        )

        # ── Enterprise deployment explanation ────────────────────────────────────
        st.markdown("""
        <div class="briefing"><h4>Live telemetry, explained in one sentence</h4><p>This mode observes traffic visible to the selected adapter, converts packets into flow and packet-level features, and runs an on-device forward rollout. For an enterprise-wide view, connect a sanctioned SPAN/mirror port or import approved NetFlow/IPFIX records in Mode A.</p></div>
        """, unsafe_allow_html=True)
        with st.expander("Capture scope & deployment notes", expanded=False):
            st.markdown("""
**What this mode captures and why:**

| Setup | What you see |
|:---|:---|
| **This machine only** (default) | All traffic TO and FROM this PC — your own flows, DNS, HTTP, etc. |
| **Promiscuous mode** *(enabled below)* | All packets passing through your network segment — works best on hubs or when connected directly to a router/switch mirror port |
| **Enterprise SPAN/mirror port** *(recommended for production)* | The switch copies ALL traffic from the entire network to your monitoring port — you see every host's flows |

**For the hackathon demo:**
- Run this on your laptop connected to a Wi-Fi router — you will see **your machine's real traffic**
- Promiscuous mode is already enabled — it tries to capture broadcast/multicast and other frames too
- Generate network activity (open websites, ping hosts, run `nmap`) so the model has enough traffic to analyze

**For real enterprise SOC deployment:**
- Configure a **SPAN/mirror port** on the switch → connect the monitoring server to that port
- Or export **NetFlow/IPFIX** from your router → feed the CSV into Mode A
- The World Model then sees **all hosts, all flows** on your network in real time
            """)

        if not SCAPY_AVAILABLE:
            st.error("❌ **Scapy not installed.** Run `pip install scapy` in your virtual environment, then restart.")
            return

        from src.inference.streaming_engine import LiveCaptureSniffer
        from src.feature_pipeline.packet_extractor import PacketFeatureExtractor

        with st.spinner("Detecting network interfaces..."):
            friendly_ifaces = LiveCaptureSniffer.get_friendly_interfaces()

        if not friendly_ifaces:
            st.error("❌ No network interfaces found. Install **Npcap** from https://npcap.com (tick *WinPcap API-compatible mode*), then restart.")
            return

        iface_labels  = [label for label, raw in friendly_ifaces]
        iface_raw_map = {label: raw for label, raw in friendly_ifaces}

        st.markdown("---")
        col_iface1, col_iface2, col_iface3 = st.columns([4, 2, 2])
        with col_iface1:
            selected_label = st.selectbox(
                "🌐 Network Interface:",
                options=iface_labels,
                index=0,
                help="Pick your active network adapter. The adapter with active internet is pre-selected at the top."
            )
            selected_iface = iface_raw_map[selected_label]

        with col_iface2:
            capture_mode = st.radio(
                "Capture Mode:",
                ["📸 One-Shot Snapshot", "🔁 Continuous Monitor"],
                index=0,
                help="One-shot: captures once for a duration and forecasts immediately. Continuous: runs rolling 2s windows updating live."
            )

        with col_iface3:
            if "One-Shot" in capture_mode:
                capture_duration = st.slider("Snapshot Duration (s)", min_value=10, max_value=120, value=25, step=5)
            else:
                run_indefinite = st.checkbox(
                    "🔄 Run Non-Stop (Indefinitely)",
                    value=True,
                    help="Keep monitoring live network traffic continuously without stopping. Uncheck to specify a fixed window limit."
                )
                if run_indefinite:
                    max_continuous_windows = None
                else:
                    max_continuous_windows = st.slider(
                        "Monitoring Windows", min_value=10, max_value=500, value=60, step=10,
                        help="Number of 2s windows to process before stopping."
                    )

        # ── Continuous Monitor Mode ────────────────────────────────────────────
        if "Continuous" in capture_mode:
            dur_text = "indefinitely (non-stop)" if max_continuous_windows is None else f"for {max_continuous_windows} windows ({max_continuous_windows * 2}s)"
            st.info(
                f"🔴 **Continuous Monitor**: Capturing **2-second rolling windows** from `{selected_label}` {dur_text}.  \n"
                f"Each window extracts packet features, fuses state vector $S_t$, and projects the World Model rollout live."
            )
            start_monitor_btn = st.button("🟥 Start Continuous Network Monitor", use_container_width=True, type="primary")

            if start_monitor_btn:
                sniffer = LiveCaptureSniffer(interface=selected_iface, window_size_sec=2.0)
                monitor_placeholder = st.empty()
                progress_bar = st.progress(0) if max_continuous_windows else None
                event_log_monitor: List[str] = []
                all_windows_df = []

                try:
                    win_gen = sniffer.continuous_window_generator(
                        window_sec=2.0,
                        max_windows=max_continuous_windows
                    )

                    for win_idx, pkts in enumerate(win_gen):
                        if progress_bar and max_continuous_windows:
                            progress_bar.progress(min(1.0, (win_idx + 1) / max_continuous_windows))
                        t_str = time.strftime("%H:%M:%S")

                        if len(pkts) == 0:
                            event_log_monitor.append(f"[{t_str}] Window {win_idx+1:02d} — 0 pkts (idle segment)")
                            continue

                        try:
                            # Save window PCAP and extract features
                            from scapy.all import wrpcap
                            tmp_pcap = Path(PROJECT_ROOT) / "data" / "raw" / f"_live_win_{win_idx}.pcap"
                            tmp_pcap.parent.mkdir(parents=True, exist_ok=True)
                            wrpcap(str(tmp_pcap), pkts)

                            pkt_ext = PacketFeatureExtractor(window_size_sec=2.0)
                            pkt_df  = pkt_ext.extract_from_pcap(str(tmp_pcap))
                            fusion  = DualLevelFusionPipeline()
                            win_df  = fusion.fuse_dataframes(df_packet=pkt_df, is_live_capture=True)
                            all_windows_df.append(win_df)

                            combined_df = pd.concat(all_windows_df, ignore_index=True)
                            
                            # Pad earlier windows if len < 10 so we don't have to wait 20s to see forecasts
                            if len(combined_df) < 10:
                                pad_needed = 10 - len(combined_df)
                                # Add slight natural micro-jitter so cloned padding rows do not
                                # appear as an artificial zero-variance Command & Control periodic beacon
                                pad_list = []
                                for p_idx in range(pad_needed):
                                    jittered_row = combined_df.iloc[[0]].copy()
                                    jitter_factor = 1.0 + (p_idx - pad_needed / 2.0) * 0.03
                                    for col in ["packet_count", "packet_rate_per_sec", "byte_rate_per_sec", "flow_rate_per_sec"]:
                                        if col in jittered_row.columns:
                                            jittered_row[col] = (jittered_row[col] * jitter_factor).clip(lower=1.0)
                                    pad_list.append(jittered_row)
                                pad_rows = pd.concat(pad_list, ignore_index=True)
                                history_df = pd.concat([pad_rows, combined_df], ignore_index=True)
                            else:
                                history_df = combined_df.iloc[-10:].copy()

                            current_window = history_df.iloc[-1]
                            forecast     = engine.simulate_from_raw_sequence(history_df, horizon_k=horizon_k)
                            peak_risk    = forecast["peak_risk"]
                            term_stage   = forecast["terminal_stage"]

                            # During the first 2 windows (cold-start), the 10-window buffer is still warming up
                            if len(combined_df) < 3:
                                log_line = f"[{t_str}] 🔄 Win {win_idx+1:02d} — Buffer Warmup ({len(combined_df)}/3) — Risk {peak_risk*100:.1f}% — {len(pkts)} pkts"
                            elif peak_risk >= alert_threshold:
                                log_line = f"[{t_str}] 🚨 Win {win_idx+1:02d} — CRITICAL: {term_stage} — Risk {peak_risk*100:.1f}% — {len(pkts)} pkts"
                            elif peak_risk >= alert_threshold * 0.7:
                                log_line = f"[{t_str}] ⚡ Win {win_idx+1:02d} — ELEVATED: {term_stage} — Risk {peak_risk*100:.1f}% — {len(pkts)} pkts"
                            else:
                                log_line = f"[{t_str}] ✅ Win {win_idx+1:02d} — Nominal — Risk {peak_risk*100:.1f}% — {len(pkts)} pkts"

                            event_log_monitor.append(log_line)

                            display_label = f"`{win_idx+1}` of `{max_continuous_windows}`" if max_continuous_windows else f"`{win_idx+1}` (Non-Stop Live Stream)"
                            total_win_count = max_continuous_windows if max_continuous_windows else max(10, win_idx + 1)

                            with monitor_placeholder.container():
                                st.markdown(f"**🔴 Live Monitor Active:** Window {display_label} | Packets this window: `{len(pkts)}`")
                                render_soc_view(
                                    current_window=current_window,
                                    history_df=history_df,
                                    forecast=forecast,
                                    mapper=mapper,
                                    explainer=explainer,
                                    horizon_k=horizon_k,
                                    alert_threshold=alert_threshold,
                                    current_idx=win_idx,
                                    total_windows=total_win_count,
                                    prev_prob=None,
                                    event_logs=event_log_monitor,
                                    full_trace_df=combined_df
                                )
                        except Exception as win_err:
                            event_log_monitor.append(f"[{t_str}] ⚠️ Window {win_idx+1:02d} — Transient extraction hiccup: {win_err}")
                            continue

                except PermissionError:
                    st.error(
                        "❌ **Permission Denied.**  \n"
                        "Open **Command Prompt as Administrator**, navigate to the project folder, and run:  \n"
                        "```\nvenv\\Scripts\\streamlit.exe run src/app.py\n```"
                    )
                    return
                except OSError as ex:
                    st.error(
                        f"❌ **Interface Error:** `{ex}`  \n"
                        "• Install **Npcap** from https://npcap.com  \n"
                        "• Ensure you launched as Administrator"
                    )
                    return
                except Exception as ex:
                    st.error(f"❌ Monitoring failed: {ex}")
                    return

                st.success(f"✅ Continuous monitoring complete — {max_continuous_windows} windows processed.")
                if all_windows_df:
                    fused_df = pd.concat(all_windows_df, ignore_index=True)
                else:
                    return
            else:
                return

        # ── One-Shot Snapshot Mode ─────────────────────────────────────────────
        else:
            st.info(
                f"🔍 Capturing live packets on **`{selected_label}`** for **{capture_duration}s** in promiscuous mode,  \n"
                f"extracting dual-level features and running the World Model K-step forward simulation."
            )
            start_live_btn = st.button("🟥 Start Live Packet Capture & Forecast", use_container_width=True, type="primary")

            if start_live_btn:
                live_status = st.empty()
                live_status.info(
                    f"🔴 **Capturing on `{selected_label}`** — please wait **{capture_duration}s** ...  \n"
                    "The page will appear frozen while Scapy collects packets. This is normal."
                )
                try:
                    sniffer = LiveCaptureSniffer(interface=selected_iface, window_size_sec=2.0)
                    captured_pkts = sniffer.capture(duration_sec=capture_duration)
                    live_status.success(f"✅ Capture complete — **{len(captured_pkts)} packets** from `{selected_label}`.")

                except PermissionError:
                    st.error(
                        "❌ **Permission Denied.**  \n"
                        "Open **Command Prompt as Administrator** and relaunch:  \n"
                        "```\nvenv\\Scripts\\streamlit.exe run src/app.py\n```"
                    )
                    return
                except OSError as ex:
                    st.error(
                        f"❌ **Interface Error:** `{ex}`  \n"
                        "• Install **Npcap** from https://npcap.com (tick *WinPcap API-compatible mode*)  \n"
                        "• Ensure you launched as Administrator"
                    )
                    return
                except Exception as ex:
                    st.error(f"❌ Live capture failed: {ex}")
                    return

                if len(captured_pkts) < 5:
                    st.warning(
                        f"⚠️ Only **{len(captured_pkts)} packets** captured — not enough for a feature window.  \n"
                        "• Select your active **Wi-Fi** adapter  \n"
                        "• Browse websites or ping hosts to generate traffic, then retry"
                    )
                    return

                with st.spinner(f"Extracting features from {len(captured_pkts)} packets..."):
                    from scapy.all import wrpcap
                    tmp_pcap = Path(PROJECT_ROOT) / "data" / "raw" / "_live_capture_tmp.pcap"
                    tmp_pcap.parent.mkdir(parents=True, exist_ok=True)
                    wrpcap(str(tmp_pcap), captured_pkts)

                    pkt_ext  = PacketFeatureExtractor(window_size_sec=2.0)
                    pkt_df   = pkt_ext.extract_from_pcap(str(tmp_pcap))
                    fusion   = DualLevelFusionPipeline()
                    # is_live_capture=True → uses CIC-IDS-2017 scaler means for
                    # synthesized flow features so the normalised state vector
                    # stays near-zero and doesn't trigger false-positive alerts.
                    fused_df = fusion.fuse_dataframes(df_packet=pkt_df, is_live_capture=True)

                    # Ensure at least 10 windows for LSTM sequence history
                    if len(fused_df) < 10:
                        pad_needed = 10 - len(fused_df)
                        pad_rows = pd.concat([fused_df.iloc[[0]]] * pad_needed, ignore_index=True)
                        fused_df = pd.concat([pad_rows, fused_df], ignore_index=True)

                    st.success(f"✅ Fused **{len(fused_df)} time windows** — running World Model rollout...")
                    st.info(
                        "ℹ️ **Live Capture Context** — Risk scores are *predictive*, not reactive.  \n"
                        "The World Model was trained on CIC-IDS-2017 (2017 lab traffic). "
                        "Your live TLS/HTTPS browsing traffic will naturally differ from that baseline.  \n"
                        "A risk score below **40 %** on live capture means the model predicts "
                        "**no attack progression** in the next 10 seconds — your network is fine.  \n"
                        "Scores above 40 % suggest anomalous packet patterns worth investigating."
                    )
            else:
                return


    if fused_df is None or len(fused_df) < 10:
        st.info("👈 Select a sample dataset or upload traffic traces above to initiate forward simulation.")
        return

    total_windows = len(fused_df)

    # Live Stream Playback Controls for Mode B
    is_streaming = False
    if "Mode B" in mode:
        st.markdown("---")
        st.markdown("#### 🔴 Live Streaming Attack Controls")
        col_ctrl1, col_ctrl2 = st.columns([2, 3])
        with col_ctrl1:
            start_stream = st.button("▶ Start Real-Time Attack Stream Simulation", use_container_width=True)
        with col_ctrl2:
            playback_delay = st.slider("Window Interval Delay (seconds)", min_value=0.1, max_value=1.5, value=0.4, step=0.1)

        if start_stream:
            is_streaming = True
            live_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            start_win = 9
            end_win = min(len(fused_df), 70)
            session_event_logs: List[str] = []
            prev_p: Optional[float] = None

            for current_step, win_idx in enumerate(range(start_win, end_win)):
                progress_bar.progress((current_step + 1) / (end_win - start_win))
                history_df = fused_df.iloc[win_idx - 9 : win_idx + 1].copy()
                current_window = history_df.iloc[-1]
                forecast = engine.simulate_from_raw_sequence(history_df, horizon_k=horizon_k)

                curr_p = float(forecast["timeline"][0]["infiltration_probability"]) if forecast.get("timeline") else forecast["peak_risk"]
                term_stage = forecast["terminal_stage"]
                tact_info = mapper.get_tactic_for_label(str(current_window.get("label", "Benign")))
                
                # Format live event line
                t_str = time.strftime("%H:%M:%S")
                if forecast["peak_risk"] >= alert_threshold:
                    log_line = f"[{t_str}] ⚠️ Window {win_idx:02d} — CRITICAL ALERT: {term_stage} trajectory — Risk {forecast['peak_risk']*100:.1f}%"
                elif forecast["peak_risk"] >= alert_threshold * 0.7:
                    log_line = f"[{t_str}] ⚡ Window {win_idx:02d} — ELEVATED: {term_stage} projected — Risk {forecast['peak_risk']*100:.1f}%"
                else:
                    log_line = f"[{t_str}] Window {win_idx:02d} — {tact_info['tactic_name']} observed — Nominal Baseline"

                session_event_logs.append(log_line)

                with live_placeholder.container():
                    st.markdown(f"**🔴 Live Stream Active:** Window `t={win_idx}` (Elapsed: `+{win_idx * 2.0:.0f}s`)")
                    render_soc_view(
                        current_window=current_window,
                        history_df=history_df,
                        forecast=forecast,
                        mapper=mapper,
                        explainer=explainer,
                        horizon_k=horizon_k,
                        alert_threshold=alert_threshold,
                        current_idx=win_idx,
                        total_windows=total_windows,
                        prev_prob=prev_p,
                        event_logs=session_event_logs,
                        full_trace_df=fused_df
                    )

                prev_p = curr_p
                time.sleep(playback_delay)

            st.success("✅ Real-Time Attack Kill-Chain Stream Replay Complete!")

    if not is_streaming:
        # Timeline Selector for Manual Inspection
        st.markdown("---")
        st.subheader("🎯 Temporal Observation Window")
        
        if total_windows > 1:
            selected_idx = st.slider(
                "Select Active Analysis Window (t):",
                min_value=0,
                max_value=total_windows - 1,
                value=min(25, total_windows - 1),
                help="Sliding historical context window of length L=10 (t-9 to t) used to forecast t+1 to t+K."
            )
        else:
            selected_idx = 0

        # Extract historical observation [selected_idx - 9 : selected_idx + 1]
        if selected_idx >= 9:
            history_df = fused_df.iloc[selected_idx - 9 : selected_idx + 1].copy()
        else:
            pad_needed = 9 - selected_idx
            pad_rows = pd.concat([fused_df.iloc[[0]]] * pad_needed, ignore_index=True)
            history_df = pd.concat([pad_rows, fused_df.iloc[:selected_idx + 1]], ignore_index=True)

        current_window = history_df.iloc[-1]

        # Run Rollout Simulation Engine
        forecast = engine.simulate_from_raw_sequence(history_df, horizon_k=horizon_k)

        # In Mode A, construct recent event logs up to selected_idx
        mode_a_logs = []
        for w in range(max(0, selected_idx - 8), selected_idx + 1):
            w_row = fused_df.iloc[w]
            if w >= 9:
                w_hist = fused_df.iloc[w - 9 : w + 1]
            else:
                p_needed = 9 - w
                p_rows = pd.concat([fused_df.iloc[[0]]] * p_needed, ignore_index=True)
                w_hist = pd.concat([p_rows, fused_df.iloc[:w + 1]], ignore_index=True)
            w_fc = engine.simulate_from_raw_sequence(w_hist, horizon_k=horizon_k)
            w_stage = w_fc["terminal_stage"]
            w_risk = w_fc["peak_risk"]
            w_label = w_row.get("label", "Benign")
            t_sim = time.strftime("%H:%M:%S")
            if w_risk >= alert_threshold:
                mode_a_logs.append(f"[{t_sim}] ⚠️ Window {w:02d} — CRITICAL ALERT: {w_stage} trajectory ({w_risk*100:.1f}%)")
            elif w_risk >= alert_threshold * 0.7:
                mode_a_logs.append(f"[{t_sim}] ⚡ Window {w:02d} — ELEVATED: {w_stage} projected ({w_risk*100:.1f}%)")
            else:
                mode_a_logs.append(f"[{t_sim}] Window {w:02d} — {w_label} — Nominal Baseline")

        prev_w_prob = None
        if selected_idx > 0:
            if selected_idx - 1 >= 9:
                prev_hist = fused_df.iloc[selected_idx - 10 : selected_idx].copy()
            else:
                p_n = 9 - (selected_idx - 1)
                p_rows = pd.concat([fused_df.iloc[[0]]] * p_n, ignore_index=True)
                prev_hist = pd.concat([p_rows, fused_df.iloc[:selected_idx]], ignore_index=True)
            prev_fc = engine.simulate_from_raw_sequence(prev_hist, horizon_k=horizon_k)
            prev_w_prob = float(prev_fc["timeline"][0]["infiltration_probability"])

        # Render SOC Dashboard View
        render_soc_view(
            current_window=current_window,
            history_df=history_df,
            forecast=forecast,
            mapper=mapper,
            explainer=explainer,
            horizon_k=horizon_k,
            alert_threshold=alert_threshold,
            current_idx=selected_idx,
            total_windows=total_windows,
            prev_prob=prev_w_prob,
            event_logs=mode_a_logs,
            full_trace_df=fused_df
        )


if __name__ == "__main__":
    main()