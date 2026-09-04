"""SentinelPay -- Adaptive Fraud Intelligence for Emerging Threats.

A thin Streamlit presentation layer over the real, frozen Phase I backend
(`sentinelpay.inference.*`). This file contains NO fraud-detection logic:
every probability, diagnostic, and feature value shown here is read
directly off objects returned by `score_transaction` / `InferenceState`,
never recomputed, approximated, or re-derived. `scoring.py`, `state.py`,
and `artifacts.py` are imported from and called unmodified -- see
`ui_state.py` for the scratch-directory and duplicate-count plumbing
shared between this app and its tests.

Everything under "Threat Intelligence" is read directly from this
project's own persisted EDA reports (`reports/eda/*.json`) -- no numbers
there are invented either; where a real investigation (Phase E's
coordinated-abuse/Union-Find study) produced an inconclusive result, this
UI says so rather than presenting it as a working feature.

The sandbox/scratch-state safety mechanism from the previous revision is
preserved internally (see `ui_state.py`) but demoted to a small "Sandbox
environment" control on the Intelligence Lifecycle page -- it is no longer
the headline UI concept.

Navigation and page layout follow the product story explicitly: investigate
a payment -> understand its behavioral context -> understand its historical
intelligence -> see the fused adaptive risk assessment -> update the
intelligence lifecycle so future scores reflect this outcome. Every number
in that flow is still read directly off `ScoreResult`/`InferenceState` --
only the presentation order and labeling changed from the previous
research-dashboard layout.

Run with:
    .venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from sentinelpay.config import load_config, load_detection_config
from sentinelpay.inference.artifacts import load_artifact
from sentinelpay.inference.scoring import score_transaction
from sentinelpay.target_history import SMOOTHING_K, SUFFICIENT_HISTORY_THRESHOLD
from ui_state import (
    CANONICAL_ARTIFACT_PATH,
    CANONICAL_STATE_DIR,
    SCRATCH_STATE_DIR,
    ensure_scratch_state,
    load_scratch_state,
    record_observed,
    resolve_transaction,
    reset_scratch_state,
    state_summary,
)

REPORTS_DIR = Path("reports/eda")

st.set_page_config(
    page_title="SentinelPay",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------- #
# Demo scenarios -- verified against the canonical state directly (not
# assumed). Internal names stay technical; only the product copy that
# describes them to the user changes.
# --------------------------------------------------------------------- #
# 60 Phase D buffer rows, 58 resolved Phase F events (0 fraud) on disk.
ESTABLISHED_EXAMPLE = {
    "TransactionID": 900_000_101,
    "TransactionDT": 15_999_999,
    "TransactionAmt": 75.5,
    "has_identity": 1,
    "card1": 2755,
    "card2": 404.0,
    "card3": 150.0,
    "card5": 102.0,
    "addr1": 325.0,
}
# Zero presence in either phase_d_buffer or phase_f_counts -- a genuine
# cold start, not a simulated one.
NEW_IDENTITY_EXAMPLE = {
    "TransactionID": 900_000_102,
    "TransactionDT": 15_999_999,
    "TransactionAmt": 12.0,
    "has_identity": 0,
    "card1": 999_999,
    "card2": 999.0,
    "card3": 999.0,
    "card5": 999.0,
    "addr1": 999.0,
}
CUSTOM_DEFAULT = {
    "TransactionID": 900_000_103,
    "TransactionDT": 15_999_999,
    "TransactionAmt": 50.0,
    "has_identity": 1,
    "card1": 1000,
    "card2": 100.0,
    "card3": 150.0,
    "card5": 100.0,
    "addr1": 200.0,
}

NAV_PAGES = ["Overview", "Investigate a Payment", "Threat Intelligence", "Intelligence Lifecycle"]


# --------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------- #
@st.cache_resource
def get_artifact():
    return load_artifact(CANONICAL_ARTIFACT_PATH)


@st.cache_data
def load_json_report(filename: str) -> dict:
    with open(REPORTS_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def init_session_state() -> None:
    ensure_scratch_state()
    if "state" not in st.session_state:
        st.session_state.state = load_scratch_state()
    if "last_scored" not in st.session_state:
        st.session_state.last_scored = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "score_version" not in st.session_state:
        st.session_state.score_version = 0
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Overview"


init_session_state()
artifact = get_artifact()
detection_config = load_detection_config()
config = load_config()

# --------------------------------------------------------------------- #
# Design system -- dark fintech theme
# --------------------------------------------------------------------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --sp-bg: #0A0F1C;
    --sp-panel: #121A2C;
    --sp-panel-2: #16203548;
    --sp-border: #223049;
    --sp-text: #E7ECF3;
    --sp-muted: #90A0B8;
    --sp-accent: #4C8DFF;
    --sp-accent-2: #8B7CFF;
    --sp-good: #22C55E;
    --sp-good-bg: #16321f;
    --sp-warn: #F5A524;
    --sp-warn-bg: #362a10;
    --sp-bad: #EF5350;
    --sp-bad-bg: #341717;
    --sp-neutral: #7C8BA6;
    --sp-neutral-bg: #1c2333;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp { background: radial-gradient(circle at 15% 0%, #101a30 0%, var(--sp-bg) 45%) fixed; color: var(--sp-text); }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #080D18;
    border-right: 1px solid var(--sp-border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }

.sp-brand { display:flex; flex-direction:column; gap:2px; padding: 0 0.25rem 1.1rem 0.25rem; border-bottom: 1px solid var(--sp-border); margin-bottom: 0.9rem; }
.sp-brand-name { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; }
.sp-brand-tag { font-size: 0.74rem; color: var(--sp-muted); letter-spacing: 0.01em; }

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 9px 12px;
    margin: 1px 0;
    transition: background 0.12s ease;
    width: 100%;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: #131c30; }
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] p { font-size: 0.92rem; font-weight: 500; color: var(--sp-text); }

.sp-hero {
    padding: 1.9rem 2.1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #101c34 0%, #0d1526 100%);
    border: 1px solid var(--sp-border);
    margin-bottom: 1.4rem;
}
.sp-hero-eyebrow { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--sp-accent); margin-bottom: 6px; }
.sp-hero-title { font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em; color: #fff; margin-bottom: 6px; }
.sp-hero-sub { font-size: 0.98rem; color: var(--sp-muted); max-width: 640px; line-height: 1.5; }

.sp-pill { display:inline-flex; align-items:center; gap:6px; padding: 5px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.sp-pill-dot { width:7px; height:7px; border-radius:50%; }
.sp-pill-live { background: var(--sp-good-bg); color: var(--sp-good); }
.sp-pill-live .sp-pill-dot { background: var(--sp-good); box-shadow: 0 0 0 3px rgba(34,197,94,0.15); }
.sp-pill-neutral { background: var(--sp-neutral-bg); color: var(--sp-muted); }

.sp-kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 1.4rem; }
.sp-kpi-card { background: var(--sp-panel); border: 1px solid var(--sp-border); border-radius: 14px; padding: 18px 20px; }
.sp-kpi-label { font-size: 0.78rem; color: var(--sp-muted); font-weight: 500; margin-bottom: 8px; }
.sp-kpi-value { font-size: 1.9rem; font-weight: 800; color: #fff; letter-spacing: -0.01em; }
.sp-kpi-sub { font-size: 0.76rem; color: var(--sp-muted); margin-top: 6px; }
.sp-kpi-accent .sp-kpi-value { color: var(--sp-accent); }

.sp-section-title { font-size: 1.15rem; font-weight: 700; color: #fff; margin: 0.3rem 0 0.2rem 0; }
.sp-section-sub { font-size: 0.86rem; color: var(--sp-muted); margin-bottom: 0.9rem; }

.sp-chip { display:inline-flex; align-items:center; gap:6px; padding: 6px 13px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; margin-right: 8px; margin-bottom: 6px; }
.sp-chip-good { background: var(--sp-good-bg); color: var(--sp-good); }
.sp-chip-warn { background: var(--sp-warn-bg); color: var(--sp-warn); }
.sp-chip-bad { background: var(--sp-bad-bg); color: var(--sp-bad); }
.sp-chip-neutral { background: var(--sp-neutral-bg); color: var(--sp-muted); }

.sp-prob-hero { display:flex; align-items:baseline; gap:12px; margin-bottom: 4px; }
.sp-prob-value { font-size: 3rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
.sp-prob-label { font-size: 0.85rem; color: var(--sp-muted); }

.sp-card { background: var(--sp-panel); border: 1px solid var(--sp-border); border-radius: 14px; padding: 20px 22px; margin-bottom: 12px; }
.sp-card-title { font-size: 1rem; font-weight: 700; color: #fff; margin-bottom: 4px; }
.sp-card-body { font-size: 0.87rem; color: var(--sp-muted); line-height: 1.55; }

.sp-finding-badge { display:inline-block; padding: 3px 10px; border-radius: 6px; background: var(--sp-warn-bg); color: var(--sp-warn); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 10px; }

.sp-activity-row { display:flex; justify-content:space-between; align-items:center; padding: 10px 4px; border-bottom: 1px solid var(--sp-border); font-size: 0.86rem; }
.sp-activity-row:last-child { border-bottom: none; }
.sp-activity-key { color: var(--sp-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; }

.sp-param-table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
.sp-param-table td { padding: 8px 6px; border-bottom: 1px solid var(--sp-border); }
.sp-param-table td:first-child { color: var(--sp-muted); }
.sp-param-table td:last-child { color: #fff; font-family: 'JetBrains Mono', monospace; text-align: right; }

.sp-step-badge { display:inline-block; padding: 3px 10px; border-radius: 6px; background: var(--sp-neutral-bg); color: var(--sp-accent); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 10px; }

.sp-flow-strip { display:flex; align-items:center; flex-wrap:wrap; gap:6px; margin-bottom: 1.4rem; }
.sp-flow-step { display:flex; align-items:center; gap:8px; background: var(--sp-panel); border:1px solid var(--sp-border); border-radius:10px; padding:8px 14px; font-size:0.82rem; font-weight:600; color: var(--sp-text); }
.sp-flow-num { display:flex; align-items:center; justify-content:center; width:20px; height:20px; min-width:20px; border-radius:50%; background: var(--sp-accent); color:#fff; font-size:0.72rem; font-weight:700; }
.sp-flow-arrow { color: var(--sp-muted); font-size:0.9rem; }

.sp-stat-row { display:flex; flex-wrap:wrap; gap:26px; margin: 10px 0 4px 0; }
.sp-stat { display:flex; flex-direction:column; gap:2px; }
.sp-stat-label { font-size:0.72rem; color:var(--sp-muted); text-transform:uppercase; letter-spacing:0.03em; }
.sp-stat-value { font-size:1.05rem; font-weight:700; color:#fff; font-family:'JetBrains Mono', monospace; }

.stButton>button, .stFormSubmitButton>button {
    border-radius: 9px;
    border: 1px solid var(--sp-border);
    background: var(--sp-panel);
    color: var(--sp-text);
    font-weight: 600;
    padding: 0.5rem 1rem;
}
.stButton>button:hover, .stFormSubmitButton>button:hover { border-color: var(--sp-accent); color: var(--sp-accent); }
/* Streamlit 1.63 exposes the primary variant via data-testid (plain
   buttons: "stBaseButton-primary"; form-submit buttons:
   "stBaseButton-primaryFormSubmit") rather than a bare kind="primary"
   attribute -- match both with a prefix selector. */
.stButton>button[data-testid^="stBaseButton-primary"], .stFormSubmitButton>button[data-testid^="stBaseButton-primary"] {
    background: var(--sp-accent); border-color: var(--sp-accent); color: #fff;
}
.stButton>button[data-testid^="stBaseButton-primary"]:hover, .stFormSubmitButton>button[data-testid^="stBaseButton-primary"]:hover {
    background: #3c78e0; color: #fff;
}

[data-testid="stExpander"] { border: 1px solid var(--sp-border); border-radius: 12px; background: var(--sp-panel); }
hr { border-color: var(--sp-border) !important; }
</style>
""",
    unsafe_allow_html=True,
)


def fmt_int(n) -> str:
    return f"{int(n):,}"


def fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


def fmt_amt(x) -> str:
    return "—" if x is None else f"${x:,.2f}"


def fmt_num(x, decimals: int = 2) -> str:
    return "—" if x is None else f"{x:.{decimals}f}"


def fmt_rate(x) -> str:
    return "—" if x is None else fmt_pct(x)


def stat_row(items: list[tuple[str, str]]) -> None:
    html = ['<div class="sp-stat-row">']
    for label, value in items:
        html.append(f'<div class="sp-stat"><div class="sp-stat-label">{label}</div><div class="sp-stat-value">{value}</div></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def flow_strip(steps: list[str]) -> None:
    parts = ['<div class="sp-flow-strip">']
    for i, s in enumerate(steps, start=1):
        parts.append(f'<div class="sp-flow-step"><span class="sp-flow-num">{i}</span>{s}</div>')
        if i < len(steps):
            parts.append('<span class="sp-flow-arrow">&rarr;</span>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def kpi_grid(cards: list[dict]) -> None:
    html = ['<div class="sp-kpi-grid">']
    for c in cards:
        accent_cls = " sp-kpi-accent" if c.get("accent") else ""
        sub = f'<div class="sp-kpi-sub">{c["sub"]}</div>' if c.get("sub") else ""
        html.append(
            f'<div class="sp-kpi-card{accent_cls}"><div class="sp-kpi-label">{c["label"]}</div>'
            f'<div class="sp-kpi-value">{c["value"]}</div>{sub}</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def chip(text: str, tone: str = "neutral") -> str:
    return f'<span class="sp-chip sp-chip-{tone}">{text}</span>'


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sp-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sp-section-sub">{subtitle}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------- #
# Sidebar -- brand + navigation only. No raw counters here.
# --------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="sp-brand"><div class="sp-brand-name">SentinelPay</div>'
        '<div class="sp-brand-tag">Adaptive Fraud Intelligence for Emerging Threats</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state.nav_page = st.radio(
        "Navigate", NAV_PAGES, label_visibility="collapsed", key="nav_radio",
        index=NAV_PAGES.index(st.session_state.nav_page),
    )
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="sp-pill sp-pill-live"><span class="sp-pill-dot"></span>Model F2 &middot; Sealed-holdout validated</div>',
        unsafe_allow_html=True,
    )


# ======================================================================= #
# Behavioral / historical signal translation helpers (no invented
# thresholds -- these translate REAL backend flags/values into product
# language; the underlying cutoffs are the ones already configured in
# configs/detection.yaml and target_history.py).
# ======================================================================= #
BEHAVIOR_COPY = {
    "scored_normal": ("Consistent with recent behavior", "good"),
    "scored_outlier": ("Deviates from recent behavior", "bad"),
    "insufficient_history": ("Insufficient behavioral history", "neutral"),
    "zero_mad": ("Behavioral baseline too uniform to assess", "neutral"),
}


def log_activity(result, txn_id: int) -> None:
    st.session_state.activity_log.insert(
        0,
        {
            "txn_id": txn_id,
            "key": result.payment_proxy_key,
            "probability": result.fraud_probability,
            "behavior": result.phase_d_diagnostics["flag"],
            "sufficient_history": bool(result.phase_f_diagnostics["sufficient_target_history"]),
        },
    )
    st.session_state.activity_log = st.session_state.activity_log[:8]


def render_lifecycle_actions(txn_id: int) -> None:
    st.markdown('<span class="sp-step-badge">Step 5 &middot; Update Intelligence Lifecycle</span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sp-card-title" style="font-size:0.92rem">Feed this outcome back into SentinelPay</div>'
        '<div class="sp-card-body">Recording and resolving are the mechanism by which Step 2 and Step 3\'s '
        "signals stay current -- future scores for this identity reflect what happens here. Resubmitting the "
        "same Transaction ID is always a safe no-op.</div>",
        unsafe_allow_html=True,
    )
    txn = st.session_state.last_scored
    v = st.session_state.score_version
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Record occurrence -- updates this identity's behavioral context (Step 2).")
        if st.button("Record this transaction", key=f"lifecycle_record_{txn_id}_{v}"):
            new_state, n_new, n_dup = record_observed(st.session_state.state, [txn])
            st.session_state.state = new_state
            st.success(f"{n_new} transaction newly recorded, {n_dup} duplicate(s) skipped.")
    with col_b:
        st.caption("Resolve outcome -- updates this identity's historical intelligence (Step 3).")
        outcome = st.selectbox(
            "Confirmed outcome",
            [0, 1],
            format_func=lambda x: "Fraudulent" if x else "Legitimate",
            key=f"lifecycle_outcome_{txn_id}_{v}",
            label_visibility="collapsed",
        )
        if st.button("Resolve this transaction", key=f"lifecycle_resolve_{txn_id}_{v}"):
            record = {**txn, "isFraud": int(outcome)}
            new_state, n_new, n_dup = resolve_transaction(st.session_state.state, [record])
            st.session_state.state = new_state
            st.success(f"{n_new} outcome newly resolved, {n_dup} duplicate(s) skipped.")


def render_score_result(result, txn_id: int) -> None:
    d = result.phase_d_diagnostics
    f = result.phase_f_diagnostics
    d_flag = d["flag"]
    behavior_label, behavior_tone = BEHAVIOR_COPY.get(d_flag, (d_flag, "neutral"))
    sufficient = bool(f["sufficient_target_history"])
    prior_events = f["payment_proxy_prior_event_count"]
    prior_fraud = f["payment_proxy_prior_fraud_count"]
    hist_rate = f["payment_proxy_prior_fraud_rate_smoothed"]
    cold_start = bool(f["global_cold_start"])

    # --- Step 2: Behavioral Context ------------------------------------ #
    st.markdown('<span class="sp-step-badge">Step 2 &middot; Behavioral Context</span>', unsafe_allow_html=True)
    st.markdown(chip(behavior_label, behavior_tone), unsafe_allow_html=True)
    stat_row(
        [
            ("Modified z-score", fmt_num(d["modified_zscore"])),
            ("Prior transactions in window", fmt_int(d["prior_count_in_window"])),
            ("Prior median amount", fmt_amt(d["prior_median"])),
            ("Prior amount variability (MAD)", fmt_amt(d["prior_mad"])),
        ]
    )
    st.caption(
        "Compares this transaction against this payment identity's own recent spending window -- not a "
        "one-size-fits-all population rule."
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # --- Step 3: Historical Intelligence -------------------------------- #
    st.markdown('<span class="sp-step-badge">Step 3 &middot; Historical Intelligence</span>', unsafe_allow_html=True)
    identity_label = (
        f"Established identity &middot; {prior_events} resolved prior event(s)"
        if sufficient
        else f"Limited identity history &middot; {prior_events} resolved prior event(s)"
    )
    identity_tone = "good" if sufficient else "warn"
    chips_html = chip(identity_label, identity_tone)
    if cold_start:
        chips_html += chip("No resolved history in the system at all", "neutral")
    st.markdown(chips_html, unsafe_allow_html=True)
    stat_row(
        [
            ("Resolved prior fraud cases", fmt_int(prior_fraud)),
            ("Resolved prior events", fmt_int(prior_events)),
            ("Raw historical rate", fmt_rate(f["payment_proxy_prior_fraud_rate_raw"])),
            ("Population baseline rate", fmt_rate(f["global_prior_fraud_rate"])),
        ]
    )
    st.caption(
        f"Smoothed toward the population baseline when history is limited: **{fmt_pct(hist_rate)}**."
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # --- Step 4: Adaptive Risk Assessment -------------------------------- #
    st.markdown('<span class="sp-step-badge">Step 4 &middot; Adaptive Risk Assessment</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sp-prob-hero"><span class="sp-prob-value">{fmt_pct(result.fraud_probability)}</span>'
        f'<span class="sp-prob-label">model fraud probability</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Fuses the behavioral signal (Step 2) and the historical intelligence signal (Step 3) through "
        "SentinelPay's validated F2 model -- see Threat Intelligence for how it was built and validated."
    )

    with st.expander("Technical details"):
        st.caption(f"payment_proxy_key: `{result.payment_proxy_key}`  ·  TransactionID: `{txn_id}`")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Behavioral diagnostics**")
            st.json(d)
        with col_b:
            st.markdown("**Historical intelligence diagnostics**")
            st.json(f)
        st.markdown("**Model features (exact values passed to the classifier)**")
        st.dataframe(
            {"feature": list(result.features.keys()), "value": list(result.features.values())},
            width="stretch",
            hide_index=True,
        )

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
    render_lifecycle_actions(txn_id)


# ======================================================================= #
# Page: Overview
# ======================================================================= #
def render_overview() -> None:
    st.markdown(
        '<div class="sp-hero"><div class="sp-hero-eyebrow">Fraud Intelligence Platform</div>'
        '<div class="sp-hero-title">SentinelPay</div>'
        '<div class="sp-hero-sub">Adaptive Fraud Intelligence for Emerging Threats. SentinelPay fuses '
        'real-time behavioral monitoring with continuously updated historical intelligence to score '
        'payment risk as new patterns emerge -- without waiting for a full model retrain.</div></div>',
        unsafe_allow_html=True,
    )

    summary = state_summary(st.session_state.state)
    phase_h = load_json_report("phase_h_results.json")
    holdout_f2 = phase_h["ladder_results_holdout"]["F2"]
    lift = phase_h["graduation_holdout"]["pr_auc_relative_lift_f2_over_b2"]
    baseline_rate = (
        summary["global_fraud_count"] / summary["global_event_count"] if summary["global_event_count"] else 0.0
    )

    kpi_grid(
        [
            {"label": "Payment Identities Monitored", "value": fmt_int(summary["phase_f_keys"])},
            {
                "label": "Transactions in Behavioral Memory",
                "value": fmt_int(summary["phase_d_buffer_size"]),
                "sub": "Recent spend patterns tracked per identity",
            },
            {"label": "Baseline Fraud Incidence", "value": fmt_pct(baseline_rate), "sub": "Resolved historical outcomes"},
            {
                "label": "Detection Lift, Sealed Holdout",
                "value": f"{lift:.2f}x",
                "sub": f"ROC-AUC {holdout_f2['roc_auc']:.3f} on data never used for tuning",
                "accent": True,
            },
        ]
    )

    section_header("The SentinelPay Workflow", "Every payment moves through the same five stages")
    flow_strip(
        [
            "Investigate a Payment",
            "Behavioral Context",
            "Historical Intelligence",
            "Adaptive Risk Assessment",
            "Intelligence Lifecycle",
        ]
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        with st.container(border=True):
            section_header("How SentinelPay Adapts", "Four signals, one continuously updating intelligence loop")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    '<div class="sp-card-title" style="font-size:0.88rem">Behavioral Context</div>'
                    '<div class="sp-card-body">Every payment identity has a running profile of its own recent '
                    'spending pattern. A new transaction is compared against that identity\'s own history, not a '
                    'one-size-fits-all rule.</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    '<div class="sp-card-title" style="font-size:0.88rem">Historical Intelligence</div>'
                    '<div class="sp-card-body">Confirmed fraud outcomes continuously update each identity\'s risk '
                    'profile. Identities with little history fall back toward the population baseline instead of '
                    'an unreliable individual estimate.</div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    '<div class="sp-card-title" style="font-size:0.88rem">Adaptive Risk Assessment</div>'
                    '<div class="sp-card-body">Both signals feed one model, validated on a sealed holdout set the '
                    'system never trained or tuned on -- so the reported lift reflects genuinely unseen data.</div>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    '<div class="sp-card-title" style="font-size:0.88rem">Intelligence Lifecycle</div>'
                    '<div class="sp-card-body">Recording occurrences and resolving outcomes feeds straight back '
                    "into Steps 2 and 3, so behavioral context and historical intelligence both keep improving.</div>",
                    unsafe_allow_html=True,
                )
    with col2:
        with st.container(border=True):
            section_header("Recent Session Activity", "Transactions analyzed in this session")
            if not st.session_state.activity_log:
                st.caption("No transactions analyzed yet. Visit Investigate a Payment to run one.")
            else:
                rows = []
                for entry in st.session_state.activity_log:
                    rows.append(
                        f'<div class="sp-activity-row"><span class="sp-activity-key">{entry["key"]}</span>'
                        f'<span>{fmt_pct(entry["probability"])}</span></div>'
                    )
                st.markdown("".join(rows), unsafe_allow_html=True)


# ======================================================================= #
# Page: Investigate a Payment
# ======================================================================= #
def render_investigate_payment() -> None:
    section_header(
        "Investigate a Payment",
        "Walk a transaction through SentinelPay's behavioral context, historical intelligence, adaptive risk "
        "assessment, and intelligence-lifecycle update -- in that order.",
    )
    st.markdown('<span class="sp-step-badge">Step 1 &middot; Investigate a Payment</span>', unsafe_allow_html=True)

    if "txn_mode" not in st.session_state:
        st.session_state.txn_mode = None

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown('<div class="sp-card-title">Established Customer</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sp-card-body">A payment identity with an extensive, resolved transaction history '
                'already known to the system.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Analyze this scenario", key="btn_established", width="stretch"):
                st.session_state.txn_mode = "established"
    with col2:
        with st.container(border=True):
            st.markdown('<div class="sp-card-title">New / Unknown Identity</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sp-card-body">A payment identity SentinelPay has never observed before -- a '
                'genuine cold start, with no behavioral or historical record.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Analyze this scenario", key="btn_new_identity", width="stretch"):
                st.session_state.txn_mode = "new_identity"
    with col3:
        with st.container(border=True):
            st.markdown('<div class="sp-card-title">Custom Transaction</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sp-card-body">Enter your own transaction details to see how SentinelPay '
                'evaluates it.</div>',
                unsafe_allow_html=True,
            )
            if st.button("Build custom transaction", key="btn_custom", width="stretch"):
                st.session_state.txn_mode = "custom"

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if st.session_state.txn_mode in ("established", "new_identity"):
        defaults = ESTABLISHED_EXAMPLE if st.session_state.txn_mode == "established" else NEW_IDENTITY_EXAMPLE
        transaction = {k: v for k, v in defaults.items() if k != "TransactionID"}
        result = score_transaction(transaction, st.session_state.state, artifact, detection_config, config)
        st.session_state.last_scored = dict(defaults)
        st.session_state.last_result = result
        st.session_state.score_version += 1
        log_activity(result, defaults["TransactionID"])
        st.session_state.txn_mode = None

    elif st.session_state.txn_mode == "custom":
        with st.container(border=True):
            st.markdown('<div class="sp-card-title">Custom Transaction Details</div>', unsafe_allow_html=True)
            with st.form("custom_txn_form"):
                col1, col2 = st.columns(2)
                with col1:
                    txn_id = st.number_input("Transaction ID", value=int(CUSTOM_DEFAULT["TransactionID"]), step=1)
                    dt = st.number_input("Transaction time (TransactionDT)", value=int(CUSTOM_DEFAULT["TransactionDT"]), step=1)
                    amt = st.number_input("Transaction amount", value=float(CUSTOM_DEFAULT["TransactionAmt"]))
                    has_identity = st.selectbox("Device/identity data available", [0, 1], index=int(CUSTOM_DEFAULT["has_identity"]))
                with col2:
                    st.caption("Payment identity components (anonymized card/address attributes)")
                    card1 = st.number_input("card1", value=int(CUSTOM_DEFAULT["card1"]), step=1)
                    card2 = st.number_input("card2", value=float(CUSTOM_DEFAULT["card2"]))
                    card3 = st.number_input("card3", value=float(CUSTOM_DEFAULT["card3"]))
                    card5 = st.number_input("card5", value=float(CUSTOM_DEFAULT["card5"]))
                    addr1 = st.number_input("addr1", value=float(CUSTOM_DEFAULT["addr1"]))
                submitted = st.form_submit_button("Analyze transaction", type="primary")
            if submitted:
                transaction = {
                    "TransactionDT": int(dt),
                    "TransactionAmt": float(amt),
                    "has_identity": int(has_identity),
                    "card1": card1,
                    "card2": card2,
                    "card3": card3,
                    "card5": card5,
                    "addr1": addr1,
                }
                result = score_transaction(transaction, st.session_state.state, artifact, detection_config, config)
                st.session_state.last_scored = {"TransactionID": int(txn_id), **transaction}
                st.session_state.last_result = result
                st.session_state.score_version += 1
                log_activity(result, int(txn_id))

    if st.session_state.last_result is not None:
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            render_score_result(st.session_state.last_result, st.session_state.last_scored["TransactionID"])


# ======================================================================= #
# Page: Threat Intelligence
# ======================================================================= #
def render_threat_intelligence() -> None:
    section_header(
        "Threat Intelligence",
        "How SentinelPay's detection capability was built and validated, and what we investigated beyond it.",
    )

    phase_h = load_json_report("phase_h_results.json")
    holdout = phase_h["ladder_results_holdout"]
    grad = phase_h["graduation_holdout"]
    validation_ladder = phase_h["phase_g_validation_reference"]["ladder_results"]
    ci = grad["bootstrap_pr_auc_delta_f2_minus_b2"]

    with st.container(border=True):
        section_header("Model Validation")
        kpi_grid(
            [
                {"label": "Holdout ROC-AUC (F2)", "value": f"{holdout['F2']['roc_auc']:.3f}"},
                {"label": "Holdout PR-AUC (F2)", "value": f"{holdout['F2']['pr_auc']:.3f}"},
                {
                    "label": "PR-AUC Lift vs. Baseline",
                    "value": f"{grad['pr_auc_relative_lift_f2_over_b2']:.2f}x",
                    "sub": f"95% CI on the raw lift: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]",
                    "accent": True,
                },
                {"label": "Validation Gates Passed", "value": "4 / 4" if grad["all_gates_pass"] else "Incomplete"},
            ]
        )
        st.caption(
            "The holdout set was sealed before any model, feature, or threshold decision was made, and evaluated "
            "exactly once. Results below compare the population-baseline model (B2: transaction attributes only) "
            "against the full adaptive model (F2: behavioral + historical intelligence)."
        )
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            st.markdown("**Validation ladder (Phase G)** -- PR-AUC by capability added")
            st.bar_chart({"PR-AUC": {k: v["pr_auc"] for k, v in validation_ladder.items()}})
        with bcol2:
            st.markdown("**Sealed holdout confirmation (Phase H)** -- PR-AUC")
            st.bar_chart({"PR-AUC": {k: v["pr_auc"] for k, v in holdout.items()}})
        st.caption(
            "Note: the holdout run only re-evaluates the B2/F1/F2 steps (the ones the graduation decision depends "
            "on) -- earlier baseline-only steps (B0/B1) were not re-run on holdout and are not shown here."
        )

    with st.container(border=True):
        section_header(
            "How Behavioral Monitoring Works",
            "Fixed, pre-declared parameters -- documented before any model tuning, never selected from results.",
        )
        params = [
            ("Prior transactions considered per identity", f"{detection_config.window_size_events}"),
            ("Minimum history required to score behavior", f"{detection_config.min_history_for_score} events"),
            ("Anomaly threshold (modified z-score)", f"{detection_config.modified_zscore_threshold}"),
            ("Historical-rate smoothing strength", f"k = {SMOOTHING_K:g}"),
            ("Minimum resolved events for a trusted rate", f"{SUFFICIENT_HISTORY_THRESHOLD} events"),
        ]
        rows = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in params)
        st.markdown(f'<table class="sp-param-table">{rows}</table>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<span class="sp-finding-badge">Research Finding &middot; Not in Production</span>', unsafe_allow_html=True)
        section_header("Coordinated Abuse Investigation")
        try:
            e1 = load_json_report("phase_e1_results.json")
            e2 = load_json_report("phase_e2_results.json")
            lift_ratio = e1["recommendation"]["per_direction"]["device_to_payment"]["lift_ratio"]
            comp = e2["component_metrics_summary_overall"]
            conclusion = e2["fanout_stratified_diagnostic_evaluation"]["conclusion"]
            st.markdown(
                f"We investigated whether payment identities sharing a device reveal coordinated fraud rings. "
                f"Devices and payment identities shared partners **{lift_ratio:.1f}x** more often than population-wide "
                f"popularity alone would predict -- strong enough to justify a full graph-clustering study."
            )
            kpi_grid(
                [
                    {"label": "Relationship Rows Analyzed", "value": fmt_int(comp["n_rows"])},
                    {"label": "Median Cluster Size", "value": fmt_int(comp["merged_component_size_total"]["p50"])},
                    {
                        "label": "Transactions Sharing a Cluster",
                        "value": fmt_pct(comp["endpoints_same_component"]["pct_true"] / 100, 1),
                    },
                ]
            )
            st.markdown("**Finding:** " + conclusion.split("\n")[0])
            st.caption(
                "This signal did not clearly survive controlling for fan-out (highly-connected identities), and is "
                "therefore not wired into the live fraud score -- it remains an active research thread, reported "
                "honestly rather than shipped as a feature."
            )
        except FileNotFoundError:
            st.caption("Entity investigation reports not found in reports/eda/.")


# ======================================================================= #
# Page: Intelligence Lifecycle
# ======================================================================= #
def render_intelligence_lifecycle() -> None:
    section_header(
        "Intelligence Lifecycle",
        "Step 5 of the workflow, available standalone: record occurrences and resolve confirmed outcomes for "
        "any transaction, not only one just investigated above.",
    )

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown('<div class="sp-card-title">Record Observed Transaction</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sp-card-body">Registers that a transaction occurred, so future scoring can '
                "reference this identity's updated behavioral pattern. Does not require a fraud outcome. "
                "Re-submitting the same Transaction ID is always a safe no-op.</div>",
                unsafe_allow_html=True,
            )
            prefill = st.session_state.last_scored or ESTABLISHED_EXAMPLE
            v = st.session_state.score_version
            with st.form("record_form"):
                r_txn_id = st.number_input("Transaction ID", value=int(prefill["TransactionID"]), step=1, key=f"rec_id_{v}")
                r_dt = st.number_input("TransactionDT", value=int(prefill["TransactionDT"]), step=1, key=f"rec_dt_{v}")
                r_amt = st.number_input("Amount", value=float(prefill["TransactionAmt"]), key=f"rec_amt_{v}")
                rc1, rc2 = st.columns(2)
                with rc1:
                    r_card1 = st.number_input("card1", value=int(prefill["card1"]), step=1, key=f"rec_card1_{v}")
                    r_card2 = st.number_input("card2", value=float(prefill["card2"]), key=f"rec_card2_{v}")
                    r_card3 = st.number_input("card3", value=float(prefill["card3"]), key=f"rec_card3_{v}")
                with rc2:
                    r_card5 = st.number_input("card5", value=float(prefill["card5"]), key=f"rec_card5_{v}")
                    r_addr1 = st.number_input("addr1", value=float(prefill["addr1"]), key=f"rec_addr1_{v}")
                record_submitted = st.form_submit_button("Record transaction", type="primary")
            if record_submitted:
                record = {
                    "TransactionID": int(r_txn_id), "TransactionDT": int(r_dt), "TransactionAmt": float(r_amt),
                    "card1": r_card1, "card2": r_card2, "card3": r_card3, "card5": r_card5, "addr1": r_addr1,
                }
                new_state, n_new, n_dup = record_observed(st.session_state.state, [record])
                st.session_state.state = new_state
                st.success(f"{n_new} transaction newly recorded, {n_dup} duplicate(s) skipped.")

    with col2:
        with st.container(border=True):
            st.markdown('<div class="sp-card-title">Resolve Transaction Outcome</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sp-card-body">Confirms whether a transaction was ultimately fraudulent, updating '
                "this identity's historical intelligence for future scoring. Re-submitting the same Transaction "
                "ID is always a safe no-op.</div>",
                unsafe_allow_html=True,
            )
            prefill = st.session_state.last_scored or ESTABLISHED_EXAMPLE
            v = st.session_state.score_version
            with st.form("resolve_form"):
                s_txn_id = st.number_input("Transaction ID", value=int(prefill["TransactionID"]), step=1, key=f"res_id_{v}")
                is_fraud = st.selectbox("Confirmed outcome", [0, 1], format_func=lambda x: "Fraudulent" if x else "Legitimate", key=f"res_isfraud_{v}")
                rc1, rc2 = st.columns(2)
                with rc1:
                    s_card1 = st.number_input("card1", value=int(prefill["card1"]), step=1, key=f"res_card1_{v}")
                    s_card2 = st.number_input("card2", value=float(prefill["card2"]), key=f"res_card2_{v}")
                    s_card3 = st.number_input("card3", value=float(prefill["card3"]), key=f"res_card3_{v}")
                with rc2:
                    s_card5 = st.number_input("card5", value=float(prefill["card5"]), key=f"res_card5_{v}")
                    s_addr1 = st.number_input("addr1", value=float(prefill["addr1"]), key=f"res_addr1_{v}")
                resolve_submitted = st.form_submit_button("Resolve outcome", type="primary")
            if resolve_submitted:
                record = {
                    "TransactionID": int(s_txn_id), "isFraud": int(is_fraud),
                    "card1": s_card1, "card2": s_card2, "card3": s_card3, "card5": s_card5, "addr1": s_addr1,
                }
                new_state, n_new, n_dup = resolve_transaction(st.session_state.state, [record])
                st.session_state.state = new_state
                st.success(f"{n_new} outcome newly resolved, {n_dup} duplicate(s) skipped.")

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    with st.expander("Sandbox environment"):
        if st.session_state.get("flash_message"):
            st.success(st.session_state.flash_message)
            del st.session_state["flash_message"]
        st.caption(
            "This workspace runs against an isolated sandbox seeded from production intelligence. Actions taken "
            "here never modify the production snapshot."
        )
        summary = state_summary(st.session_state.state)
        kpi_grid(
            [
                {"label": "Behavioral records", "value": fmt_int(summary["phase_d_buffer_size"])},
                {"label": "Identities tracked", "value": fmt_int(summary["phase_f_keys"])},
                {"label": "Resolved outcomes", "value": fmt_int(summary["phase_f_processed_ids"])},
                {"label": "Confirmed fraud cases", "value": fmt_int(summary["global_fraud_count"])},
            ]
        )
        st.caption(f"Production snapshot (read-only): `{CANONICAL_STATE_DIR}`  ·  Sandbox: `{SCRATCH_STATE_DIR}`")
        if st.button("Reset sandbox to production snapshot"):
            reset_scratch_state()
            st.session_state.state = load_scratch_state()
            st.session_state.last_scored = None
            st.session_state.last_result = None
            st.session_state.flash_message = "Sandbox reset to the production snapshot."
            st.rerun()


# --------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------- #
PAGE_RENDERERS = {
    "Overview": render_overview,
    "Investigate a Payment": render_investigate_payment,
    "Threat Intelligence": render_threat_intelligence,
    "Intelligence Lifecycle": render_intelligence_lifecycle,
}
PAGE_RENDERERS[st.session_state.nav_page]()
