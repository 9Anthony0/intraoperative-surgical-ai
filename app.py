#!/usr/bin/env python3
"""
app.py — "Surgeon Console": clinical dark-mode dashboard (Streamlit + Plotly).

Part of the *Intraoperative Surgical Phase & Gesture Recognition Engine*.

Run with::

    streamlit run app.py

Design system
-------------
* **Clinical dark canvas.** Deep slate background (``#0B1220``), slightly lifted cards
  (``#111B2E``) with 1 px ``#334155`` borders and 8 px radii. Low-luminance surfaces reduce
  glare and eye strain in a dimmed OR. Saturated colour is reserved for *state* (cyan =
  live data, amber = tremor, red = hesitation, green = nominal), so anything bright on
  screen means something.
* **Two typefaces, two jobs.** A monospace face with tabular figures for every number
  (values never jitter sideways as digits change at 2-4 Hz) and a humanist sans for prose.
  Both are loaded from Google Fonts with system fallbacks, so the UI also works offline.
* **Hardware annunciator.** The alert banner imitates an LED annunciator panel: a lamp,
  a channel strip (HES / TRM / NOM), and a glowing bezel that *breathes* with an ease-in-out
  curve rather than strobing on and off. Its HTML depends only on the alert level, so
  Streamlit leaves the DOM node untouched between refreshes and the animation never restarts.
* **Settings terminal.** The sidebar is grouped into numbered hardware-style panels
  with uppercase monospace labels and a live system readout.

Architecture (unchanged)
------------------------
One long-lived ``LiveInferenceService`` (``st.cache_resource``) runs telemetry and
inference at 30 Hz on a background thread. The console is an ``st.fragment`` with
``run_every``, so only the live region redraws, from a thread-safe snapshot.

For the most consistent widget styling, pair this file with ``.streamlit/config.toml``
(``base = "dark"``). The CSS below also restyles the key widgets directly as a fallback.

Research prototype only: not a medical device, not clinically validated.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from inference import (DEFAULT_CHECKPOINT, DIFFICULTY_PRESETS, LATENCY_BUDGET_MS, AlertLevel,
                       InferenceConfig, LiveInferenceService, PhaseSegment, Prediction,
                       ServiceSnapshot)
from model import count_parameters
from simulator import Anomaly, Gesture

# --------------------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------------------

TEXT: Final[str] = "#E2E8F0"
MUTED: Final[str] = "#94A3B8"
DIM: Final[str] = "#64748B"
BORDER: Final[str] = "#334155"
BORDER_SOFT: Final[str] = "#1E293B"
CYAN: Final[str] = "#22D3EE"
GRID: Final[str] = "rgba(148, 163, 184, 0.07)"
MONO: Final[str] = "'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', Menlo, Consolas, monospace"

GESTURE_COLORS: Final[dict[int, str]] = {
    Gesture.IDLE: "#64748B",        # slate: nothing happening
    Gesture.SUTURING: "#2DD4BF",    # teal
    Gesture.DISSECTION: "#A78BFA",  # violet
}
ALERT_COLORS: Final[dict[AlertLevel, str]] = {
    AlertLevel.NOMINAL: "#22C55E",
    AlertLevel.TREMOR: "#FACC15",
    AlertLevel.HESITATION: "#EF4444",
}
PLOTLY_CONFIG: Final[dict[str, object]] = {"displayModeBar": False, "displaylogo": False}

#: Annunciator content per alert level: (css class, channel code, headline, detail).
#: Static strings only, so the DOM stays stable and the glow animation runs uninterrupted.
ANNUNCIATORS: Final[dict[AlertLevel, tuple[str, str, str, str]]] = {
    AlertLevel.HESITATION: (
        "ann-hes", "CH-02 · MOTOR PLAN", "MICRO-HESITATION DETECTED",
        "Velocity collapse mid-gesture. Confirm instrument intent before proceeding."),
    AlertLevel.TREMOR: (
        "ann-trm", "CH-01 · STEADINESS", "TREMOR DETECTED",
        "Elevated 6–12 Hz tool-tip oscillation. Consider a micro-pause or fatigue check."),
    AlertLevel.NOMINAL: (
        "ann-nom", "ALL CHANNELS", "KINEMATICS NOMINAL",
        "No hesitation or tremor above the detection threshold."),
}

FONT_URL: Final[str] = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
                        "&family=JetBrains+Mono:wght@400;500;600;700&display=swap")

CSS = "<style>@import url('" + FONT_URL + "');" + """

:root {
  --canvas: #0B1220; --card: #111B2E; --card-2: #0E1726; --border: #334155;
  --border-soft: #1E293B; --text: #E2E8F0; --muted: #94A3B8; --dim: #64748B;
  --cyan: #22D3EE; --red: #EF4444; --amber: #FACC15; --green: #22C55E;
  --mono: 'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', Menlo, Consolas, monospace;
  --sans: 'Inter', 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}

/* ---------- Standalone-app chrome: hide Streamlit's menu, toolbar, footer ---------- */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] { display: none !important; }
/* The header stays (it hosts the sidebar expand control) but becomes invisible. */
header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
header[data-testid="stHeader"] * { pointer-events: auto; }

/* ---------- Canvas ---------- */
.stApp {
  background: radial-gradient(1200px 520px at 75% -12%, rgba(34, 211, 238, 0.055), transparent 60%),
              var(--canvas) !important;
  color: var(--text); font-family: var(--sans);
}
.block-container { padding: 1rem 1.5rem 1.25rem 1.5rem !important; max-width: 100% !important; }
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] { gap: 0.75rem; }
[data-testid="stMarkdownContainer"] p { margin-bottom: 0; }
/* Streamlit pulls HTML markdown blocks up by -16px; that collapses our layout gaps. */
[data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }

/* ---------- Top bar ---------- */
.topbar {
  display: flex; justify-content: space-between; align-items: center; gap: 16px;
  padding: 10px 16px; border: 1px solid var(--border-soft); border-radius: 8px;
  background: linear-gradient(180deg, #0F1829 0%, #0B1220 100%);
}
.brand { display: flex; gap: 12px; align-items: center; }
.brand-mark {
  width: 34px; height: 34px; border-radius: 7px; display: grid; place-items: center;
  border: 1px solid rgba(34, 211, 238, 0.45); color: var(--cyan); font-size: 17px;
  box-shadow: inset 0 0 14px rgba(34, 211, 238, 0.18), 0 0 10px rgba(34, 211, 238, 0.12);
}
.brand-title { font-family: var(--mono); font-weight: 700; font-size: 0.92rem;
  letter-spacing: 0.24em; color: var(--text); }
.brand-sub { font-size: 0.72rem; color: var(--dim); letter-spacing: 0.03em; margin-top: 2px; }
.status { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.pill {
  display: flex; align-items: center; gap: 7px; padding: 5px 10px; border-radius: 5px;
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.12em; color: var(--dim);
  border: 1px solid var(--border-soft); background: #0A101C;
}
.pill b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #475569; }
.dot.ok { background: var(--green); box-shadow: 0 0 8px rgba(34, 197, 94, 0.85); }
.dot.warn { background: var(--amber); box-shadow: 0 0 8px rgba(250, 204, 21, 0.85); }

/* ---------- Hardware annunciator (alert banner) ---------- */
.annunciator {
  display: flex; align-items: center; gap: 18px; padding: 14px 20px; border-radius: 8px;
  border: 1px solid; font-family: var(--mono);
}
.lamp { width: 20px; height: 20px; border-radius: 50%; flex: 0 0 auto; }
.ann-body { flex: 1 1 auto; min-width: 0; }
.ann-code { font-size: 0.62rem; letter-spacing: 0.2em; opacity: 0.75; }
.ann-headline { font-size: 1.12rem; font-weight: 700; letter-spacing: 0.16em; margin-top: 2px; }
.ann-detail { font-family: var(--sans); font-size: 0.8rem; opacity: 0.8; margin-top: 3px; }
.channels { display: flex; gap: 6px; }
.chan {
  font-size: 0.62rem; letter-spacing: 0.14em; padding: 4px 8px; border-radius: 4px;
  border: 1px solid #263244; color: #3B4A60; background: rgba(2, 6, 23, 0.4);
}
.chan.on-red { color: #FECACA; border-color: var(--red); background: rgba(239, 68, 68, 0.16);
  box-shadow: 0 0 10px rgba(239, 68, 68, 0.45); }
.chan.on-amber { color: #FEF08A; border-color: var(--amber);
  background: rgba(250, 204, 21, 0.12); box-shadow: 0 0 10px rgba(250, 204, 21, 0.4); }
.chan.on-green { color: #BBF7D0; border-color: rgba(34, 197, 94, 0.6);
  background: rgba(34, 197, 94, 0.1); }

.ann-nom { background: linear-gradient(90deg, #0A1C15 0%, #0C1712 100%);
  border-color: rgba(34, 197, 94, 0.35); color: #86EFAC; }
.ann-nom .lamp {
  background: radial-gradient(circle at 35% 35%, #DCFCE7 0, #22C55E 40%, #14532D 100%);
  box-shadow: 0 0 12px rgba(34, 197, 94, 0.7); }

.ann-hes { background: linear-gradient(90deg, #2A0B10 0%, #1A090D 100%); color: #FCA5A5;
  animation: bezel-red 1.3s cubic-bezier(0.45, 0, 0.55, 1) infinite; }
.ann-hes .lamp {
  background: radial-gradient(circle at 35% 35%, #FFE4E6 0, #EF4444 42%, #7F1D1D 100%);
  animation: lamp-red 1.3s cubic-bezier(0.45, 0, 0.55, 1) infinite; }
@keyframes bezel-red {
  0%, 100% { border-color: rgba(239, 68, 68, 0.5);
    box-shadow: 0 0 0 1px rgba(239, 68, 68, 0.25), 0 0 8px rgba(239, 68, 68, 0.12),
                inset 0 0 14px rgba(239, 68, 68, 0.06); }
  50% { border-color: #F87171;
    box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.9), 0 0 28px rgba(239, 68, 68, 0.45),
                inset 0 0 30px rgba(239, 68, 68, 0.18); }
}
@keyframes lamp-red {
  0%, 100% { opacity: 0.55; box-shadow: 0 0 6px rgba(239, 68, 68, 0.4); }
  50% { opacity: 1; box-shadow: 0 0 20px 3px rgba(239, 68, 68, 0.9); }
}

.ann-trm { background: linear-gradient(90deg, #241C05 0%, #191405 100%); color: #FDE68A;
  animation: bezel-amber 1.7s cubic-bezier(0.45, 0, 0.55, 1) infinite; }
.ann-trm .lamp {
  background: radial-gradient(circle at 35% 35%, #FEF9C3 0, #FACC15 42%, #713F12 100%);
  animation: lamp-amber 1.7s cubic-bezier(0.45, 0, 0.55, 1) infinite; }
@keyframes bezel-amber {
  0%, 100% { border-color: rgba(250, 204, 21, 0.45);
    box-shadow: 0 0 0 1px rgba(250, 204, 21, 0.2), 0 0 8px rgba(250, 204, 21, 0.1),
                inset 0 0 14px rgba(250, 204, 21, 0.05); }
  50% { border-color: #FDE047;
    box-shadow: 0 0 0 1px rgba(253, 224, 71, 0.85), 0 0 24px rgba(250, 204, 21, 0.38),
                inset 0 0 26px rgba(250, 204, 21, 0.14); }
}
@keyframes lamp-amber {
  0%, 100% { opacity: 0.55; box-shadow: 0 0 6px rgba(250, 204, 21, 0.35); }
  50% { opacity: 1; box-shadow: 0 0 18px 3px rgba(250, 204, 21, 0.85); }
}
.ann-err { background: #1A090D; border-color: var(--red); color: #FCA5A5; }

/* ---------- Metric tiles ---------- */
.tiles { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
.tiles.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
@media (max-width: 1150px) { .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
.tile {
  background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
  border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px 11px 14px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
.tile-label { font-family: var(--mono); font-size: 0.6rem; letter-spacing: 0.18em;
  color: var(--muted); text-transform: uppercase; }
.tile-value { font-family: var(--mono); font-size: 1.6rem; font-weight: 600; color: var(--text);
  font-variant-numeric: tabular-nums; letter-spacing: -0.01em; margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tile-value .unit { font-size: 0.72rem; color: var(--dim); font-weight: 500; margin-left: 4px;
  letter-spacing: 0.06em; }
.tile-sub { font-family: var(--mono); font-size: 0.64rem; color: var(--dim); margin-top: 4px;
  letter-spacing: 0.04em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.t-red .tile-value { color: #F87171; text-shadow: 0 0 14px rgba(239, 68, 68, 0.45); }
.t-amber .tile-value { color: #FDE047; text-shadow: 0 0 12px rgba(250, 204, 21, 0.35); }
.t-green .tile-value { color: #4ADE80; }
.t-cyan .tile-value { color: var(--cyan); text-shadow: 0 0 12px rgba(34, 211, 238, 0.3); }
.t-red { border-color: rgba(239, 68, 68, 0.55); }
.t-amber { border-color: rgba(250, 204, 21, 0.45); }
.meter { position: relative; height: 4px; margin-top: 9px; border-radius: 2px;
  background: #1A2436; overflow: visible; }
.meter > .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px;
  background: var(--cyan); box-shadow: 0 0 6px rgba(34, 211, 238, 0.6); }
.t-red .meter > .fill { background: var(--red); box-shadow: 0 0 6px rgba(239, 68, 68, 0.7); }
.t-amber .meter > .fill { background: var(--amber); box-shadow: 0 0 6px rgba(250, 204, 21, 0.6); }
.t-green .meter > .fill { background: var(--green); box-shadow: 0 0 6px rgba(34, 197, 94, 0.6); }
.meter > .tick { position: absolute; top: -3px; width: 2px; height: 10px; background: #F87171;
  border-radius: 1px; }

/* ---------- Cards (keyed st.container elements) ---------- */
[class*="st-key-card_"] {
  background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
  border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px 10px 14px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 10px 28px rgba(0, 0, 0, 0.28);
  gap: 0.4rem !important;
}
.card-head {
  display: flex; justify-content: space-between; align-items: center;
  padding-bottom: 8px; border-bottom: 1px solid var(--border-soft);
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.18em;
  color: var(--muted); text-transform: uppercase;
}
.card-head .title { display: flex; align-items: center; gap: 8px; }
.card-head .title::before { content: ""; width: 3px; height: 12px; border-radius: 2px;
  background: var(--cyan); box-shadow: 0 0 8px rgba(34, 211, 238, 0.7); }
.card-head .tag { color: var(--cyan); border: 1px solid rgba(34, 211, 238, 0.35);
  padding: 1px 7px; border-radius: 4px; font-size: 0.6rem; letter-spacing: 0.14em; }
.legend-chips { display: flex; gap: 14px; font-family: var(--mono); font-size: 0.62rem;
  letter-spacing: 0.12em; color: var(--muted); padding-top: 2px; }
.legend-chips span::before { content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 6px; vertical-align: -1px; background: var(--c); }
.footnote { font-family: var(--mono); font-size: 0.6rem; color: var(--dim);
  letter-spacing: 0.04em; line-height: 1.5; padding-top: 4px; }

/* ---------- Event log ---------- */
.evlog { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.72rem; }
.evlog th { color: var(--dim); font-weight: 500; text-align: left; font-size: 0.6rem;
  letter-spacing: 0.14em; padding: 6px 8px; border-bottom: 1px solid var(--border-soft); }
.evlog td { color: var(--text); padding: 7px 8px; border-bottom: 1px solid rgba(30, 41, 59, 0.6);
  font-variant-numeric: tabular-nums; }
.evlog tr:last-child td { border-bottom: none; }
.led { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px;
  vertical-align: 0; }
.led.red { background: var(--red); box-shadow: 0 0 8px rgba(239, 68, 68, 0.9); }
.led.amber { background: var(--amber); box-shadow: 0 0 8px rgba(250, 204, 21, 0.85); }
.live { color: var(--cyan); }
.empty { color: var(--dim); font-family: var(--mono); font-size: 0.7rem; padding: 16px 4px;
  letter-spacing: 0.1em; }

/* ---------- Boot / warm-up panel ---------- */
.boot { border: 1px solid var(--border); border-radius: 8px; padding: 26px 22px;
  background: var(--card); font-family: var(--mono); color: var(--muted);
  font-size: 0.75rem; letter-spacing: 0.14em; }
.boot .bar { height: 4px; background: #1A2436; border-radius: 2px; margin-top: 14px; }
.boot .bar > div { height: 100%; background: var(--cyan); border-radius: 2px;
  box-shadow: 0 0 8px rgba(34, 211, 238, 0.7); }

/* ---------- Sidebar: settings terminal ---------- */
section[data-testid="stSidebar"] { background: #070C16 !important;
  border-right: 1px solid var(--border-soft); }
[data-testid="stSidebarHeader"] { height: 2.4rem; min-height: 0; padding-top: 0.6rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.2rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
.term-head { font-family: var(--mono); padding: 4px 2px 6px 2px; }
.term-head .t1 { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.24em; color: var(--text); }
.term-head .t2 { font-size: 0.6rem; letter-spacing: 0.16em; color: var(--dim); margin-top: 3px; }
[class*="st-key-grp_"] { border: 1px solid var(--border-soft); border-radius: 6px;
  padding: 10px 12px 8px 12px; background: #0B1220; gap: 0.35rem !important; }
.grp-head { display: flex; justify-content: space-between; font-family: var(--mono);
  font-size: 0.62rem; letter-spacing: 0.2em; color: var(--cyan); text-transform: uppercase;
  padding-bottom: 6px; border-bottom: 1px dashed #1E293B; margin-bottom: 2px; }
.grp-head .idx { color: var(--dim); }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-family: var(--mono) !important; font-size: 0.64rem !important; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--muted) !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  font-family: var(--mono) !important; font-size: 0.6rem !important; color: var(--dim) !important;
  letter-spacing: 0.04em; line-height: 1.45; }
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"],
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
section[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
  font-family: var(--mono) !important; font-size: 0.62rem !important; }
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
  color: var(--cyan) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] input {
  background-color: #0A101C !important; border-color: var(--border) !important;
  color: var(--text) !important; font-family: var(--mono) !important;
  font-size: 0.74rem !important; }
section[data-testid="stSidebar"] button[kind="secondary"] {
  background: #0A101C; border: 1px solid var(--border); color: var(--text);
  font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.18em; text-transform: uppercase; }
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
  border-color: var(--cyan); color: var(--cyan); box-shadow: 0 0 10px rgba(34, 211, 238, 0.2); }
.readout { font-family: var(--mono); font-size: 0.62rem; line-height: 1.75; color: var(--dim);
  border: 1px solid var(--border-soft); border-radius: 6px; padding: 8px 12px;
  background: #0B1220; }
.readout div { display: flex; justify-content: space-between; gap: 8px; }
.readout b { color: var(--muted); font-weight: 500; }
.legal { font-family: var(--mono); font-size: 0.56rem; letter-spacing: 0.14em; color: #475569;
  text-align: center; padding: 6px 0 2px 0; }

/* ---------- Accessibility ---------- */
@media (prefers-reduced-motion: reduce) {
  .ann-hes, .ann-trm, .ann-hes .lamp, .ann-trm .lamp { animation: none !important; }
  .ann-hes { border-color: #F87171; } .ann-trm { border-color: #FDE047; }
}
</style>
"""

# --------------------------------------------------------------------------------------
# Settings & service lifecycle
# --------------------------------------------------------------------------------------


@dataclass
class Settings:
    """Values collected from the sidebar on each full script run."""

    checkpoint: str
    seed: int
    streaming: bool
    difficulty: str
    sensitivity: float
    ema_alpha: float
    trail_frames: int
    timeline_seconds: int
    refresh_hz: float
    show_truth: bool
    reset: bool


@st.cache_resource(show_spinner="Loading model and starting the 30 Hz telemetry stream…")
def get_service(checkpoint: str, seed: int) -> LiveInferenceService:
    """Create and start the single background inference service.

    ``st.cache_resource`` makes it a process-wide singleton, so it survives reruns and is
    shared by browser tabs. That fits a single-console demo. A multi-user deployment would
    need one service per session, or a separate inference server.
    """
    service = LiveInferenceService(checkpoint, seed=seed)
    service.start()
    return service


def _group_head(index: str, title: str) -> None:
    st.markdown(f'<div class="grp-head"><span>{title}</span><span class="idx">{index}</span>'
                '</div>', unsafe_allow_html=True)


def render_sidebar() -> Settings:
    """Hardware-style settings terminal: numbered, bordered control groups."""
    sb = st.sidebar
    sb.markdown('<div class="term-head"><div class="t1">◧ SYSTEM CONFIG</div>'
                '<div class="t2">SETTINGS TERMINAL · LIVE-APPLIED</div></div>',
                unsafe_allow_html=True)

    with sb.container(key="grp_stream"):
        _group_head("01", "Telemetry")
        streaming = st.toggle("Live stream", value=True,
                              help="Pause/resume the 30 Hz robot stream and inference.")
        refresh = st.select_slider(
            "Display refresh (Hz)", options=[1.0, 2.0, 4.0], value=2.0,
            help="Chart redraw rate only; inference always runs at 30 Hz. Use 4 Hz on a "
                 "machine with a dedicated GPU, since each redraw rebuilds a WebGL scene.")

    with sb.container(key="grp_case"):
        _group_head("02", "Case profile")
        difficulty = st.selectbox(
            "Simulated case", list(DIFFICULTY_PRESETS), index=1,
            help="Changes anomaly frequency and mix. Sensor noise stays at training levels "
                 "to avoid an unfair distribution shift.")
        preset = DIFFICULTY_PRESETS[difficulty]
        st.caption(f"≈1 anomaly / {preset['anomaly_mean_interval_s'] + 1:.0f} s · "
                   f"{100 * preset['hesitation_probability']:.0f}% hesitation")

    with sb.container(key="grp_detect"):
        _group_head("03", "Detection")
        sensitivity = st.slider("Alert sensitivity", 0.0, 1.0, 0.4, 0.05,
                                help="Higher = alerts fire at lower probability: earlier "
                                     "warnings, more false alarms.")
        threshold = InferenceConfig.thresholds_for_sensitivity(sensitivity)
        st.caption(f"TRIP P≥{threshold:.2f} × 3 FR · RELEASE P<{0.7 * threshold:.2f}")
        ema_alpha = st.slider("EMA smoothing factor", 0.10, 1.00, 0.35, 0.05,
                              help="Alpha of the probability EMA. Lower = smoother, fewer "
                                   "flickers, slightly slower; 1.0 disables smoothing.")

    with sb.container(key="grp_display"):
        _group_head("04", "Display")
        trail = st.slider("3D trail (frames)", 30, 600, 150, 30)
        timeline = st.slider("Timeline window (s)", 20, 180, 60, 10)
        show_truth = st.toggle("Ground-truth overlay", value=True,
                               help="Only possible because the data is simulated; useful "
                                    "for judging the model live.")

    with sb.container(key="grp_session"):
        _group_head("05", "Session")
        seed = int(st.number_input("Procedure seed", min_value=0, value=2026, step=1))
        checkpoint = st.text_input("Checkpoint", str(DEFAULT_CHECKPOINT))
        reset = st.button("↻ Restart procedure", width="stretch")

    return Settings(checkpoint, seed, streaming, difficulty, sensitivity, ema_alpha, trail,
                    timeline, float(refresh), show_truth, reset)


def render_system_readout(service: LiveInferenceService) -> None:
    """Static model/runtime facts at the bottom of the settings terminal."""
    engine = service.engine
    cfg = engine.model.config
    rows = [
        ("MODEL", "AttentionTCN"),
        ("PARAMS", f"{count_parameters(engine.model):,}"),
        ("WINDOW", f"{engine.window_size} FR / {engine.window_size / 30:.1f} S"),
        ("RECEPTIVE FIELD", f"{cfg.receptive_field} FR"),
        ("HEADS", "GESTURE · ANOMALY"),
        ("LATENCY BUDGET", f"{LATENCY_BUDGET_MS:.0f} MS"),
    ]
    body = "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in rows)
    st.sidebar.markdown(f'<div class="readout">{body}</div>'
                        '<div class="legal">RESEARCH PROTOTYPE · NOT A MEDICAL DEVICE</div>',
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Figures (all: plotly_dark, transparent backgrounds, near-invisible grids)
# --------------------------------------------------------------------------------------


def _style(fig: go.Figure, height: int, **layout: object) -> go.Figure:
    """Apply the console's chart language to any figure."""
    base: dict[str, object] = dict(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, font=dict(family=MONO, size=11, color=MUTED),
        margin=dict(l=0, r=0, t=6, b=0),
        hoverlabel=dict(bgcolor="#0B1220", bordercolor=BORDER,
                        font=dict(family=MONO, size=11, color=TEXT)))
    base.update(layout)  # caller-supplied keys (e.g. margin) override the defaults
    fig.update_layout(**base)
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=BORDER_SOFT,
                tickfont=dict(size=10, color=DIM), title_font=dict(size=10, color=DIM))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _alert_runs(preds: list[Prediction]) -> list[tuple[float, float, AlertLevel]]:
    """Collapse per-frame alert levels into contiguous (start, end, level) runs."""
    runs: list[tuple[float, float, AlertLevel]] = []
    start, level = None, AlertLevel.NOMINAL
    for p in preds:
        if p.alert != level:
            if level is not AlertLevel.NOMINAL and start is not None:
                runs.append((start, p.sim_time, level))
            start, level = p.sim_time, p.alert
    if level is not AlertLevel.NOMINAL and start is not None and preds:
        runs.append((start, preds[-1].sim_time, level))
    return runs


def trajectory_figure(preds: list[Prediction], trail: int) -> go.Figure:
    """Neon tool-tip path in a dark 3D void.

    Plotly's WebGL lines can neither blur nor blend alpha (translucent 3D lines draw as
    opaque), so the glow is built from *solid* colours: two wide, dark-teal halo passes
    under a thin core line whose colour brightens from old (deep teal) to recent (ice cyan).
    On a near-black canvas, dark teal reads as light bleeding from the core. A floor shadow
    gives depth cues, and the aspect ratio is fixed so millimetre-scale vertical motion
    is not flattened. Plain lists are passed instead of NumPy arrays, because some bundled
    plotly.js builds misread typed-array colour data for 3D lines.
    """
    recent = preds[-trail:]
    pos = np.stack([p.position for p in recent])
    x, y, z = pos[:, 0].tolist(), pos[:, 1].tolist(), pos[:, 2].tolist()
    n = len(recent)
    age = np.linspace(0.0, 1.0, n).tolist()
    alerts = np.array([int(p.alert) for p in recent])
    floor = float(pos[:, 2].min()) - 1.5

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=[floor] * n, mode="lines", hoverinfo="skip",
                               showlegend=False, line=dict(width=2, color="#123440")))
    for width, color in ((12, "#0A2A33"), (7, "#0D4654")):   # outer, inner halo
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip",
                                   showlegend=False, line=dict(width=width, color=color)))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(width=4, color=age, cmin=0, cmax=1,
                  colorscale=[[0.0, "#0E7490"], [0.65, "#22D3EE"], [1.0, "#CFFAFE"]])))
    for level in (AlertLevel.TREMOR, AlertLevel.HESITATION):
        mask = alerts == int(level)
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=pos[mask, 0].tolist(), y=pos[mask, 1].tolist(), z=pos[mask, 2].tolist(),
                mode="markers", name=level.name.title(),
                marker=dict(size=3.5, color=ALERT_COLORS[level], opacity=0.95),
                hovertemplate=f"{level.name.title()}<extra></extra>"))
    tip = recent[-1]
    tip_color = CYAN if tip.alert is AlertLevel.NOMINAL else ALERT_COLORS[tip.alert]
    halo_color = {AlertLevel.NOMINAL: "#0D4654", AlertLevel.TREMOR: "#4A3B08",
                  AlertLevel.HESITATION: "#4C1218"}[tip.alert]
    fig.add_trace(go.Scatter3d(x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers",
                               hoverinfo="skip", showlegend=False,
                               marker=dict(size=18, color=halo_color)))
    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]], mode="markers", name="Tip",
        marker=dict(size=7, symbol="diamond", color=tip_color,
                    line=dict(width=1, color="#F8FAFC")),
        hovertemplate="x %{x:.1f} · y %{y:.1f} · z %{z:.1f} mm<extra>TIP</extra>"))

    axis = dict(showbackground=False, gridcolor="#162133", zeroline=False,
                showspikes=False, linecolor=BORDER_SOFT, tickfont=dict(size=9, color=DIM),
                title_font=dict(size=10, color=DIM))
    # Height matches the right-hand card stack; constant uirevision keeps the user's camera.
    _style(fig, 586, uirevision="trajectory",
           legend=dict(orientation="h", x=0.0, y=1.0, font=dict(size=10),
                       bgcolor="rgba(0,0,0,0)"),
           scene=dict(xaxis=dict(title="X MM", **axis), yaxis=dict(title="Y MM", **axis),
                      zaxis=dict(title="Z MM", **axis), bgcolor="rgba(0,0,0,0)",
                      aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=0.75),
                      camera=dict(eye=dict(x=1.25, y=1.2, z=0.75))))
    return fig


def probability_figure(latest: Prediction, threshold: float) -> go.Figure:
    """Gauge-style bars for both heads: dim full-scale track, bright value, trip line."""
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.26,
                        subplot_titles=("ANOMALY HEAD", "GESTURE HEAD"))
    heads = (
        (1, [a.name for a in Anomaly], latest.anomaly_probs.tolist(),
         [ALERT_COLORS[AlertLevel.NOMINAL], ALERT_COLORS[AlertLevel.TREMOR],
          ALERT_COLORS[AlertLevel.HESITATION]]),
        (2, [g.name for g in Gesture], latest.gesture_probs.tolist(),
         [GESTURE_COLORS[g] for g in Gesture]),
    )
    for row, names, values, colors in heads:
        fig.add_trace(go.Bar(x=[1.0] * len(names), y=names, orientation="h",
                             marker=dict(color="rgba(148, 163, 184, 0.07)"),
                             hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Bar(
            x=values, y=names, orientation="h", marker=dict(color=colors, line=dict(width=0)),
            texttemplate="%{x:.0%}", textposition="outside", cliponaxis=False,
            textfont=dict(family=MONO, size=11, color=TEXT),
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"), row=row, col=1)
    fig.add_vline(x=threshold, line=dict(color="#F87171", dash="dot", width=1.5), row=1, col=1)
    _style(fig, 290, barmode="overlay", bargap=0.38, showlegend=False,
           margin=dict(l=0, r=36, t=22, b=0))
    fig.update_xaxes(range=[0, 1.0], tickformat=".0%", showgrid=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, automargin=True, tickfont=dict(size=10, color=MUTED))
    fig.update_annotations(font=dict(family=MONO, size=10, color=DIM), x=0.0,
                           xanchor="left")
    return fig


def kinematics_figure(preds: list[Prediction], trail: int, threshold: float) -> go.Figure:
    """Tip speed (cyan area) with anomaly probabilities. Hesitation shows as a speed collapse."""
    recent = preds[-trail:]
    t = [p.sim_time for p in recent]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for start, end, level in _alert_runs(recent):
        fig.add_vrect(x0=start, x1=max(end, start + 1 / 30), fillcolor=ALERT_COLORS[level],
                      opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=t, y=[p.velocity for p in recent], name="SPEED",
                             line=dict(color=CYAN, width=1.8), fill="tozeroy",
                             fillcolor="rgba(34, 211, 238, 0.07)",
                             hovertemplate="%{y:.1f} mm/s<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=[p.anomaly_probs[2] for p in recent], name="P(HES)",
                             line=dict(color=ALERT_COLORS[AlertLevel.HESITATION], width=1.4,
                                       dash="dot"),
                             hovertemplate="P(hes) %{y:.2f}<extra></extra>"), secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=[p.anomaly_probs[1] for p in recent], name="P(TRM)",
                             line=dict(color=ALERT_COLORS[AlertLevel.TREMOR], width=1.4,
                                       dash="dot"),
                             hovertemplate="P(trm) %{y:.2f}<extra></extra>"), secondary_y=True)
    fig.add_hline(y=threshold, line=dict(color="#F87171", dash="dot", width=1),
                  secondary_y=True)
    _style(fig, 250, hovermode="x unified",
           legend=dict(orientation="h", x=0.0, y=-0.22, font=dict(size=10),
                       bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(title_text="MM/S", rangemode="tozero", secondary_y=False)
    fig.update_yaxes(title_text="P", range=[0, 1.05], tickvals=[0, 0.5, 1.0],
                     showgrid=False, secondary_y=True)
    fig.update_yaxes(nticks=4, secondary_y=False)
    fig.update_xaxes(showgrid=False, ticksuffix=" s")
    return fig


def timeline_figure(snap: ServiceSnapshot, window_s: int, show_truth: bool) -> go.Figure:
    """Gantt-style surgical workflow: model phases, ground truth, alert episodes."""
    now = snap.latest.sim_time if snap.latest else 0.0
    t0 = max(0.0, now - window_s)
    rows: list[tuple[str, list[PhaseSegment]]] = [("MODEL", snap.predicted_phases)]
    if show_truth:
        rows.append(("TRUTH", snap.true_phases))

    fig = go.Figure()
    for row_name, segments in rows:
        for gesture in Gesture:
            spans = [(max(s.start, t0), min(s.end, now)) for s in segments
                     if s.gesture == gesture and s.end >= t0]
            spans = [(a, b) for a, b in spans if b > a]
            if spans:
                fig.add_trace(go.Bar(
                    y=[row_name] * len(spans), x=[b - a for a, b in spans],
                    base=[a for a, _ in spans], orientation="h", showlegend=False,
                    marker=dict(color=GESTURE_COLORS[gesture], line=dict(width=0)),
                    hovertemplate=f"{gesture.name} · %{{base:.1f}}s +%{{x:.1f}}s"
                                  f"<extra>{row_name}</extra>"))
    for level in (AlertLevel.TREMOR, AlertLevel.HESITATION):
        spans = [(max(e.start, t0), min(e.end if e.end is not None else now, now))
                 for e in snap.alert_events
                 if e.level == level and (e.end is None or e.end >= t0)]
        spans = [(a, max(b, a + 0.12)) for a, b in spans]
        if spans:
            fig.add_trace(go.Bar(
                y=["ALERTS"] * len(spans), x=[b - a for a, b in spans],
                base=[a for a, _ in spans], orientation="h", showlegend=False,
                marker=dict(color=ALERT_COLORS[level], line=dict(width=0)),
                hovertemplate=f"{level.name} · %{{base:.1f}}s, %{{x:.1f}}s<extra></extra>"))

    order = ["ALERTS"] + [name for name, _ in reversed(rows)]
    fig.add_vline(x=now, line=dict(color=CYAN, width=1.5, dash="dot"))
    _style(fig, 52 + 40 * len(order), barmode="overlay", bargap=0.42,
           xaxis=dict(range=[t0, max(now, t0 + 1)], ticksuffix=" s", showgrid=True),
           yaxis=dict(categoryorder="array", categoryarray=order, automargin=True,
                      showgrid=False, tickfont=dict(size=10, color=MUTED)))
    return fig


# --------------------------------------------------------------------------------------
# HTML panels
# --------------------------------------------------------------------------------------


def _card_head(title: str, tag: str = "") -> None:
    tag_html = f'<span class="tag">{html.escape(tag)}</span>' if tag else ""
    st.markdown(f'<div class="card-head"><span class="title">{html.escape(title)}</span>'
                f'{tag_html}</div>', unsafe_allow_html=True)


def _tile(label: str, value: str, sub: str, tone: str = "", meter: float | None = None,
          tick: float | None = None, value_style: str = "") -> str:
    """One metric tile as HTML. ``meter``/``tick`` are fractions in [0, 1]."""
    meter_html = ""
    if meter is not None:
        tick_html = (f'<div class="tick" style="left:calc({100 * tick:.1f}% - 1px)"></div>'
                     if tick is not None else "")
        meter_html = (f'<div class="meter"><div class="fill" '
                      f'style="width:{100 * min(max(meter, 0.0), 1.0):.1f}%"></div>'
                      f'{tick_html}</div>')
    style = f' style="{value_style}"' if value_style else ""
    return (f'<div class="tile {tone}"><div class="tile-label">{label}</div>'
            f'<div class="tile-value"{style}>{value}</div><div class="tile-sub">{sub}</div>'
            f'{meter_html}</div>')


def _prob_tone(p: float, threshold: float, channel: AlertLevel) -> str:
    """Tile tone in the channel's own colour: red for hesitation, amber for tremor."""
    if p >= threshold:
        return "t-red" if channel is AlertLevel.HESITATION else "t-amber"
    if channel is AlertLevel.HESITATION and p >= 0.7 * threshold:
        return "t-amber"   # approaching the trip point
    return ""


def render_topbar(snap: ServiceSnapshot, service: LiveInferenceService,
                  settings: Settings) -> None:
    clock = snap.latest.sim_time if snap.latest else 0.0
    minutes, seconds = divmod(clock, 60.0)
    if not service.running:
        stream = '<div class="pill"><span class="dot"></span>STREAM <b>PAUSED</b></div>'
    else:
        dot = "ok" if snap.stream_hz >= 27.0 else "warn"
        stream = (f'<div class="pill"><span class="dot {dot}"></span>STREAM '
                  f'<b>{snap.stream_hz:4.1f} HZ</b></div>')
    st.markdown(
        '<div class="topbar"><div class="brand"><div class="brand-mark">◈</div><div>'
        '<div class="brand-title">SURGEON CONSOLE</div>'
        '<div class="brand-sub">Intraoperative phase &amp; gesture recognition · '
        '8-DOF kinematics · attention-augmented causal TCN</div></div></div>'
        f'<div class="status">{stream}'
        f'<div class="pill">CASE <b>{html.escape(settings.difficulty.upper())}</b></div>'
        f'<div class="pill">SEED <b>{settings.seed}</b></div>'
        f'<div class="pill">T+ <b>{int(minutes):02d}:{seconds:04.1f}</b></div>'
        '</div></div>', unsafe_allow_html=True)


def render_annunciator(level: AlertLevel) -> None:
    """LED annunciator. The markup depends only on ``level`` (see module docstring)."""
    css_class, code, headline, detail = ANNUNCIATORS[level]
    chans = {
        AlertLevel.HESITATION: ("on-red", "", ""),
        AlertLevel.TREMOR: ("", "on-amber", ""),
        AlertLevel.NOMINAL: ("", "", "on-green"),
    }[level]
    channels = "".join(f'<span class="chan {state}">{name}</span>'
                       for name, state in zip(("HES", "TRM", "NOM"), chans))
    st.markdown(f'<div class="annunciator {css_class}"><div class="lamp"></div>'
                f'<div class="ann-body"><div class="ann-code">{code}</div>'
                f'<div class="ann-headline">{headline}</div>'
                f'<div class="ann-detail">{detail}</div></div>'
                f'<div class="channels">{channels}</div></div>', unsafe_allow_html=True)


def render_tiles(snap: ServiceSnapshot, latest: Prediction, threshold: float) -> None:
    recent = snap.predictions[-90:]
    model_p50 = float(np.percentile([p.model_latency_ms for p in recent], 50))
    e2e_p99 = float(np.percentile([p.e2e_latency_ms for p in recent], 99))
    phase = snap.predicted_phases[-1] if snap.predicted_phases else None
    elapsed = (phase.end - phase.start) if phase else 0.0
    p_hes, p_trm = float(latest.anomaly_probs[2]), float(latest.anomaly_probs[1])
    color = GESTURE_COLORS[latest.gesture]

    tiles = [
        _tile("Surgical phase", latest.gesture.name,
              f"CONF {100 * latest.gesture_probs.max():3.0f}% · {elapsed:5.1f} S IN PHASE",
              meter=float(latest.gesture_probs.max()),
              value_style=f"color:{color};text-shadow:0 0 12px {color}55"),
        _tile("P(hesitation)", f"{100 * p_hes:3.0f}<span class='unit'>%</span>",
              f"RAW {100 * latest.raw_anomaly_probs[2]:3.0f}% · TRIP {100 * threshold:.0f}%",
              tone=_prob_tone(p_hes, threshold, AlertLevel.HESITATION), meter=p_hes,
              tick=threshold),
        _tile("P(tremor)", f"{100 * p_trm:3.0f}<span class='unit'>%</span>",
              f"RAW {100 * latest.raw_anomaly_probs[1]:3.0f}% · TRIP {100 * threshold:.0f}%",
              tone=_prob_tone(p_trm, threshold, AlertLevel.TREMOR), meter=p_trm,
              tick=threshold),
        _tile("Inference latency", f"{model_p50:4.1f}<span class='unit'>ms</span>",
              f"E2E P99 {e2e_p99:5.1f} MS · BUDGET {LATENCY_BUDGET_MS:.0f}",
              tone="t-green" if e2e_p99 < LATENCY_BUDGET_MS else "t-red",
              meter=e2e_p99 / LATENCY_BUDGET_MS),
        _tile("Tip speed", f"{latest.velocity:5.1f}<span class='unit'>mm/s</span>",
              f"GRIP {latest.grip:4.2f} RAD · FRAME {latest.frame_idx}",
              tone="t-cyan", meter=latest.velocity / 40.0),
    ]
    st.markdown(f'<div class="tiles">{"".join(tiles)}</div>', unsafe_allow_html=True)


def render_event_log(snap: ServiceSnapshot) -> None:
    events = list(reversed(snap.alert_events))[:8]
    if not events:
        st.markdown('<div class="empty">NO ALERT EPISODES RECORDED</div>',
                    unsafe_allow_html=True)
        return
    leds = {AlertLevel.HESITATION: ("red", "HESITATION"), AlertLevel.TREMOR: ("amber", "TREMOR")}
    rows = []
    for e in events:
        led, name = leds.get(e.level, ("", e.level.name))
        duration = (f"{e.duration:5.1f} s" if e.duration is not None
                    else '<span class="live">ACTIVE</span>')
        rows.append(f'<tr><td>{e.start:7.1f} s</td><td><span class="led {led}"></span>{name}'
                    f'</td><td>{duration}</td><td>{100 * e.peak_prob:3.0f}%</td></tr>')
    st.markdown('<table class="evlog"><thead><tr><th>T+</th><th>CHANNEL</th><th>DURATION</th>'
                f'<th>PEAK P</th></tr></thead><tbody>{"".join(rows)}</tbody></table>',
                unsafe_allow_html=True)


def render_validation(snap: ServiceSnapshot, window_s: int) -> None:
    """Live model-vs-truth tiles over the visible window (possible only in simulation)."""
    now = snap.latest.sim_time if snap.latest else 0.0
    preds = [p for p in snap.predictions if p.sim_time >= now - window_s]
    if not preds:
        return
    gesture_acc = float(np.mean([int(p.gesture) == p.true_gesture for p in preds]))

    def recall_precision(level: AlertLevel, anomaly: Anomaly) -> tuple[str, str]:
        truth = np.array([p.true_anomaly == int(anomaly) for p in preds])
        fired = np.array([p.alert == level for p in preds])
        rec = f"{100 * (fired & truth).sum() / truth.sum():.0f}" if truth.any() else "—"
        prec = f"{100 * (fired & truth).sum() / fired.sum():.0f}" if fired.any() else "—"
        return rec, prec

    h_rec, h_prec = recall_precision(AlertLevel.HESITATION, Anomaly.HESITATION)
    t_rec, t_prec = recall_precision(AlertLevel.TREMOR, Anomaly.TREMOR)
    tiles = [
        _tile("Gesture accuracy", f"{100 * gesture_acc:.0f}<span class='unit'>%</span>",
              f"{len(preds)} FRAMES", tone="t-cyan", meter=gesture_acc),
        _tile("Hesitation R / P", f"{h_rec} / {h_prec}", "RECALL / PRECISION %"),
        _tile("Tremor R / P", f"{t_rec} / {t_prec}", "RECALL / PRECISION %"),
    ]
    st.markdown(f'<div class="tiles three">{"".join(tiles)}</div>'
                f'<div class="footnote">Frame-level over the last {window_s} s. Alerts are '
                'debounced, so onset frames count as misses; episode-level detection is '
                'higher.</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Live console (fragment body)
# --------------------------------------------------------------------------------------


def render_console(service: LiveInferenceService, settings: Settings) -> None:
    """The live region. Wrapped in ``st.fragment(run_every=…)`` by the caller."""
    snap = service.snapshot()
    render_topbar(snap, service, settings)

    if snap.error:
        st.markdown('<div class="annunciator ann-err"><div class="ann-body">'
                    '<div class="ann-code">SYSTEM FAULT</div>'
                    '<div class="ann-headline">INFERENCE SERVICE STOPPED</div>'
                    f'<div class="ann-detail">{html.escape(snap.error)}</div></div></div>',
                    unsafe_allow_html=True)
        return
    latest = snap.latest
    if latest is None:
        warm = min(snap.frames_received / service.engine.window_size, 1.0)
        st.markdown('<div class="boot">INITIALIZING · BUFFERING 2.0 S ROLLING WINDOW · '
                    f'{100 * warm:.0f}%<div class="bar"><div style="width:{100 * warm:.0f}%">'
                    '</div></div></div>', unsafe_allow_html=True)
        return

    threshold = InferenceConfig.thresholds_for_sensitivity(settings.sensitivity)
    render_annunciator(latest.alert)
    render_tiles(snap, latest, threshold)

    left, right = st.columns([3, 2], gap="small")
    with left:
        with st.container(key="card_trajectory"):
            _card_head("Instrument tip · 3D trajectory",
                       f"{min(settings.trail_frames, len(snap.predictions))} FR")
            st.plotly_chart(trajectory_figure(snap.predictions, settings.trail_frames),
                            width="stretch", theme=None, config=PLOTLY_CONFIG,
                            key="trajectory")
    with right:
        with st.container(key="card_classifier"):
            _card_head("Classifier output", "EMA-SMOOTHED")
            st.plotly_chart(probability_figure(latest, threshold), width="stretch",
                            theme=None, config=PLOTLY_CONFIG, key="probabilities")
        with st.container(key="card_kinematics"):
            _card_head("Kinematic trace", "30 HZ")
            st.plotly_chart(kinematics_figure(snap.predictions, settings.trail_frames,
                                              threshold),
                            width="stretch", theme=None, config=PLOTLY_CONFIG,
                            key="kinematics")

    with st.container(key="card_timeline"):
        _card_head("Surgical workflow timeline", f"LAST {settings.timeline_seconds} S")
        chips = "".join(f'<span style="--c:{GESTURE_COLORS[g]}">{g.name}</span>'
                        for g in Gesture)
        chips += "".join(f'<span style="--c:{ALERT_COLORS[a]}">{a.name}</span>'
                         for a in (AlertLevel.TREMOR, AlertLevel.HESITATION))
        st.markdown(f'<div class="legend-chips">{chips}</div>', unsafe_allow_html=True)
        st.plotly_chart(timeline_figure(snap, settings.timeline_seconds, settings.show_truth),
                        width="stretch", theme=None, config=PLOTLY_CONFIG, key="timeline")

    if settings.show_truth:
        log_col, val_col = st.columns([2, 3], gap="small")
    else:
        log_col, val_col = st.container(), None
    with log_col:
        with st.container(key="card_events"):
            _card_head("Alert event log", f"{len(snap.alert_events)} EPISODES")
            render_event_log(snap)
    if val_col is not None:
        with val_col:
            with st.container(key="card_validation"):
                _card_head("Validation · simulator ground truth", "SIM ONLY")
                render_validation(snap, settings.timeline_seconds)


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Surgeon Console", page_icon="◈", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    settings = render_sidebar()
    if not Path(settings.checkpoint).exists():
        st.markdown('<div class="annunciator ann-err"><div class="ann-body">'
                    '<div class="ann-code">SYSTEM FAULT</div>'
                    '<div class="ann-headline">NO TRAINED MODEL FOUND</div>'
                    f'<div class="ann-detail">Expected checkpoint at '
                    f'{html.escape(settings.checkpoint)}. Run: python train.py --epochs 25'
                    '</div></div></div>', unsafe_allow_html=True)
        st.stop()

    service = get_service(settings.checkpoint, settings.seed)
    if settings.reset:
        service.stop()
        get_service.clear()
        st.rerun()
    render_system_readout(service)

    # Sidebar values are applied live to the running service (no restart needed).
    service.set_difficulty(settings.difficulty)
    service.set_sensitivity(settings.sensitivity)
    service.set_smoothing(settings.ema_alpha)
    if settings.streaming and not service.running:
        service.start()
    elif not settings.streaming and service.running:
        service.stop()

    interval = 1.0 / settings.refresh_hz if settings.streaming else None

    @st.fragment(run_every=interval)
    def live_console() -> None:
        render_console(service, settings)

    live_console()


if __name__ == "__main__":
    main()