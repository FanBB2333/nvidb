"""
Dash web interface for nvidb (Live GPU + Log viewer).

Usage:
    nvidb web                 # Web dashboard (Live + Logs)

Replaces the previous Streamlit UI; the data layer in `nvidb.web` is reused.
"""

import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import dash
    from dash import (
        ALL,
        Dash,
        Input,
        Output,
        State,
        ctx,
        dash_table,
        dcc,
        html,
        no_update,
    )
    from dash.exceptions import PreventUpdate
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    dash = None

try:
    from nvidb import config
except ImportError:  # pragma: no cover
    from . import config

try:
    from nvidb.metrics import ADVANCED_METRIC_GROUPS, ADVANCED_METRIC_LABELS, present_columns
except ImportError:  # pragma: no cover
    from .metrics import ADVANCED_METRIC_GROUPS, ADVANCED_METRIC_LABELS, present_columns

# Reuse the UI-agnostic data layer from the legacy web module.
try:
    from nvidb.web import (
        _LOG_METRICS,
        _as_path,
        _build_log_snapshot_table,
        _downsample_per_gpu,
        _format_datetime,
        _format_duration,
        _format_gb,
        _format_mib,
        _load_server_list,
        _parse_mib_pair,
        _parse_percent,
        _server_summary,
        _strip_gpu_name,
        _user_memory_from_df,
        _user_summary_df,
        _user_time_share_df,
        get_db_path,
        load_session_logs,
        load_sessions,
    )
except ImportError:  # pragma: no cover
    from .web import (
        _LOG_METRICS,
        _as_path,
        _build_log_snapshot_table,
        _downsample_per_gpu,
        _format_datetime,
        _format_duration,
        _format_gb,
        _format_mib,
        _load_server_list,
        _parse_mib_pair,
        _parse_percent,
        _server_summary,
        _strip_gpu_name,
        _user_memory_from_df,
        _user_summary_df,
        _user_time_share_df,
        get_db_path,
        load_session_logs,
        load_sessions,
    )


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

ACCENT = "#00BFA5"
CHART_COLORS = [
    "#00BFA5",
    "#00ACC1",
    "#1E88E5",
    "#7C4DFF",
    "#EC407A",
    "#FFB300",
    "#8BC34A",
    "#FF7043",
]

# Server-side palettes (plotly figures are rendered server-side, so figure
# colors come from here; everything HTML/CSS uses the CSS variables below).
_PALETTES = {
    "dark": {
        "text": "#e2e8f0",
        "muted": "#8fa3bf",
        "grid": "rgba(148,163,184,0.14)",
        "border": "rgba(148,163,184,0.16)",
        "hoverbg": "#0e1626",
        "good": "#34d399",
        "warn": "#fbbf24",
        "bad": "#f87171",
    },
    "light": {
        "text": "#0f172a",
        "muted": "#5b6b81",
        "grid": "rgba(15,23,42,0.08)",
        "border": "rgba(15,23,42,0.14)",
        "hoverbg": "#ffffff",
        "good": "#059669",
        "warn": "#b45309",
        "bad": "#dc2626",
    },
}


def _pal(theme):
    return _PALETTES.get(str(theme or "dark"), _PALETTES["dark"])


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# The page itself is themed with CSS variables; `data-theme` on <html> flips
# them. Dark is the default (no attribute).
_CSS = """
/* `html:root` (0,1,1) deliberately outranks the `:root` (0,1,0) blocks that
   dcc/dash_table inject at runtime, so our theme tokens always win. The
   `--Dash-*` names are dash 4's own design tokens (dropdowns etc. read them);
   overriding them themes the components without fighting their selectors. */
html:root {
  color-scheme: dark;
  --nv-bg: #0b1220;
  --nv-panel: #101a2c;
  --nv-panel2: #0e1626;
  --nv-border: rgba(148,163,184,0.16);
  --nv-text: #e2e8f0;
  --nv-muted: #8fa3bf;
  --nv-good: #34d399;
  --nv-bad: #f87171;
  --nv-accent: %(accent)s;
  --nv-accent-soft: rgba(0,191,165,0.14);
  --nv-shadow: 0 10px 28px rgba(0,0,0,0.45);

  --Dash-Fill-Inverse-Strong: var(--nv-panel2);
  --Dash-Fill-Disabled: rgba(148,163,184,0.15);
  --Dash-Fill-Interactive-Strong: var(--nv-accent);
  --Dash-Fill-Interactive-Weak: rgba(0,191,165,0.12);
  --Dash-Fill-Primary-Hover: rgba(0,191,165,0.10);
  --Dash-Fill-Primary-Active: rgba(0,191,165,0.16);
  --Dash-Stroke-Strong: rgba(148,163,184,0.35);
  --Dash-Stroke-Weak: rgba(148,163,184,0.16);
  --Dash-Text-Strong: var(--nv-text);
  --Dash-Text-Primary: var(--nv-text);
  --Dash-Text-Weak: var(--nv-muted);
  --Dash-Text-Disabled: rgba(143,163,191,0.6);
  --Dash-Shading-Strong: rgba(0,0,0,0.55);
  --Dash-Shading-Weak: rgba(0,0,0,0.35);
}
html[data-theme="light"] {
  color-scheme: light;
  --nv-bg: #f4f6fa;
  --nv-panel: #ffffff;
  --nv-panel2: #eef2f7;
  --nv-border: rgba(15,23,42,0.14);
  --nv-text: #0f172a;
  --nv-muted: #5b6b81;
  --nv-good: #059669;
  --nv-bad: #dc2626;
  --nv-accent-soft: rgba(0,191,165,0.18);
  --nv-shadow: 0 10px 28px rgba(15,23,42,0.12);

  --Dash-Fill-Inverse-Strong: #ffffff;
  --Dash-Fill-Disabled: rgba(0,24,102,0.1);
  --Dash-Fill-Interactive-Strong: var(--nv-accent);
  --Dash-Fill-Interactive-Weak: rgba(0,191,165,0.10);
  --Dash-Fill-Primary-Hover: rgba(0,18,77,0.04);
  --Dash-Fill-Primary-Active: rgba(0,18,77,0.08);
  --Dash-Stroke-Strong: rgba(0,18,77,0.35);
  --Dash-Stroke-Weak: rgba(0,24,102,0.1);
  --Dash-Text-Strong: rgba(0,9,38,0.9);
  --Dash-Text-Primary: rgba(0,18,77,0.87);
  --Dash-Text-Weak: rgba(0,12,51,0.65);
  --Dash-Text-Disabled: rgba(0,21,89,0.6);
  --Dash-Shading-Strong: rgba(22,23,24,0.35);
  --Dash-Shading-Weak: rgba(22,23,24,0.2);
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--nv-bg); color: var(--nv-text);
  font-family: "Inter", -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 14px;
  transition: background 0.2s ease;
}
._dash-loading { color: var(--nv-muted); padding: 24px; }
.nv-shell { max-width: 1500px; margin: 0 auto; padding: 18px 22px 60px; }
.nv-header { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; margin-bottom: 6px; }
.nv-brand { font-size: 22px; font-weight: 700; letter-spacing: 0.3px; }
.nv-brand b { color: var(--nv-accent); }
.nv-sub { color: var(--nv-muted); font-size: 12px; }
.nv-card {
  background: var(--nv-panel); border: 1px solid var(--nv-border); border-radius: 12px;
  padding: 14px 16px; margin: 10px 0;
}
.nv-card-tight { padding: 10px 12px; }
.nv-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.nv-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 12px; }
.nv-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.nv-stat { background: var(--nv-panel2); border: 1px solid var(--nv-border); border-radius: 10px; padding: 10px 12px; }
.nv-stat .k { color: var(--nv-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; }
.nv-stat .v { font-size: 20px; font-weight: 650; margin-top: 2px; }
.nv-stat .s { color: var(--nv-muted); font-size: 11px; margin-top: 2px; }
.nv-server-title { font-size: 15px; font-weight: 650; }
.nv-caption { color: var(--nv-muted); font-size: 12px; }
.nv-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
  border: 1px solid var(--nv-border); color: var(--nv-muted); background: var(--nv-panel2); margin-left: 8px;
}
.nv-badge-ok { color: var(--nv-good); border-color: var(--nv-good); }
.nv-badge-err { color: var(--nv-bad); border-color: var(--nv-bad); }
.nv-bar-wrap { background: var(--nv-border); border-radius: 6px; height: 8px; width: 100%%; margin-top: 6px; }
.nv-bar { height: 8px; border-radius: 6px; }
.nv-controls label { color: var(--nv-muted); font-size: 12px; display: block; margin-bottom: 4px; }
.nv-control { min-width: 150px; }
.nv-flex1 { flex: 1; }
a { color: var(--nv-accent); }
summary { color: var(--nv-muted); }

input[type="text"], input[type="number"] {
  background: var(--nv-panel2); color: var(--nv-text); border: 1px solid var(--nv-border);
  border-radius: 8px; padding: 6px 10px;
}
button.nv-btn {
  background: var(--nv-accent-soft); color: var(--nv-accent); border: 1px solid var(--nv-accent);
  border-radius: 8px; padding: 6px 14px; cursor: pointer; font-size: 13px;
}
button.nv-btn:hover { filter: brightness(1.15); }

/* ---- Tabs ---- */
.nv-tabs .tab { background: transparent !important; border: none !important; color: var(--nv-muted) !important;
  padding: 10px 18px !important; font-size: 14px; border-bottom: 2px solid transparent !important; }
.nv-tabs .tab--selected { color: var(--nv-text) !important; border-bottom: 2px solid var(--nv-accent) !important; font-weight: 600; }
.nv-tabs { border-bottom: 1px solid var(--nv-border); margin-bottom: 8px; }

/* ---- checklist / radio ---- */
.nv-check label { color: var(--nv-text); margin-right: 14px; font-size: 13px; }
.nv-check input { accent-color: var(--nv-accent); margin-right: 5px; }

/* ---- dcc.Dropdown ---- */
/* Colors come from the --Dash-* tokens above; only shape tweaks here. */
.dash-dropdown, .dash-dropdown-content { border-radius: 8px; }
.dash-dropdown-option:hover { background: var(--nv-accent-soft); }

/* ---- dash_table.DataTable ---- */
html .dash-table-container { --accent: %(accent)s; --hover: var(--nv-accent-soft); --text-color: var(--nv-text); }
/* dash_table hard-codes --hover to #fdfdfd in its own (later-loaded) sheet;
   force row hover to the accent tint instead of near-white. */
.dash-table-container .dash-spreadsheet-inner tr:hover { background-color: transparent !important; }
.dash-table-container .dash-spreadsheet-inner tr:hover td.dash-cell { background-color: var(--nv-accent-soft) !important; }
.dash-spreadsheet-container .dash-spreadsheet-inner table { background-color: transparent !important; }
.dash-spreadsheet-container .dash-spreadsheet-inner th {
  background-color: var(--nv-panel2) !important; color: var(--nv-muted) !important; border: none !important;
}
.dash-spreadsheet-container .dash-spreadsheet-inner td {
  background-color: transparent !important; border: none !important;
  border-bottom: 1px solid var(--nv-border) !important; color: var(--nv-text);
}
.dash-spreadsheet-container .dash-spreadsheet-inner td.focused {
  background-color: var(--nv-accent-soft) !important;
}
.dash-spreadsheet-container input.dash-cell-value,
.dash-spreadsheet-container .dash-cell-value {
  background-color: transparent !important; color: inherit !important;
}
.dash-spreadsheet-container .dash-filter { background-color: var(--nv-panel2) !important; }
.dash-spreadsheet-container .dash-filter input { color: var(--nv-text) !important; }
.dash-table-tooltip {
  background-color: var(--nv-panel2) !important; color: var(--nv-text) !important;
  border: 1px solid var(--nv-border); border-radius: 8px; box-shadow: var(--nv-shadow);
}
.dash-table-container .previous-next-container { color: var(--nv-muted) !important; }
.dash-table-container .previous-next-container button {
  background: transparent !important; color: var(--nv-muted) !important; border: none; cursor: pointer;
}
.dash-table-container .previous-next-container button:hover { color: var(--nv-accent) !important; }
.dash-table-container input.current-page, .dash-table-container .current-page {
  background: transparent !important; color: var(--nv-text) !important;
  border: none !important; border-bottom: 1px solid var(--nv-border) !important;
}
.dash-table-container button.export {
  background: var(--nv-accent-soft) !important; color: var(--nv-accent) !important;
  border: 1px solid var(--nv-accent) !important; border-radius: 8px !important;
  padding: 4px 12px !important; cursor: pointer; margin-bottom: 6px;
}
.dash-table-container .column-header--sort { color: var(--nv-muted) !important; }

/* ---- sliders ---- */
.rc-slider-rail { background: var(--nv-border); }
.rc-slider-track { background: var(--nv-accent); }
.rc-slider-handle { border-color: var(--nv-accent); background: var(--nv-accent); }
.rc-slider-mark-text { color: var(--nv-muted); }
""" % {"accent": ACCENT}

_INDEX_STRING = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>nvidb web</title>
    {%favicon%}
    {%css%}
    <style>__NV_CSS__</style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>""".replace("__NV_CSS__", _CSS)


def _fig_layout(fig, pal, *, title=None, height=240, ymax=None, unit=None):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=pal["text"]), x=0, xanchor="left") if title else None,
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=pal["muted"], size=11),
        height=height,
        margin=dict(l=42, r=12, t=34 if title else 12, b=28),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=pal["hoverbg"], font=dict(color=pal["text"], size=11), bordercolor=pal["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1.0, xanchor="right", font=dict(size=10)),
        colorway=CHART_COLORS,
    )
    fig.update_xaxes(gridcolor=pal["grid"], zeroline=False, linecolor=pal["border"], tickformat="%H:%M:%S")
    fig.update_yaxes(gridcolor=pal["grid"], zeroline=False, linecolor=pal["border"], rangemode="tozero")
    if ymax is not None:
        fig.update_yaxes(range=[0, ymax])
    if unit:
        fig.update_yaxes(ticksuffix=unit)
    return fig


_GRAPH_CONFIG = {"displayModeBar": "hover", "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ---------------------------------------------------------------------------
# Live data cache (background fetcher + rolling history)
# ---------------------------------------------------------------------------

HISTORY_MAX_POINTS = 360  # rolling live history per GPU


class LiveCache:
    """Background fetcher over NVClientPool with a rolling metric history."""

    def __init__(self, include_remote: bool, *, interval_seconds: int = 5):
        try:
            from nvidb.connection import NVClientPool
        except ImportError:  # pragma: no cover
            from .connection import NVClientPool

        self.include_remote = bool(include_remote)
        self.remote_error = None
        server_list = None
        if include_remote:
            try:
                server_list = _load_server_list()
            except Exception as e:
                self.remote_error = str(e)
        self.pool = NVClientPool(server_list)

        self._lock = threading.Lock()
        self._interval = max(0.25, float(interval_seconds))
        self._enabled = True
        self.raw = None
        self.last_updated = None
        self.last_error = None
        self.last_duration = None
        self.fetching = False
        self.history = deque(maxlen=HISTORY_MAX_POINTS)  # [{ts, rows: [{server, gpu, util, mem_used, mem_total, power, temp}]}]

        self._stop = threading.Event()
        self._kick = threading.Event()
        self._kick.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def configure(self, *, interval_seconds=None, enabled=None):
        with self._lock:
            if interval_seconds is not None:
                # Sub-second intervals are allowed; the fetch loop is serial, so
                # the effective period is max(interval, fetch duration).
                self._interval = max(0.25, min(3600.0, float(interval_seconds)))
            if enabled is not None:
                self._enabled = bool(enabled)

    def refresh_now(self):
        self._kick.set()

    def snapshot(self):
        with self._lock:
            return {
                "raw": self.raw,
                "last_updated": self.last_updated,
                "last_error": self.last_error,
                "last_duration": self.last_duration,
                "fetching": self.fetching,
                "history": list(self.history),
                "remote_error": self.remote_error,
            }

    def _record_history(self, raw, now):
        rows = []
        if not isinstance(raw, dict):
            return
        for idx, client in enumerate(self.pool.pool):
            entry = raw.get(idx)
            if not entry:
                continue
            table, _info = entry
            if not isinstance(table, pd.DataFrame) or table.empty:
                continue
            desc = getattr(client, "description", f"server {idx}")
            for _, r in table.iterrows():
                used, total = _parse_mib_pair(r.get("memory[used/total]"))
                power_txt = str(r.get("power", ""))
                power_w = None
                # formats like "P1 114.86/300.00" or "134.68 W"
                for tok in power_txt.replace("/", " ").split():
                    try:
                        power_w = float(tok)
                        break
                    except ValueError:
                        continue
                temp_txt = str(r.get("temp", "")).replace("C", "").strip()
                try:
                    temp_c = float(temp_txt)
                except ValueError:
                    temp_c = None
                rows.append(
                    {
                        "server": desc,
                        "gpu": f"GPU {r.get('GPU', '?')}",
                        "util": _parse_percent(r.get("util")),
                        "mem_used_gib": (used / 1024.0) if used is not None else None,
                        "mem_total_gib": (total / 1024.0) if total is not None else None,
                        "power_w": power_w,
                        "temp_c": temp_c,
                    }
                )
        if rows:
            self.history.append({"ts": now, "rows": rows})

    def _run(self):  # pragma: no cover - thread loop
        # `interval` is a target *period*: the fetch duration is subtracted from
        # the wait, so sub-second intervals refresh as fast as fetches allow.
        while not self._stop.is_set():
            with self._lock:
                enabled = self._enabled
                interval = self._interval

            if not enabled:
                self._kick.wait()
                self._kick.clear()
                if self._stop.is_set():
                    break
                continue

            start = time.monotonic()
            with self._lock:
                self.fetching = True
            try:
                _fmt, raw = self.pool.get_client_gpus_info(return_raw=True)
                now = datetime.now()
                with self._lock:
                    self.raw = raw
                    self.last_updated = now
                    self.last_error = None
                    self._record_history(raw, now)
            except Exception as e:
                with self._lock:
                    self.last_error = str(e)
            finally:
                elapsed = time.monotonic() - start
                with self._lock:
                    self.last_duration = elapsed
                    self.fetching = False

            if self._stop.is_set():
                break
            self._kick.wait(timeout=max(0.05, interval - elapsed))
            self._kick.clear()


_live_caches = {}
_live_lock = threading.Lock()


def _get_live_cache(include_remote: bool) -> LiveCache:
    key = bool(include_remote)
    with _live_lock:
        cache = _live_caches.get(key)
        if cache is None:
            cache = LiveCache(key)
            _live_caches[key] = cache
        return cache


# ---------------------------------------------------------------------------
# Logs data (TTL cache)
# ---------------------------------------------------------------------------

_logs_cache = {}
_logs_lock = threading.Lock()


def _cached(key, ttl, producer):
    now = time.monotonic()
    with _logs_lock:
        hit = _logs_cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = producer()
    with _logs_lock:
        _logs_cache[key] = (now, value)
    return value


def _sessions_df(db_path) -> pd.DataFrame:
    return _cached(("sessions", str(db_path)), 5, lambda: load_sessions(db_path))


def _session_logs_df(db_path, session_id) -> pd.DataFrame:
    return _cached(("logs", str(db_path), int(session_id)), 5, lambda: load_session_logs(db_path, int(session_id)))


# ---------------------------------------------------------------------------
# Small component builders
# ---------------------------------------------------------------------------


def _stat(label, value, sub=None):
    children = [html.Div(label, className="k"), html.Div(value, className="v")]
    if sub:
        children.append(html.Div(sub, className="s"))
    return html.Div(children, className="nv-stat")


def _pct_color(p, pal):
    if p is None:
        return pal["muted"]
    if p >= 80:
        return pal["bad"]
    if p >= 40:
        return pal["warn"]
    return pal["good"]


def _bar(pct, pal, color=None):
    p = max(0.0, min(100.0, float(pct or 0)))
    return html.Div(
        html.Div(className="nv-bar", style={"width": f"{p:.0f}%", "background": color or _pct_color(p, pal)}),
        className="nv-bar-wrap",
    )


# util/mem_util data bars via background-image so the theme CSS (which only
# overrides background-color) never wipes them.
_PERCENT_BUCKET_STYLES = []
for lo in range(0, 100, 10):
    hi = lo + 10
    alpha = 0.10 + lo / 100 * 0.45
    for col in ("util", "mem_util"):
        _PERCENT_BUCKET_STYLES.append(
            {
                "if": {
                    "column_id": col,
                    "filter_query": f"{{_{col}_num}} >= {lo} && {{_{col}_num}} < {hi if hi < 100 else 101}",
                },
                "background": (
                    f"linear-gradient(90deg, rgba(0,191,165,{alpha:.2f}) 0%, "
                    f"rgba(0,191,165,{alpha:.2f}) {hi}%, transparent {hi}%)"
                ),
            }
        )

# Structural-only styles: all colors come from the theme CSS.
_TABLE_STYLE = dict(
    style_as_list_view=True,
    style_table={"overflowX": "auto"},
    style_header={"fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "0.5px", "fontWeight": "600"},
    style_cell={
        "fontSize": "13px",
        "padding": "6px 10px",
        "textAlign": "center",
        "maxWidth": 340,
        "overflow": "hidden",
        "textOverflow": "ellipsis",
        "fontFamily": "inherit",
    },
)


def _gpu_datatable(table: pd.DataFrame, table_id=None):
    display = table.copy()
    for col in ("util", "mem_util"):
        if col in display.columns:
            display[f"_{col}_num"] = display[col].map(lambda v: _parse_percent(v) if v is not None else None)
    visible = [c for c in display.columns if not c.startswith("_")]
    id_kwargs = {"id": table_id} if table_id is not None else {}
    return dash_table.DataTable(
        **id_kwargs,
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in visible],
        sort_action="native",
        cell_selectable=False,
        style_data_conditional=_PERCENT_BUCKET_STYLES,
        tooltip_data=[
            {c: {"value": str(row.get(c, "")), "type": "text"} for c in visible}
            for row in display.to_dict("records")
        ],
        tooltip_duration=None,
        **_TABLE_STYLE,
    )


def _user_vram_fig(user_summary: dict, pal, *, title):
    df = _user_summary_df(user_summary)
    if df.empty:
        return None
    df = df.head(15)
    fig = go.Figure(
        go.Bar(
            x=df["vram_mib"] / 1024.0,
            y=df["user"],
            orientation="h",
            marker=dict(color=ACCENT, opacity=0.85),
            text=df["vram"],
            textposition="outside",
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=None, ticksuffix=" GB")
    return _fig_layout(fig, pal, title=title, height=max(120, 44 + 26 * len(df)))


# ---------------------------------------------------------------------------
# Live tab
# ---------------------------------------------------------------------------


def _empty_fig(pal, message="No data"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=pal["muted"], size=12))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _fig_layout(fig, pal, height=160)


def _live_history_figs(history, server_desc, pal):
    """Rolling util% + VRAM charts for one server."""
    util_fig = go.Figure()
    mem_fig = go.Figure()
    series = {}
    for point in history:
        ts = point["ts"]
        for row in point["rows"]:
            if row["server"] != server_desc:
                continue
            key = row["gpu"]
            s = series.setdefault(key, {"ts": [], "util": [], "mem": []})
            s["ts"].append(ts)
            s["util"].append(row["util"])
            s["mem"].append(row["mem_used_gib"])
    if not series:
        return None, None
    mem_total = None
    for point in reversed(history):
        for row in point["rows"]:
            if row["server"] == server_desc and row.get("mem_total_gib"):
                mem_total = row["mem_total_gib"]
                break
        if mem_total:
            break
    for i, (gpu, s) in enumerate(sorted(series.items())):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        util_fig.add_trace(
            go.Scatter(x=s["ts"], y=s["util"], name=gpu, mode="lines", line=dict(width=2, color=color),
                       hovertemplate="%{y:.0f}%<extra>" + gpu + "</extra>")
        )
        mem_fig.add_trace(
            go.Scatter(x=s["ts"], y=s["mem"], name=gpu, mode="lines", line=dict(width=2, color=color),
                       fill="tozeroy", fillcolor=_rgba(color, 0.10),
                       hovertemplate="%{y:.1f} GiB<extra>" + gpu + "</extra>")
        )
    _fig_layout(util_fig, pal, title="GPU Util (rolling)", height=200, ymax=100, unit="%")
    _fig_layout(mem_fig, pal, title="VRAM Used (rolling)", height=200, unit=" GiB")
    if mem_total:
        mem_fig.update_yaxes(range=[0, mem_total * 1.05])
    # Keep user zoom/legend state across periodic figure updates.
    util_fig.update_layout(uirevision=f"util-{server_desc}")
    mem_fig.update_layout(uirevision=f"mem-{server_desc}")
    return util_fig, mem_fig


def _live_signature(cache: LiveCache):
    """Structural fingerprint of the live view. The DOM skeleton is rebuilt only
    when this changes; every refresh tick merely updates data/figure props so
    scroll position, plot zoom and open <details> panels survive."""
    snap = cache.snapshot()
    raw = snap["raw"]
    servers = []
    for idx, client in enumerate(cache.pool.pool):
        desc = getattr(client, "description", f"server {idx}")
        entry = (raw.get(idx) if isinstance(raw, dict) else None) or (pd.DataFrame(), {})
        table, info = entry
        info = info if isinstance(info, dict) else {}
        if info.get("error"):
            servers.append({"desc": desc, "kind": "error", "cols": [], "adv": False})
        elif isinstance(table, pd.DataFrame) and not table.empty:
            servers.append(
                {
                    "desc": desc,
                    "kind": "gpus",
                    "cols": [str(c) for c in table.columns],
                    "adv": info.get("advanced_supported") is True,
                }
            )
        else:
            servers.append({"desc": desc, "kind": "empty", "cols": [], "adv": False})
    return {"remote": cache.include_remote, "fetched": raw is not None, "servers": servers}


def _build_live_skeleton(sig):
    """Static DOM for the live view; ids are targeted by the update callback."""
    if not sig.get("fetched"):
        return [html.Div("Fetching GPU data…", className="nv-card nv-caption"), html.Div(id="live-alerts")]

    children = [
        html.Div(id="live-alerts"),
        html.Div(id="live-overview", className="nv-stats"),
        html.Div(
            dcc.Graph(id="live-global-vram", config=_GRAPH_CONFIG),
            id="live-global-wrap",
            className="nv-card nv-card-tight",
            style={"display": "none"},
        ),
    ]
    for idx, s in enumerate(sig.get("servers") or []):
        body = [
            html.Div(id={"type": "live-head", "index": idx}),
            html.Div(id={"type": "live-caption", "index": idx}, className="nv-caption", style={"margin": "6px 0 10px"}),
        ]
        if s["kind"] == "gpus":
            body.append(
                dash_table.DataTable(
                    id={"type": "live-table", "index": idx},
                    columns=[{"name": c, "id": c} for c in s["cols"]],
                    data=[],
                    sort_action="native",
                    cell_selectable=False,
                    style_data_conditional=_PERCENT_BUCKET_STYLES,
                    tooltip_duration=None,
                    **_TABLE_STYLE,
                )
            )
            body.append(
                html.Div(
                    [
                        html.Div(dcc.Graph(id={"type": "live-util", "index": idx}, config=_GRAPH_CONFIG),
                                 className="nv-flex1", style={"minWidth": "320px"}),
                        html.Div(dcc.Graph(id={"type": "live-mem", "index": idx}, config=_GRAPH_CONFIG),
                                 className="nv-flex1", style={"minWidth": "320px"}),
                    ],
                    className="nv-row",
                    style={"marginTop": "8px"},
                )
            )
            if s["adv"]:
                body.append(
                    html.Details(
                        [html.Summary("Advanced (DCGM)", className="nv-caption", style={"cursor": "pointer"}),
                         html.Div(id={"type": "live-adv", "index": idx})],
                        style={"marginTop": "8px"},
                    )
                )
            body.append(
                html.Details(
                    [html.Summary("User VRAM totals", className="nv-caption", style={"cursor": "pointer"}),
                     dcc.Graph(id={"type": "live-uservram", "index": idx}, config=_GRAPH_CONFIG)],
                    style={"marginTop": "8px"},
                )
            )
        children.append(html.Div(body, className="nv-card"))
    return children


def _live_update_payload(cache: LiveCache, sig, theme):
    """Per-tick prop values for the skeleton built from `sig`."""
    pal = _pal(theme)
    snap = cache.snapshot()
    raw = snap["raw"] if isinstance(snap["raw"], dict) else {}
    history = snap["history"]

    status_bits = []
    if snap["fetching"]:
        status_bits.append("Updating…")
    if snap["last_updated"]:
        status_bits.append(f"Last updated: {snap['last_updated'].strftime('%H:%M:%S')}")
    if snap["last_duration"] is not None:
        status_bits.append(f"fetch {snap['last_duration']:.1f}s")
    status = " | ".join(status_bits) or "Fetching GPU data…"

    alerts = []
    if snap["remote_error"]:
        alerts.append(html.Div(f"Remote disabled: {snap['remote_error']}", className="nv-card", style={"color": pal["bad"]}))
    if snap["last_error"]:
        alerts.append(html.Div(f"Last fetch error: {snap['last_error']}", className="nv-card", style={"color": pal["warn"]}))

    meta = raw.get("_nvidb", {}) or {}
    user_mem_by_client = meta.get("user_memory_by_client", {}) or {}
    global_user_mem = meta.get("user_memory_global", {}) or {}

    payload = {
        "alerts": alerts,
        "status": status,
        "overview": [],
        "heads": [],
        "captions": [],
        "tables": [],
        "tooltips": [],
        "util_figs": [],
        "mem_figs": [],
        "adv_children": [],
        "user_figs": [],
    }

    global_fig = _user_vram_fig(global_user_mem, pal, title="User VRAM totals (all nodes)")
    payload["global_fig"] = global_fig or _empty_fig(pal)
    payload["global_style"] = {} if global_fig is not None else {"display": "none"}

    for idx, s in enumerate(sig.get("servers") or []):
        desc = s["desc"]
        entry = raw.get(idx) or (pd.DataFrame(), {})
        table, info = entry
        info = info if isinstance(info, dict) else {}
        sys_stats = info.get("system_stats", {}) or {}

        # ---- head + caption (every server) ----
        head = [html.Span(f"[{idx + 1}] {desc}", className="nv-server-title")]
        if info.get("error"):
            head.append(html.Span("error", className="nv-badge nv-badge-err"))
            caption = html.Span(str(info.get("error")), style={"color": pal["bad"]})
        else:
            src = info.get("data_source")
            if src:
                head.append(html.Span(f"src:{src}", className="nv-badge"))
            adv_src = info.get("advanced_source")
            if adv_src:
                adv_ok = info.get("advanced_supported") is True
                head.append(
                    html.Span(
                        f"advanced:{adv_src}" + ("" if adv_ok else " (profiling unavailable)"),
                        className="nv-badge " + ("nv-badge-ok" if adv_ok else ""),
                    )
                )
            parts = []
            if any(k in info for k in ("driver_version", "cuda_version", "attached_gpus")):
                parts.append(
                    f"Driver {info.get('driver_version', 'N/A')} | CUDA {info.get('cuda_version', 'N/A')} | GPUs {info.get('attached_gpus', '0')}"
                )
            summary = _server_summary(table if isinstance(table, pd.DataFrame) else pd.DataFrame(), info)
            if summary:
                parts.append(summary)
            caption = " — ".join(parts)
        payload["heads"].append(head)
        payload["captions"].append(caption)

        # ---- overview card (every server) ----
        if info.get("error"):
            payload["overview"].append(_stat(desc, "ERROR", str(info.get("error"))[:80]))
        else:
            gpu_count = 0 if not isinstance(table, pd.DataFrame) or table.empty else len(table)
            utils = [] if not gpu_count else [u for u in (_parse_percent(r.get("util")) for _, r in table.iterrows()) if u is not None]
            avg_util = sum(utils) / len(utils) if utils else 0.0
            used_sum = total_sum = 0.0
            if gpu_count:
                for _, r in table.iterrows():
                    used, total = _parse_mib_pair(r.get("memory[used/total]"))
                    if used is not None and total is not None:
                        used_sum += used
                        total_sum += total
            mem_str = f"{_format_gb(used_sum)}/{_format_gb(total_sum)}" if total_sum else "—"
            cpu_pct = sys_stats.get("cpu_percent")
            cpu_str = f"CPU {cpu_pct:.0f}%" if isinstance(cpu_pct, (int, float)) and sys_stats.get("cpu_cores") else None
            payload["overview"].append(
                html.Div(
                    [
                        html.Div(desc, className="k"),
                        html.Div(f"{avg_util:.0f}%" if gpu_count else "No GPUs", className="v",
                                 style={"color": _pct_color(avg_util, pal) if gpu_count else pal["muted"]}),
                        html.Div(
                            " | ".join(x for x in [f"{gpu_count} GPU" + ("s" if gpu_count != 1 else ""),
                                                   mem_str if gpu_count else None, cpu_str] if x),
                            className="s",
                        ),
                        _bar(avg_util, pal) if gpu_count else html.Div(),
                    ],
                    className="nv-stat",
                )
            )

        # ---- gpu-only slots (order matches skeleton's pattern components) ----
        if s["kind"] != "gpus":
            continue
        display = table.copy() if isinstance(table, pd.DataFrame) else pd.DataFrame()
        for col in ("util", "mem_util"):
            if col in display.columns:
                display[f"_{col}_num"] = display[col].map(lambda v: _parse_percent(v) if v is not None else None)
        records = display.to_dict("records")
        payload["tables"].append(records)
        payload["tooltips"].append(
            [{c: {"value": str(row.get(c, "")), "type": "text"} for c in s["cols"]} for row in records]
        )

        util_fig, mem_fig = _live_history_figs(history, desc, pal)
        payload["util_figs"].append(util_fig or _empty_fig(pal, "Collecting history…"))
        payload["mem_figs"].append(mem_fig or _empty_fig(pal, "Collecting history…"))

        if s["adv"]:
            adv_children = []
            adv_df = info.get("advanced_metrics")
            if isinstance(adv_df, pd.DataFrame) and not adv_df.empty:
                for title, cols in ADVANCED_METRIC_GROUPS:
                    cols_present = present_columns(adv_df.columns, list(cols))
                    if len(cols_present) < 2:
                        continue
                    sub = adv_df[cols_present].rename(columns=ADVANCED_METRIC_LABELS)
                    adv_children.append(html.Div(title, className="nv-caption", style={"margin": "8px 0 4px"}))
                    adv_children.append(_gpu_datatable(sub, None))
            payload["adv_children"].append(adv_children or html.Div("No data", className="nv-caption"))

        user_fig = _user_vram_fig(user_mem_by_client.get(idx, {}) or {}, pal, title="User VRAM (this node)")
        payload["user_figs"].append(user_fig or _empty_fig(pal, "No user processes"))

    return payload


def _live_tab(include_remote_default):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Servers"),
                            dcc.Checklist(
                                id="live-remote",
                                options=[{"label": " include remote", "value": "remote"}],
                                value=["remote"] if include_remote_default else [],
                                className="nv-check",
                            ),
                        ],
                        className="nv-control",
                    ),
                    html.Div(
                        [
                            html.Label("Refresh every"),
                            dcc.Dropdown(
                                id="live-interval-select",
                                options=[{"label": f"{s:g} s", "value": s} for s in (0.5, 1, 2, 5, 10, 30)],
                                value=5,
                                clearable=False,
                                style={"width": "110px"},
                            ),
                        ],
                        className="nv-control",
                    ),
                    html.Div(
                        [
                            html.Label("Auto refresh"),
                            dcc.Checklist(
                                id="live-auto",
                                options=[{"label": " enabled", "value": "on"}],
                                value=["on"],
                                className="nv-check",
                            ),
                        ],
                        className="nv-control",
                    ),
                    html.Button("Refresh now", id="live-refresh-btn", className="nv-btn"),
                    html.Div(id="live-status", className="nv-caption nv-flex1", style={"textAlign": "right"}),
                ],
                className="nv-card nv-card-tight nv-row nv-controls",
            ),
            dcc.Interval(id="live-interval", interval=5000),
            dcc.Store(id="live-sig"),
            html.Div(id="live-skeleton"),
        ]
    )


# ---------------------------------------------------------------------------
# Logs tab
# ---------------------------------------------------------------------------


def _session_records(db_path):
    try:
        df = _sessions_df(db_path)
    except Exception:
        return []
    records = []
    for _, row in df.sort_values(by="id", ascending=False).iterrows():
        records.append(
            {
                "id": int(row["id"]),
                "start": _format_datetime(row.get("start_time"), include_seconds=False),
                "duration": _format_duration(row.get("start_time"), row.get("end_time")),
                "status": str(row.get("status") or "unknown"),
                "snaps": int(row.get("snapshot_count") or 0),
                "remote": "yes" if row.get("include_remote") else "no",
            }
        )
    return records


def _logs_tab(db_path, initial_session_id):
    db_missing = db_path is None or not Path(db_path).exists()
    sidebar = html.Div(
        [
            html.Div(
                [
                    dcc.Input(id="session-search", type="text", placeholder="filter sessions…", debounce=True,
                              style={"width": "100%"}),
                ],
                style={"marginBottom": "8px"},
            ),
            dash_table.DataTable(
                id="session-table",
                columns=[
                    {"name": "#", "id": "id"},
                    {"name": "start", "id": "start"},
                    {"name": "dur", "id": "duration"},
                    {"name": "status", "id": "status"},
                    {"name": "snaps", "id": "snaps"},
                ],
                data=[],
                row_selectable="single",
                selected_rows=[],
                page_size=14,
                style_data_conditional=[
                    {"if": {"filter_query": '{status} = "running"', "column_id": "status"}, "color": "#34d399"},
                ],
                **{
                    **_TABLE_STYLE,
                    "style_cell": {**_TABLE_STYLE["style_cell"], "fontSize": "12px", "padding": "5px 8px"},
                },
            ),
        ],
        className="nv-card",
        style={"width": "360px", "flexShrink": 0, "alignSelf": "flex-start"},
    )

    main = html.Div(
        [
            html.Div(
                "No log database found — run `nvidb log` first." if db_missing else "Select a session on the left.",
                id="logs-placeholder",
                className="nv-card nv-caption",
            ),
            html.Div(id="logs-body"),
        ],
        className="nv-flex1",
        style={"minWidth": 0},
    )

    return html.Div(
        [
            dcc.Store(id="store-session", data=initial_session_id),
            dcc.Store(id="store-snapshot-ts"),
            html.Div([sidebar, main], className="nv-row", style={"alignItems": "flex-start", "flexWrap": "nowrap"}),
        ]
    )


def _logs_controls(session_id, nodes, ts_count):
    metric_options = [{"label": spec["label"], "value": key} for key, spec in _LOG_METRICS.items()]
    default_metrics = [k for k, spec in _LOG_METRICS.items() if spec.get("default")]
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Nodes"),
                    dcc.Dropdown(
                        id="logs-nodes",
                        options=[{"label": n, "value": n} for n in nodes],
                        value=nodes,
                        multi=True,
                        style={"minWidth": "260px"},
                    ),
                ],
                className="nv-control nv-flex1",
            ),
            html.Div(
                [
                    html.Label("Metrics"),
                    dcc.Dropdown(
                        id="logs-metrics",
                        options=metric_options,
                        value=default_metrics,
                        multi=True,
                        style={"minWidth": "320px"},
                    ),
                ],
                className="nv-control nv-flex1",
            ),
            html.Div(
                [
                    html.Label("Time range (snapshots)"),
                    dcc.RangeSlider(
                        id="logs-range",
                        min=0,
                        max=max(0, ts_count - 1),
                        value=[0, max(0, ts_count - 1)],
                        allowCross=False,
                        tooltip={"placement": "bottom"},
                        marks=None,
                    ),
                ],
                className="nv-control",
                style={"minWidth": "300px", "flex": 2},
            ),
        ],
        className="nv-card nv-card-tight nv-row nv-controls",
        key=f"controls-{session_id}",
    )


def _metric_figure(node_df, metric_key, spec, gpu_label_map, pal, snapshot_ts=None):
    out_col = f"_v_{metric_key}"
    long_df = node_df[["timestamp", "gpu_id", out_col]].rename(columns={out_col: "value"}).dropna(subset=["value"])
    if long_df.empty:
        return None
    long_df = _downsample_per_gpu(long_df, max_points_per_gpu=800)
    fig = go.Figure()
    use_gl = len(long_df) > 4000
    trace_cls = go.Scattergl if use_gl else go.Scatter
    for i, (gpu_id, g) in enumerate(long_df.groupby("gpu_id", sort=True)):
        try:
            label = gpu_label_map.get(int(gpu_id), f"GPU {gpu_id}")
        except Exception:
            label = f"GPU {gpu_id}"
        g = g.sort_values("timestamp")
        fig.add_trace(
            trace_cls(
                x=g["timestamp"],
                y=g["value"],
                name=label,
                mode="lines",
                line=dict(width=2, color=CHART_COLORS[i % len(CHART_COLORS)]),
            )
        )
    ymax = 100 if metric_key in ("util_gpu", "util_mem", "fan_speed") else None
    _fig_layout(fig, pal, title=spec.get("label", metric_key), height=int(spec.get("height", 220) * 1.15), ymax=ymax)
    fig.update_xaxes(tickformat="%m-%d %H:%M")
    if snapshot_ts is not None:
        vline_color = "rgba(15,23,42,0.45)" if pal is _PALETTES["light"] else "rgba(226,232,240,0.55)"
        fig.add_vline(x=snapshot_ts, line_width=1, line_dash="dot", line_color=vline_color)
    return fig


def _time_share_fig(time_share_df, pal):
    if time_share_df is None or time_share_df.empty:
        return None
    df = time_share_df.head(12)
    fig = go.Figure(
        go.Bar(
            x=df["share"],
            y=df["user"],
            orientation="h",
            marker=dict(color="#1E88E5", opacity=0.85),
            text=[f"{s:.0f}% ({n} snaps)" for s, n in zip(df["share"], df["snapshots"])],
            textposition="outside",
            hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 115], ticksuffix="%")
    return _fig_layout(fig, pal, title="User GPU-time share (selected range)", height=max(120, 44 + 26 * len(df)))


def _render_session_body(db_path, session_id, nodes_selected, metrics_selected, range_idx, snapshot_ts_iso, theme):
    pal = _pal(theme)
    df_all = _session_logs_df(db_path, session_id)
    if df_all.empty:
        return html.Div("No logs found for this session.", className="nv-card nv-caption")

    sessions = _sessions_df(db_path)
    row = sessions[sessions["id"] == int(session_id)]
    info = row.iloc[0] if not row.empty else {}

    end_raw = info.get("end_time")
    if end_raw in ("None", ""):
        end_raw = None
    kpis = html.Div(
        [
            _stat("Status", str(info.get("status") or "unknown")),
            _stat("Interval", f"{info.get('interval_seconds')}s" if info.get("interval_seconds") else "N/A"),
            _stat("Snapshots", f"{int(info.get('snapshot_count') or 0)}"),
            _stat("Records", f"{int(info.get('record_count') or 0)}"),
            _stat("Remote", "Yes" if info.get("include_remote") else "No"),
            _stat("Duration", _format_duration(info.get("start_time"), end_raw),
                  sub=f"{_format_datetime(info.get('start_time'))} → {_format_datetime(end_raw) if end_raw else 'running'}"),
        ],
        className="nv-stats",
    )

    nodes_all = sorted({str(n) for n in df_all["node"].dropna().unique()})
    nodes_selected = [n for n in (nodes_selected or nodes_all) if n in nodes_all] or nodes_all
    df = df_all[df_all["node"].isin(nodes_selected)]

    ts_all = list(df["timestamp"].dropna().drop_duplicates().sort_values())
    if not ts_all:
        return html.Div([kpis, html.Div("No timestamps in this session.", className="nv-card nv-caption")])

    lo, hi = 0, len(ts_all) - 1
    if isinstance(range_idx, (list, tuple)) and len(range_idx) == 2:
        lo = max(0, min(int(range_idx[0]), len(ts_all) - 1))
        hi = max(lo, min(int(range_idx[1]), len(ts_all) - 1))
    start_ts, end_ts = ts_all[lo], ts_all[hi]
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
    ts_window = [t for t in ts_all if start_ts <= t <= end_ts]

    # snapshot timestamp: clicked/stored, else window end
    snapshot_ts = ts_window[-1]
    if snapshot_ts_iso:
        try:
            want = pd.Timestamp(snapshot_ts_iso)
            snapshot_ts = min(ts_window, key=lambda t: abs(t - want))
        except Exception:
            pass

    range_caption = html.Div(
        f"{_format_datetime(start_ts)} → {_format_datetime(end_ts)}  ·  {len(ts_window)} snapshots  ·  "
        f"snapshot @ {_format_datetime(snapshot_ts)} (click any chart point to inspect)",
        className="nv-caption",
        style={"margin": "2px 4px 8px"},
    )

    # ---- parse metrics ----
    metrics_selected = [m for m in (metrics_selected or []) if m in _LOG_METRICS]
    parsed = df[["timestamp", "node", "gpu_id"]].copy()
    for key in metrics_selected:
        spec = _LOG_METRICS[key]
        src = spec.get("source", key)
        if src in df.columns:
            parsed[f"_v_{key}"] = df[src].map(spec["parser"])

    snapshot = df[df["timestamp"] == snapshot_ts]

    node_sections = []
    for node in sorted({str(n) for n in df["node"].dropna().unique()}):
        node_df = parsed[parsed["node"] == node]
        node_snapshot = snapshot[snapshot["node"] == node]
        node_table = _build_log_snapshot_table(node_snapshot)

        gpu_ids = sorted({int(g) for g in node_df["gpu_id"].dropna().unique()})
        gpu_names = {}
        name_source = node_snapshot if not node_snapshot.empty else df[df["node"] == node]
        if not name_source.empty and "name" in name_source.columns:
            for _, r in name_source[["gpu_id", "name"]].dropna().drop_duplicates().iterrows():
                try:
                    gid = int(r["gpu_id"])
                except Exception:
                    continue
                nm = _strip_gpu_name(r["name"])
                if nm and nm != "N/A" and gid not in gpu_names:
                    gpu_names[gid] = nm
        gpu_label_map = {g: (f"GPU {g} ({gpu_names[g]})" if g in gpu_names else f"GPU {g}") for g in gpu_ids}

        charts = []
        for key in metrics_selected:
            if f"_v_{key}" not in node_df.columns:
                continue
            fig = _metric_figure(node_df, key, _LOG_METRICS[key], gpu_label_map, pal, snapshot_ts=snapshot_ts)
            if fig is None:
                continue
            charts.append(
                dcc.Graph(
                    id={"type": "log-chart", "node": node, "metric": key},
                    figure=fig,
                    config=_GRAPH_CONFIG,
                )
            )

        section = [html.Div([html.Span(node, className="nv-server-title"),
                             html.Span(_server_summary(node_table), className="nv-badge")])]
        if charts:
            section.append(html.Div(charts, className="nv-grid", style={"marginTop": "8px"}))
        else:
            section.append(html.Div("No chart data (pick metrics above).", className="nv-caption", style={"margin": "8px 0"}))

        share_fig = _time_share_fig(_user_time_share_df(df[df["node"] == node]), pal)
        vram_fig = _user_vram_fig(_user_memory_from_df(node_snapshot), pal, title="User VRAM @ snapshot")
        aux = [g for g in (
            dcc.Graph(figure=share_fig, config=_GRAPH_CONFIG) if share_fig else None,
            dcc.Graph(figure=vram_fig, config=_GRAPH_CONFIG) if vram_fig else None,
        ) if g is not None]
        if aux:
            section.append(html.Div([html.Div(g, className="nv-flex1", style={"minWidth": "320px"}) for g in aux],
                                    className="nv-row"))

        if not node_table.empty:
            section.append(html.Div(f"Snapshot @ {_format_datetime(snapshot_ts)}", className="nv-caption",
                                    style={"margin": "8px 0 4px"}))
            section.append(_gpu_datatable(node_table, {"type": "log-snap-table", "index": node}))

        node_sections.append(html.Div(section, className="nv-card"))

    raw_view = html.Details(
        [
            html.Summary("Raw data (filter/sort/export)", className="nv-caption", style={"cursor": "pointer"}),
            dash_table.DataTable(
                data=df.assign(timestamp=df["timestamp"].astype(str)).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=20,
                sort_action="native",
                filter_action="native",
                export_format="csv",
                **{
                    **_TABLE_STYLE,
                    "style_cell": {**_TABLE_STYLE["style_cell"], "fontSize": "12px", "padding": "4px 8px"},
                },
            ),
        ],
        className="nv-card nv-card-tight",
    )

    return html.Div([kpis, range_caption, *node_sections, raw_view])


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(*, db_path=None, session_id=None, include_remote=False):
    if dash is None:  # pragma: no cover
        raise RuntimeError("dash is required for `nvidb web` (install with `pip install dash`).")

    db_path = _as_path(db_path or get_db_path())

    app = Dash(__name__, title="nvidb web", update_title=None, suppress_callback_exceptions=True)
    app.index_string = _INDEX_STRING

    # A browser tab loaded before a server restart keeps POSTing callbacks that
    # may no longer exist after a code change. Answer those with 204 (and one
    # warning) instead of a 500 traceback on every tick.
    import logging as _logging

    _stale_seen = set()

    @app.server.errorhandler(KeyError)
    def _stale_callback_from_old_tab(err):
        if "Callback function not found" in str(err):
            key = str(err)
            if key not in _stale_seen:
                _stale_seen.add(key)
                _logging.getLogger(__name__).warning(
                    "Ignoring callback from a browser tab loaded before the server restart — reload that tab. (%s)",
                    key,
                )
            return "", 204
        raise err

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div([html.Span("nvi"), html.B("db")], className="nv-brand"),
                    html.Div("aggregated NVIDIA GPU monitor", className="nv-sub nv-flex1"),
                    dcc.RadioItems(
                        id="theme-toggle",
                        options=[
                            {"label": "🌙 Dark", "value": "dark"},
                            {"label": "☀️ Light", "value": "light"},
                        ],
                        value="dark",
                        inline=True,
                        persistence=True,
                        persistence_type="local",
                        className="nv-check",
                    ),
                    html.Div(id="theme-sink", style={"display": "none"}),
                ],
                className="nv-header",
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-logs" if session_id is not None else "tab-live",
                className="nv-tabs",
                children=[
                    dcc.Tab(label="⚡ Live", value="tab-live", className="tab", selected_className="tab--selected",
                            children=_live_tab(include_remote)),
                    dcc.Tab(label="🗄 Logs", value="tab-logs", className="tab", selected_className="tab--selected",
                            children=_logs_tab(db_path, session_id)),
                ],
            ),
        ],
        className="nv-shell",
    )

    # Flip the CSS variable set instantly on the client; server-rendered
    # figures/tables follow on their next render (theme is an Input there).
    app.clientside_callback(
        "function(theme) { document.documentElement.setAttribute('data-theme', theme || 'dark'); return ''; }",
        Output("theme-sink", "children"),
        Input("theme-toggle", "value"),
    )

    # ---------------- Live callbacks ----------------

    @app.callback(
        Output("live-interval", "interval"),
        Output("live-interval", "disabled"),
        Input("live-interval-select", "value"),
        Input("live-auto", "value"),
        Input("live-remote", "value"),
        Input("live-refresh-btn", "n_clicks"),
    )
    def live_settings(interval_s, auto, remote, _clicks):
        include_remote_now = "remote" in (remote or [])
        enabled = "on" in (auto or [])
        try:
            interval = float(interval_s)
        except (TypeError, ValueError):
            interval = 5.0
        cache = _get_live_cache(include_remote_now)
        cache.configure(interval_seconds=interval, enabled=enabled)
        if ctx.triggered_id in ("live-refresh-btn", "live-remote"):
            cache.refresh_now()
        return max(250, int(interval * 1000)), not enabled

    # Skeleton pass: rebuilds the live DOM only when its structure changes
    # (server set / columns / first fetch). Ticks where nothing structural
    # changed return no_update, so the browser keeps scroll position, plot
    # zoom, table sort and open <details> panels.
    @app.callback(
        Output("live-skeleton", "children"),
        Output("live-sig", "data"),
        Input("live-interval", "n_intervals"),
        Input("live-remote", "value"),
        State("live-sig", "data"),
        State("tabs", "value"),
        prevent_initial_call=False,
    )
    def live_skeleton(_n, remote, stored_sig, tab):
        if tab != "tab-live":
            raise PreventUpdate
        cache = _get_live_cache("remote" in (remote or []))
        sig = _live_signature(cache)
        if sig == stored_sig:
            raise PreventUpdate
        return _build_live_skeleton(sig), sig

    # Update pass: writes data/figures into the existing skeleton every tick.
    # `live-sig` is an Input so dash sequences this after the skeleton pass
    # whenever the structure just changed.
    @app.callback(
        Output("live-alerts", "children"),
        Output("live-overview", "children"),
        Output("live-global-vram", "figure"),
        Output("live-global-wrap", "style"),
        Output({"type": "live-head", "index": ALL}, "children"),
        Output({"type": "live-caption", "index": ALL}, "children"),
        Output({"type": "live-table", "index": ALL}, "data"),
        Output({"type": "live-table", "index": ALL}, "tooltip_data"),
        Output({"type": "live-util", "index": ALL}, "figure"),
        Output({"type": "live-mem", "index": ALL}, "figure"),
        Output({"type": "live-adv", "index": ALL}, "children"),
        Output({"type": "live-uservram", "index": ALL}, "figure"),
        Output("live-status", "children"),
        Input("live-sig", "data"),
        Input("live-interval", "n_intervals"),
        Input("live-refresh-btn", "n_clicks"),
        Input("theme-toggle", "value"),
        State("live-remote", "value"),
        State("tabs", "value"),
        prevent_initial_call=False,
    )
    def live_update(sig, _n, _clicks, theme, remote, tab):
        if tab != "tab-live" or not sig or not sig.get("fetched"):
            raise PreventUpdate
        cache = _get_live_cache("remote" in (remote or []))
        p = _live_update_payload(cache, sig, theme)
        return (
            p["alerts"],
            p["overview"],
            p["global_fig"],
            p["global_style"],
            p["heads"],
            p["captions"],
            p["tables"],
            p["tooltips"],
            p["util_figs"],
            p["mem_figs"],
            p["adv_children"],
            p["user_figs"],
            p["status"],
        )

    # ---------------- Logs callbacks ----------------

    @app.callback(
        Output("session-table", "data"),
        Output("session-table", "selected_rows"),
        Input("tabs", "value"),
        Input("session-search", "value"),
        State("session-table", "data"),
        State("session-table", "selected_rows"),
        State("store-session", "data"),
    )
    def sessions_refresh(tab, query, current_data, current_selected, stored_session):
        if tab != "tab-logs":
            return no_update, no_update
        records = _session_records(db_path)
        if query:
            q = str(query).strip().lower()
            records = [r for r in records if q in json.dumps(r, default=str).lower()]
        selected = []
        target = stored_session
        if target is None and records:
            target = records[0]["id"]  # newest first
        for i, r in enumerate(records):
            if target is not None and r["id"] == int(target):
                selected = [i]
                break
        if not selected and records:
            selected = [0]
        return records, selected

    @app.callback(
        Output("store-session", "data"),
        Output("logs-placeholder", "style"),
        Output("logs-body", "children"),
        Output("store-snapshot-ts", "data", allow_duplicate=True),
        Input("session-table", "selected_rows"),
        State("session-table", "data"),
        prevent_initial_call=True,
    )
    def session_selected(selected_rows, data):
        if not selected_rows or not data:
            return no_update, no_update, no_update, no_update
        try:
            session_id = int(data[selected_rows[0]]["id"])
        except Exception:
            return no_update, no_update, no_update, no_update

        df = _session_logs_df(db_path, session_id)
        nodes = sorted({str(n) for n in df["node"].dropna().unique()}) if not df.empty else []
        ts_count = int(df["timestamp"].dropna().nunique()) if not df.empty else 0
        body = html.Div(
            [
                _logs_controls(session_id, nodes, ts_count),
                dcc.Loading(html.Div(id="session-body"), type="dot", color=ACCENT),
            ]
        )
        return session_id, {"display": "none"}, body, None

    @app.callback(
        Output("session-body", "children"),
        Input("store-session", "data"),
        Input("logs-nodes", "value"),
        Input("logs-metrics", "value"),
        Input("logs-range", "value"),
        Input("store-snapshot-ts", "data"),
        Input("theme-toggle", "value"),
    )
    def session_render(session_id, nodes, metrics, range_idx, snapshot_ts, theme):
        if session_id is None:
            return no_update
        return _render_session_body(db_path, int(session_id), nodes, metrics, range_idx, snapshot_ts, theme)

    @app.callback(
        Output("store-snapshot-ts", "data", allow_duplicate=True),
        Input({"type": "log-chart", "node": ALL, "metric": ALL}, "clickData"),
        prevent_initial_call=True,
    )
    def chart_clicked(click_data_list):
        trigger = ctx.triggered
        if not trigger:
            return no_update
        click = trigger[0].get("value")
        if not click or not click.get("points"):
            return no_update
        return click["points"][0].get("x")

    return app


def run_dash_app(*, session_id=None, db_path=None, port=8501, include_remote=False):
    import importlib.util
    import logging

    if importlib.util.find_spec("dash") is None:
        print("dash is required for `nvidb web`.\nInstall: pip install dash")
        raise SystemExit(1)

    # Keep the terminal quiet: no per-request werkzeug lines, no dash/paramiko
    # INFO chatter (connection errors and warnings still come through).
    logging.getLogger().setLevel(logging.WARNING)
    for name in ("werkzeug", "dash.dash", "paramiko", "paramiko.transport"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # dash re-enables its INFO logging inside app.run(); a filter survives that.
    logging.getLogger("dash.dash").addFilter(lambda record: record.levelno > logging.INFO)
    try:  # hide Flask's "Serving Flask app ..." banner
        import flask.cli

        flask.cli.show_server_banner = lambda *args, **kwargs: None
    except Exception:
        pass

    app = create_app(db_path=db_path, session_id=session_id, include_remote=include_remote)
    print(f"Starting nvidb web (Dash) on port {port}...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(host="127.0.0.1", port=int(port), debug=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    run_dash_app(db_path=args.db_path, port=args.port)
