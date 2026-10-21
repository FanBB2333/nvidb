"""Regression tests for the web dashboard's live cache and log cache.

These cover the parts of nvidb.dashboard that run without Dash/plotly
installed (Dash is an optional dependency).
"""

import threading
import time
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from nvidb import dashboard


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakePool:
    def __init__(self, server_list):
        self.pool = []
        self.calls = 0
        self._lock = threading.Lock()

    def get_client_gpus_info(self, return_raw=False):
        with self._lock:
            self.calls += 1
        return "", {"_nvidb": {}}


def test_live_cache_resume_and_manual_refresh_fetch_while_paused(monkeypatch):
    monkeypatch.setattr(dashboard, "NVClientPool", _FakePool)
    cache = dashboard.LiveCache(False)
    pool = cache.pool
    try:
        assert _wait_for(lambda: pool.calls >= 1)
        time.sleep(0.2)
        assert pool.calls == 1  # one startup fetch, not two

        cache.configure(interval_seconds=5, enabled=False)
        time.sleep(0.2)
        assert pool.calls == 1  # pausing triggers no fetch

        cache.refresh_now()
        assert _wait_for(lambda: pool.calls == 2)
        time.sleep(0.2)
        assert pool.calls == 2  # a manual refresh while paused fetches once

        cache.refresh_now()
        assert _wait_for(lambda: pool.calls == 3)  # ...and the next one is not swallowed

        cache.configure(interval_seconds=0.25, enabled=True)
        assert _wait_for(lambda: pool.calls >= 5)  # resuming fetches without a click
    finally:
        cache.configure(interval_seconds=5, enabled=False)


def test_a_paused_manual_refresh_reports_when_its_own_fetch_lands(monkeypatch):
    monkeypatch.setattr(dashboard, "NVClientPool", _FakePool)
    cache = dashboard.LiveCache(False)
    pool = cache.pool
    try:
        assert _wait_for(lambda: pool.calls >= 1)
        cache.configure(interval_seconds=5, enabled=False)
        time.sleep(0.1)
        before = cache.snapshot()["last_updated"]

        landed = cache.refresh_now()
        assert landed.wait(timeout=3)
        assert pool.calls == 2
        assert cache.snapshot()["last_updated"] > before
    finally:
        cache.configure(interval_seconds=5, enabled=False)


def test_live_signature_hides_renderer_only_columns():
    table = pd.DataFrame([{"GPU": 0, "util": "5%", "_nvidb_rx_percent": 1.0, "_nvidb_link_max": "Gen4"}])
    cache = SimpleNamespace(
        snapshot=lambda: {"raw": {0: (table, {}), "_nvidb": {}}},
        pool=SimpleNamespace(pool=[SimpleNamespace(description="node")]),
        include_remote=False,
    )
    sig = dashboard._live_signature(cache)
    assert sig["servers"][0]["cols"] == ["GPU", "util"]


def test_logs_cache_drops_expired_entries(monkeypatch):
    monkeypatch.setattr(dashboard, "_logs_cache", {("logs", "db", 1): (time.monotonic() - 60, "old")})
    assert dashboard._cached(("logs", "db", 2), lambda: "new") == "new"
    assert list(dashboard._logs_cache) == [("logs", "db", 2)]


class _FakeFigure:
    def __init__(self, *args, **kwargs):
        self.yaxes = {}

    def add_trace(self, trace):
        pass

    def update_layout(self, **kwargs):
        pass

    def update_xaxes(self, **kwargs):
        pass

    def update_yaxes(self, **kwargs):
        self.yaxes.update(kwargs)


def test_rolling_vram_range_fits_the_largest_gpu(monkeypatch):
    monkeypatch.setattr(dashboard, "go", SimpleNamespace(Figure=_FakeFigure, Scatter=dict), raising=False)
    rows = [
        {"server": "406", "gpu": "GPU 0", "util": 10.0, "mem_used_gib": 4.0, "mem_total_gib": 32.0},
        {"server": "406", "gpu": "GPU 1", "util": 90.0, "mem_used_gib": 60.0, "mem_total_gib": 72.0},
    ]
    history = [{"ts": datetime(2026, 1, 1), "rows": rows}]
    _util_fig, mem_fig = dashboard._live_history_figs(history, "406", dashboard._pal("dark"))
    assert mem_fig.yaxes["range"] == [0, 72.0 * 1.05]
