"""Shared metrics dict for the streaming pipeline.

Lives in its own module so every consumer (ASR loop, MT loop, the
``get_metrics`` aggregator, and tests) imports the *same* deque
instances regardless of import order.
"""

from __future__ import annotations

from collections import deque

_metrics: dict[str, deque[float]] = {
    "asr_times": deque(maxlen=100),
    "mt_times": deque(maxlen=100),
    "diar_times": deque(maxlen=100),
    "tick_times": deque(maxlen=100),
}


def _avg_ms(samples: deque) -> float:
    return sum(samples) / len(samples) * 1000 if samples else 0


def get_metrics() -> dict:
    """Aggregate per-stage timings.

    Resolves ``_metrics`` through the parent package at call time so that
    tests which ``monkeypatch.setattr(streaming, "_metrics", fresh_dict)``
    actually see their mock here.
    """
    from app import streaming as _streaming

    m = _streaming._metrics
    return {
        "avg_asr_time_ms": _avg_ms(m["asr_times"]),
        "avg_mt_time_ms": _avg_ms(m["mt_times"]),
        "avg_diar_time_ms": _avg_ms(m["diar_times"]),
        "avg_tick_time_ms": _avg_ms(m["tick_times"]),
        "sample_count": len(m["tick_times"]),
    }
