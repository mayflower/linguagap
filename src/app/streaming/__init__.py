"""Real-time streaming transcription and translation pipeline.

This package implements the WebSocket streaming handler for real-time
bilingual conversation transcription. The architecture is layered:

- :mod:`app.streaming.session`   — ChannelBuffer + StreamingSession state
- :mod:`app.streaming.serialize` — Segment → wire-format dict helpers
- :mod:`app.streaming.broadcast` — viewer fan-out
- :mod:`app.streaming.asr`       — ASR / MT pipeline (run_asr_*, run_translation)
- :mod:`app.streaming.trace`     — high-resolution lifecycle logging
- :mod:`app.streaming.handler`   — host WebSocketHandler + handle_websocket
- :mod:`app.streaming.viewer`    — read-only viewer endpoint

Public API of this package re-exports everything callers (main, tests)
relied on when this lived in a single ``streaming.py`` module.
"""

from app.session_registry import registry
from app.streaming._metrics import _avg_ms, _metrics, get_metrics
from app.streaming.asr import (
    _CROSSTALK_DOMINANCE_RATIO,
    _CROSSTALK_MIN_ACTIVE_RMS,
    _build_prompt,
    _executor,
    _is_effective_silence,
    _transcribe_channel,
    run_asr_dual_channel,
    run_asr_german_channel,
    run_translation,
)
from app.streaming.broadcast import _maybe_broadcast, broadcast_to_viewers
from app.streaming.handler import WebSocketHandler, handle_websocket
from app.streaming.serialize import (
    _resolve_segment_role,
    _resolve_translation_pair,
    _role_from_lang,
    _serialize_segments,
)
from app.streaming.session import (
    MAX_BUFFER_SEC,
    TICK_SEC,
    WINDOW_SEC,
    ChannelBuffer,
    StreamingSession,
)
from app.streaming.trace import TRACE_ENABLED, _trace
from app.streaming.viewer import _delayed_viewer_speaking_off, handle_viewer_websocket

__all__ = [
    # Session + buffer
    "ChannelBuffer",
    "StreamingSession",
    "MAX_BUFFER_SEC",
    "TICK_SEC",
    "WINDOW_SEC",
    # ASR pipeline
    "run_asr_german_channel",
    "run_asr_dual_channel",
    "run_translation",
    "_build_prompt",
    "_transcribe_channel",
    "_is_effective_silence",
    "_CROSSTALK_DOMINANCE_RATIO",
    "_CROSSTALK_MIN_ACTIVE_RMS",
    "_executor",
    # Serialization
    "_role_from_lang",
    "_resolve_segment_role",
    "_resolve_translation_pair",
    "_serialize_segments",
    # Broadcast
    "broadcast_to_viewers",
    "_maybe_broadcast",
    # WebSocket handlers
    "WebSocketHandler",
    "handle_websocket",
    "handle_viewer_websocket",
    "_delayed_viewer_speaking_off",
    # Tracing + metrics
    "_trace",
    "TRACE_ENABLED",
    "_metrics",
    "_avg_ms",
    "get_metrics",
    # Re-export so tests can monkeypatch streaming.registry directly
    "registry",
]
