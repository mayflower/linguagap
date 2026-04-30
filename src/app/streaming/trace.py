"""High-resolution lifecycle tracing for the streaming pipeline.

Set ``LINGUAGAP_TRACE=1`` to emit a TRACE line at each stage so
lost-transcription bugs can be reconstructed from logs. Lines look like::

    TRACE 12:34:56.789 ev=asr_emit tok=abc12345 seg=42 final=False src_lang=de text="Guten Tag"
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("app.streaming")

TRACE_ENABLED = os.getenv("LINGUAGAP_TRACE", "1") == "1"


def _trace(event: str, **fields) -> None:
    """Emit a single high-resolution trace line if TRACE_ENABLED.

    Looks up ``TRACE_ENABLED`` on the parent package at call time so tests
    that ``monkeypatch.setattr(streaming, "TRACE_ENABLED", ...)`` to
    silence/enable tracing in a single test still take effect here.
    """
    from app import streaming as _streaming

    if not _streaming.TRACE_ENABLED:
        return
    now = time.time()
    ts = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
    parts = [f"ev={event}"]
    for k, v in fields.items():
        if isinstance(v, str):
            flat = v.replace("\n", " ").replace("\r", " ")
            if len(flat) > 200:
                flat = flat[:197] + "..."
            parts.append(f'{k}="{flat}"')
        else:
            parts.append(f"{k}={v}")
    logger.info("TRACE %s %s", ts, " ".join(parts))
