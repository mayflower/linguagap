"""Tests for the pure helpers in app.streaming.

These cover the small functions that operate on Segments / dicts / numpy
arrays without needing a real WebSocket or Whisper. They make up a large
chunk of streaming.py's coverage with no infrastructure.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app import streaming
from app.streaming import (
    ChannelBuffer,
    _avg_ms,
    _is_effective_silence,
    _maybe_broadcast,
    _resolve_segment_role,
    _resolve_translation_pair,
    _role_from_lang,
    _serialize_segments,
    broadcast_to_viewers,
    get_metrics,
)
from app.streaming_policy import Segment

# ---------------------------------------------------------------------------
# _role_from_lang
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lang,expected",
    [
        ("de", "german"),
        ("en", "foreign"),
        ("fr", "foreign"),
        ("ar", "foreign"),
        (None, None),
        ("unknown", None),
        ("", None),
    ],
)
def test_role_from_lang(lang: str | None, expected: str | None) -> None:
    assert _role_from_lang(lang) == expected


# ---------------------------------------------------------------------------
# _resolve_segment_role
# ---------------------------------------------------------------------------


def _seg(**kwargs: Any) -> Segment:
    defaults = dict(
        id=0,
        abs_start=0.0,
        abs_end=1.0,
        src="x",
        src_lang="en",
        final=True,
        speaker_id=None,
        speaker_role=None,
    )
    defaults.update(kwargs)
    return Segment(**defaults)  # type: ignore[arg-type]


def test_resolve_role_uses_explicit_role_when_set() -> None:
    s = _seg(speaker_role="german", src_lang="en")  # explicit overrides language
    assert _resolve_segment_role(s, dual_channel=False) == "german"


def test_resolve_role_dual_channel_uses_speaker_id() -> None:
    assert _resolve_segment_role(_seg(speaker_id="SPEAKER_00"), dual_channel=True) == "german"
    assert _resolve_segment_role(_seg(speaker_id="SPEAKER_01"), dual_channel=True) == "foreign"


def test_resolve_role_dual_channel_unknown_speaker_returns_none() -> None:
    """Dual channel must not guess from language if the speaker_id is unfamiliar."""
    assert _resolve_segment_role(_seg(speaker_id="SPEAKER_42"), dual_channel=True) is None
    assert _resolve_segment_role(_seg(speaker_id=None), dual_channel=True) is None


def test_resolve_role_single_channel_falls_back_to_language() -> None:
    assert _resolve_segment_role(_seg(src_lang="de"), dual_channel=False) == "german"
    assert _resolve_segment_role(_seg(src_lang="fr"), dual_channel=False) == "foreign"


# ---------------------------------------------------------------------------
# _serialize_segments
# ---------------------------------------------------------------------------


def _fake_session(
    *,
    foreign_lang: str | None = "en",
    dual_channel: bool = False,
    translations: dict[int, dict[str, str]] | None = None,
) -> Any:
    return SimpleNamespace(
        foreign_lang=foreign_lang,
        is_dual_channel=lambda: dual_channel,
        translations=translations or {},
    )


def test_serialize_attaches_role_and_translations() -> None:
    session = _fake_session(translations={0: {"de": "Hallo"}})
    seg = _seg(id=0, src="hello", src_lang="en")
    out = _serialize_segments(session, [seg])
    assert out[0]["speaker_role"] == "foreign"
    assert out[0]["translations"] == {"de": "Hallo"}


def test_serialize_overrides_src_lang_for_german_role() -> None:
    """German role wins even if Whisper reported a different src_lang."""
    session = _fake_session(foreign_lang="en")
    seg = _seg(speaker_role="german", src_lang="en")  # mistaken Whisper detection
    out = _serialize_segments(session, [seg])
    assert out[0]["src_lang"] == "de"


def test_serialize_overrides_src_lang_for_foreign_role_with_known_foreign() -> None:
    session = _fake_session(foreign_lang="fr")
    seg = _seg(speaker_role="foreign", src_lang="es")  # mistaken
    out = _serialize_segments(session, [seg])
    assert out[0]["src_lang"] == "fr"


def test_serialize_keeps_src_lang_when_foreign_lang_unknown() -> None:
    """If the session's foreign_lang isn't in LANG_INFO we leave src_lang alone."""
    session = _fake_session(foreign_lang="zz")
    seg = _seg(speaker_role="foreign", src_lang="en")
    out = _serialize_segments(session, [seg])
    assert out[0]["src_lang"] == "en"


# ---------------------------------------------------------------------------
# _resolve_translation_pair
# ---------------------------------------------------------------------------


def test_translation_pair_german_role_targets_foreign() -> None:
    seg = _seg(src_lang="de")
    assert _resolve_translation_pair(seg, "german", "en") == ("de", "en")


def test_translation_pair_german_role_skips_when_no_foreign() -> None:
    seg = _seg(src_lang="de")
    assert _resolve_translation_pair(seg, "german", None) is None


def test_translation_pair_foreign_role_targets_de_using_session_lang() -> None:
    seg = _seg(src_lang="es")  # misdetected
    assert _resolve_translation_pair(seg, "foreign", "fr") == ("fr", "de")


def test_translation_pair_no_role_falls_back_to_src_lang() -> None:
    # German source with foreign session lang -> de->foreign
    assert _resolve_translation_pair(_seg(src_lang="de"), None, "en") == ("de", "en")
    # Non-German source -> source->de
    assert _resolve_translation_pair(_seg(src_lang="fr"), None, "en") == ("fr", "de")


def test_translation_pair_unknown_foreign_lang_treated_as_none() -> None:
    seg = _seg(src_lang="de")
    # foreign_lang="zz" is not in LANG_INFO -> resolved as None -> skip.
    assert _resolve_translation_pair(seg, "german", "zz") is None


def test_translation_pair_german_only_session_skips_translation() -> None:
    """When the host picks German as the foreign language, src and tgt are
    identical for both speaker roles. _resolve_translation_pair must return
    None so the MT loop never enqueues a no-op de->de translation."""
    seg_de = _seg(src_lang="de")
    seg_en_misdetected = _seg(src_lang="en")
    assert _resolve_translation_pair(seg_de, "german", "de") is None
    assert _resolve_translation_pair(seg_en_misdetected, "foreign", "de") is None
    # Auto-detected role with German source and a German session also skips.
    assert _resolve_translation_pair(seg_de, None, "de") is None


# ---------------------------------------------------------------------------
# _is_effective_silence
# ---------------------------------------------------------------------------


def test_silence_detector_treats_empty_array_as_silence() -> None:
    assert _is_effective_silence(np.zeros(0, dtype=np.float32)) is True


def test_silence_detector_zeros_are_silent() -> None:
    assert _is_effective_silence(np.zeros(16000, dtype=np.float32)) is True


def test_silence_detector_loud_signal_is_not_silence() -> None:
    audio = np.full(16000, 0.5, dtype=np.float32)
    assert _is_effective_silence(audio) is False


def test_silence_detector_low_rms_but_above_peak_is_not_silence() -> None:
    audio = np.zeros(16000, dtype=np.float32)
    audio[100] = 0.9  # one loud sample dominates the peak even with low RMS
    assert _is_effective_silence(audio) is False


# ---------------------------------------------------------------------------
# _avg_ms / get_metrics
# ---------------------------------------------------------------------------


def test_avg_ms_handles_empty_deque() -> None:
    assert _avg_ms(deque()) == 0


def test_avg_ms_returns_milliseconds() -> None:
    # Times are stored in seconds; helper returns ms.
    assert _avg_ms(deque([0.1, 0.2, 0.3])) == pytest.approx(200.0, abs=1e-6)


def test_get_metrics_structure_and_zero_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh: dict[str, deque[float]] = {
        "asr_times": deque(),
        "mt_times": deque(),
        "diar_times": deque(),
        "tick_times": deque(),
    }
    monkeypatch.setattr(streaming, "_metrics", fresh)
    m = get_metrics()
    assert set(m) == {
        "avg_asr_time_ms",
        "avg_mt_time_ms",
        "avg_diar_time_ms",
        "avg_tick_time_ms",
        "sample_count",
    }
    assert m["sample_count"] == 0


# ---------------------------------------------------------------------------
# ChannelBuffer
# ---------------------------------------------------------------------------


def test_channel_buffer_reset_clears_buffer_but_preserves_offset() -> None:
    """reset() drops audio + counters and flips ``started`` off so the next
    add_audio() call can re-anchor start_offset_sec to wall-clock. The
    pre-existing offset is intentionally NOT cleared (see docstring on
    ChannelBuffer.reset)."""
    cb = ChannelBuffer()
    cb.audio.append(b"ABCD")
    cb.total_samples = 2048
    cb.trimmed_samples = 1024
    cb.start_offset_sec = 5.0
    cb.started = True

    cb.reset()

    assert list(cb.audio) == []
    assert cb.total_samples == 0
    assert cb.trimmed_samples == 0
    assert cb.started is False
    assert cb.start_offset_sec == 5.0  # preserved by design


# ---------------------------------------------------------------------------
# broadcast_to_viewers / _maybe_broadcast
# ---------------------------------------------------------------------------


class _FakeWS:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[str] = []
        self.fail = fail

    async def send_text(self, text: str) -> None:
        if self.fail:
            raise RuntimeError("client gone")
        self.sent.append(text)


async def test_broadcast_to_viewers_sends_to_everyone() -> None:
    a, b = _FakeWS(), _FakeWS()
    entry = SimpleNamespace(viewers={a, b})
    await broadcast_to_viewers(entry, {"type": "ping"})  # type: ignore[arg-type]
    assert a.sent and b.sent
    assert json.loads(a.sent[0]) == {"type": "ping"}


async def test_broadcast_to_viewers_drops_dead_viewers() -> None:
    good = _FakeWS()
    dead = _FakeWS(fail=True)
    viewers: set[Any] = {good, dead}
    entry = SimpleNamespace(viewers=viewers)
    await broadcast_to_viewers(entry, {"type": "ping"})  # type: ignore[arg-type]
    assert dead not in entry.viewers
    assert good in entry.viewers


async def test_broadcast_to_viewers_noop_on_empty_set() -> None:
    entry = SimpleNamespace(viewers=set())
    # Doesn't raise, doesn't try to serialize.
    await broadcast_to_viewers(entry, {"type": "ping"})  # type: ignore[arg-type]


async def test_maybe_broadcast_does_nothing_when_token_is_none() -> None:
    # If this called registry.get(None) we'd see TypeError downstream.
    await _maybe_broadcast(None, {"type": "x"})


async def test_maybe_broadcast_skips_unknown_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(_token: str) -> None:
        return None

    monkeypatch.setattr(streaming.registry, "get", fake_get)
    # Should not raise even though no entry exists.
    await _maybe_broadcast("missing", {"type": "x"})


async def test_maybe_broadcast_dispatches_to_broadcast_when_entry_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _FakeWS()
    entry = SimpleNamespace(viewers={a})

    async def fake_get(_token: str) -> Any:
        return entry

    monkeypatch.setattr(streaming.registry, "get", fake_get)
    await _maybe_broadcast("tok", {"type": "session_ended"})
    assert a.sent and json.loads(a.sent[0]) == {"type": "session_ended"}


# ---------------------------------------------------------------------------
# event-loop bookkeeping — a regression guard, not a behavior test
# ---------------------------------------------------------------------------


async def test_streaming_module_uses_python_event_loop() -> None:
    """Ensure pytest-asyncio + module-level dataclasses don't trip a sync loop."""
    await asyncio.sleep(0)
