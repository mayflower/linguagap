"""Tests for StreamingSession state and the dual-channel helpers."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from app import streaming
from app.streaming import (
    StreamingSession,
    _build_prompt,
    _delayed_viewer_speaking_off,
    _trace,
    _transcribe_channel,
    run_translation,
)
from app.streaming_policy import Segment


# ---------------------------------------------------------------------------
# _trace
# ---------------------------------------------------------------------------


def test_trace_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(streaming, "TRACE_ENABLED", False)
    with caplog.at_level("INFO", logger="app.streaming"):
        _trace("foo", k="v")
    assert "TRACE" not in caplog.text


def test_trace_emits_quoted_strings_and_bare_numbers(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr(streaming, "TRACE_ENABLED", True)
    with caplog.at_level("INFO", logger="app.streaming"):
        _trace("event_x", text="hello\nworld", count=42)
    assert "ev=event_x" in caplog.text
    assert 'text="hello world"' in caplog.text  # newline collapsed to space
    assert "count=42" in caplog.text


def test_trace_truncates_overlong_strings(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(streaming, "TRACE_ENABLED", True)
    long_text = "x" * 500
    with caplog.at_level("INFO", logger="app.streaming"):
        _trace("ev", text=long_text)
    # Truncated payload has 197 chars + "..."
    assert "x" * 197 + "..." in caplog.text


# ---------------------------------------------------------------------------
# StreamingSession dual-channel buffers
# ---------------------------------------------------------------------------


def _silence_chunk(samples: int = 1600) -> bytes:
    return (np.zeros(samples, dtype=np.int16)).tobytes()


def test_add_german_audio_populates_german_channel_only() -> None:
    s = StreamingSession()
    s.add_german_audio(_silence_chunk(800))
    assert s.german_channel.total_samples == 800
    assert s.foreign_channel.total_samples == 0
    assert s.is_dual_channel() is False  # foreign mic still untouched


def test_add_foreign_audio_locks_dual_channel() -> None:
    s = StreamingSession()
    s.add_foreign_audio(_silence_chunk(800))
    assert s.is_dual_channel() is True
    assert s.foreign_channel.started is True
    assert s.viewer_last_audio_time > 0


def test_get_window_audio_for_german_channel_returns_float32() -> None:
    s = StreamingSession()
    samples = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
    s.add_german_audio(samples)
    audio, start = s.get_german_window_audio()
    assert audio.dtype == np.float32
    assert audio.shape == (1600,)
    assert start >= 0.0


def test_get_window_audio_empty_channel_returns_empty_array() -> None:
    s = StreamingSession()
    audio, start = s.get_foreign_window_audio()
    assert audio.shape == (0,)
    assert start == 0.0


def test_resolve_foreign_lang_uses_user_selection() -> None:
    s = StreamingSession(src_lang="fr")
    assert s.foreign_lang is None
    s.resolve_foreign_lang()
    assert s.foreign_lang == "fr"


def test_resolve_foreign_lang_skips_when_already_set() -> None:
    s = StreamingSession(src_lang="fr")
    s.foreign_lang = "es"  # already auto-detected, do not override
    s.resolve_foreign_lang()
    assert s.foreign_lang == "es"


def test_resolve_foreign_lang_skips_when_src_is_de_or_auto() -> None:
    s = StreamingSession(src_lang="auto")
    s.resolve_foreign_lang()
    assert s.foreign_lang is None

    s = StreamingSession(src_lang="de")
    s.resolve_foreign_lang()
    assert s.foreign_lang is None


def test_get_current_time_dual_channel_uses_max_of_both() -> None:
    s = StreamingSession()
    s.add_german_audio(_silence_chunk(8000))  # 0.5s @ 16kHz
    s.add_foreign_audio(_silence_chunk(16000))  # 1.0s
    # Dual channel locked once foreign added; current time is max of channels.
    assert s.get_current_time() >= 1.0


def test_get_buffered_seconds_dual_channel_returns_max() -> None:
    s = StreamingSession()
    s.add_german_audio(_silence_chunk(8000))  # 0.5s
    s.add_foreign_audio(_silence_chunk(16000))  # 1.0s
    assert s.get_buffered_seconds() == pytest.approx(1.0, abs=0.01)


def test_get_buffered_seconds_single_channel_returns_buffer_length() -> None:
    s = StreamingSession()
    s.add_audio(_silence_chunk(8000))
    assert s.get_buffered_seconds() == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------


def _backend_with_prompt(prompt: str | None) -> Any:
    return SimpleNamespace(get_bilingual_prompt=lambda _lang: prompt)


def test_build_prompt_returns_none_when_no_lang_no_text() -> None:
    backend = _backend_with_prompt("never used")
    assert _build_prompt(backend, language=None, finalized_text="") is None


def test_build_prompt_uses_backend_prompt_when_only_language_given() -> None:
    backend = _backend_with_prompt("Bilingual context")
    assert _build_prompt(backend, language="en", finalized_text="") == "Bilingual context"


def test_build_prompt_appends_recent_finalized_text() -> None:
    backend = _backend_with_prompt("Hint")
    out = _build_prompt(backend, language="en", finalized_text="prior sentence")
    assert out == "Hint prior sentence"


def test_build_prompt_text_only_when_backend_has_no_hint() -> None:
    backend = _backend_with_prompt(None)
    out = _build_prompt(backend, language="en", finalized_text="prior sentence")
    assert out == "prior sentence"


def test_build_prompt_truncates_to_last_200_chars() -> None:
    backend = _backend_with_prompt(None)
    long_text = "abcdefghij" * 50  # 500 chars
    out = _build_prompt(backend, language=None, finalized_text=long_text)
    assert out is not None and len(out) == 200
    assert out.endswith("abcdefghij")


# ---------------------------------------------------------------------------
# _transcribe_channel
# ---------------------------------------------------------------------------


def _asr_backend(seg_text: str = "hello", seg_language: str = "en") -> Any:
    fake_seg = SimpleNamespace(start=0.0, end=1.0, text=seg_text, language=seg_language)
    fake_result = SimpleNamespace(segments=[fake_seg])
    return SimpleNamespace(
        get_bilingual_prompt=lambda _lang: None,
        transcribe=lambda audio, *, language=None, initial_prompt=None: fake_result,
    )


def test_transcribe_channel_skips_short_audio() -> None:
    short_audio = np.ones(1599, dtype=np.float32)
    out = _transcribe_channel(_asr_backend(), short_audio, "de", "S0", "german")
    assert out == []


def test_transcribe_channel_skips_silence() -> None:
    silence = np.zeros(8000, dtype=np.float32)
    out = _transcribe_channel(_asr_backend(), silence, "de", "S0", "german")
    assert out == []


def test_transcribe_channel_returns_segments_with_role_and_force_lang() -> None:
    audio = np.ones(3200, dtype=np.float32) * 0.5  # loud enough to clear silence gate
    out = _transcribe_channel(
        _asr_backend(seg_language="de"),
        audio,
        language="de",
        speaker_id="SPEAKER_00",
        speaker_role="german",
        force_lang="de",
    )
    assert len(out) == 1
    assert out[0]["speaker_id"] == "SPEAKER_00"
    assert out[0]["speaker_role"] == "german"
    assert out[0]["lang"] == "de"
    assert out[0]["text"] == "hello"


def test_transcribe_channel_uses_detected_language_when_no_force() -> None:
    audio = np.ones(3200, dtype=np.float32) * 0.5
    out = _transcribe_channel(
        _asr_backend(seg_language="es"),
        audio,
        language=None,
        speaker_id="SPEAKER_01",
        speaker_role="foreign",
    )
    assert out[0]["lang"] == "es"


# ---------------------------------------------------------------------------
# run_translation
# ---------------------------------------------------------------------------


def test_run_translation_records_metric_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = SimpleNamespace(
        translate=lambda texts, src_lang, tgt_lang: [f"[{src_lang}->{tgt_lang}]" + texts[0]]
    )
    monkeypatch.setattr(streaming, "get_translation_backend", lambda: fake_backend)

    out = run_translation("Hello", src_lang="en", tgt_lang="de")
    assert out == "[en->de]Hello"
    assert len(streaming._metrics["mt_times"]) >= 1


# ---------------------------------------------------------------------------
# _delayed_viewer_speaking_off — exercise both branches
# ---------------------------------------------------------------------------


class _SilentWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


async def test_delayed_viewer_speaking_off_is_noop_when_token_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get(_token: str) -> None:
        return None

    monkeypatch.setattr(streaming.registry, "get", fake_get)
    # Should not raise even though token doesn't resolve.
    await _delayed_viewer_speaking_off("missing", 0.0)


async def test_delayed_viewer_speaking_off_finalizes_and_broadcasts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: pending live segments get force-finalized, the host
    receives a 'segments' update, viewers get the same broadcast, and the
    host gets a final 'speaking_state off' message."""
    main_ws = _SilentWS()
    viewer = _SilentWS()
    # Live segment that should be finalized.
    final_seg = Segment(
        id=0, abs_start=0.0, abs_end=1.0, src="hello", src_lang="en", final=False
    )

    fake_session = SimpleNamespace(
        segment_tracker=SimpleNamespace(
            force_finalize_all=lambda: [final_seg],
            finalized_segments=[final_seg],
        ),
        translation_queue=asyncio.Queue(),
        foreign_channel=SimpleNamespace(reset=lambda: None),
        translations={},
        is_dual_channel=lambda: False,
        get_current_time=lambda: 5.0,
        foreign_lang="en",
    )
    entry = SimpleNamespace(session=fake_session, main_ws=main_ws, viewers={viewer})

    async def fake_get(_token: str) -> Any:
        return entry

    monkeypatch.setattr(streaming.registry, "get", fake_get)

    await _delayed_viewer_speaking_off("tok123456", 0.0)

    # Translation queue got the newly-finalized segment.
    assert fake_session.translation_queue.qsize() == 1
    queued = fake_session.translation_queue.get_nowait()
    assert queued is final_seg

    # Host received both a 'segments' broadcast and a speaking_state-off.
    sent_types = {json.loads(m).get("type") for m in main_ws.sent}
    assert "segments" in sent_types
    assert "speaking_state" in sent_types
    # Viewer also received the segments broadcast.
    assert any(json.loads(m).get("type") == "segments" for m in viewer.sent)


async def test_delayed_viewer_speaking_off_handles_no_pending_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pending session (entry.session is None): just notify the host, no
    finalize-all and no segments broadcast."""
    main_ws = _SilentWS()
    entry = SimpleNamespace(session=None, main_ws=main_ws, viewers=set())

    async def fake_get(_token: str) -> Any:
        return entry

    monkeypatch.setattr(streaming.registry, "get", fake_get)

    await _delayed_viewer_speaking_off("tok123456", 0.0)
    sent_types = [json.loads(m).get("type") for m in main_ws.sent]
    assert "speaking_state" in sent_types
    assert "segments" not in sent_types


# ---------------------------------------------------------------------------
# Smoke for run_asr_german_channel via mocked backend (covers happy path)
# ---------------------------------------------------------------------------


def test_run_asr_german_channel_appends_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The German channel ASR loop should produce segments when audio is loud enough."""
    fake_seg = SimpleNamespace(start=0.0, end=1.0, text="Hallo", language="de")
    fake_result = SimpleNamespace(segments=[fake_seg])
    fake_backend = SimpleNamespace(
        get_bilingual_prompt=lambda _lang: None,
        transcribe=lambda audio, *, language=None, initial_prompt=None: fake_result,
    )

    s = StreamingSession()
    # 2.0s of moderately loud audio so silence gate passes.
    pcm = (np.ones(32000, dtype=np.int16) * 2000).tobytes()
    s.add_german_audio(pcm)

    with patch("app.streaming.get_asr_backend", return_value=fake_backend):
        all_segs, _newly = streaming.run_asr_german_channel(s)

    assert any(seg.src == "Hallo" for seg in all_segs)
