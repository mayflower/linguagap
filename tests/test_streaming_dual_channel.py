"""Tests for run_asr_dual_channel — the dual-mic ASR path.

We mock the ASR backend to return fixed segments per call so we can drive
crosstalk suppression, language auto-detect, and bleed filtering branches
without spinning up Whisper.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from app import streaming
from app.streaming import StreamingSession, run_asr_dual_channel


def _seg(start: float, end: float, text: str, language: str) -> Any:
    return SimpleNamespace(start=start, end=end, text=text, language=language)


def _backend(german_segs: list[Any], foreign_segs: list[Any]) -> Any:
    """Return an ASR backend mock whose transcribe() picks per call.

    The first transcribe() call goes to the german channel (always called
    first by run_asr_dual_channel), the second to the foreign channel —
    we honour that ordering by popping from a queued list.
    """
    queue = [german_segs, foreign_segs]

    def transcribe(audio, *, language=None, initial_prompt=None):  # type: ignore[no-untyped-def]
        segs = queue.pop(0) if queue else []
        return SimpleNamespace(segments=segs)

    return SimpleNamespace(
        get_bilingual_prompt=lambda _lang: None,
        transcribe=transcribe,
    )


def _loud_pcm(samples: int, amplitude_int16: int = 4000) -> bytes:
    arr = np.full(samples, amplitude_int16, dtype=np.int16)
    return arr.tobytes()


def test_dual_channel_short_audio_returns_only_finalized() -> None:
    """If both channels are below the 1600-sample threshold, no ASR runs."""
    s = StreamingSession()
    s.add_german_audio(_loud_pcm(800))  # 0.05s
    s.add_foreign_audio(_loud_pcm(800))

    fake = _backend([], [])
    with patch("app.streaming.get_asr_backend", return_value=fake):
        all_segs, newly = run_asr_dual_channel(s)

    assert all_segs == []  # no finalized segments yet
    assert newly == []


def test_dual_channel_emits_segments_for_both_speakers() -> None:
    s = StreamingSession()
    # 2 seconds @ 16kHz, loud — passes silence + length gates.
    s.add_german_audio(_loud_pcm(32000))
    s.add_foreign_audio(_loud_pcm(32000))

    # Segments are placed at non-overlapping times so SegmentTracker doesn't
    # treat the second one as a time-duplicate of the first.
    fake = _backend(
        german_segs=[_seg(0.0, 0.8, "Hallo", "de")],
        foreign_segs=[_seg(1.2, 1.9, "Hello", "en")],
    )
    with patch("app.streaming.get_asr_backend", return_value=fake):
        all_segs, _ = run_asr_dual_channel(s)

    texts = [seg.src for seg in all_segs]
    assert "Hallo" in texts
    assert "Hello" in texts


def test_dual_channel_filters_bleed_when_german_text_appears_in_foreign() -> None:
    """Duplicate text on the foreign channel should be dropped."""
    s = StreamingSession()
    s.add_german_audio(_loud_pcm(32000))
    s.add_foreign_audio(_loud_pcm(32000))

    fake = _backend(
        german_segs=[_seg(0.0, 1.0, "Hallo", "de")],
        # Foreign channel happens to transcribe identical German text — drop it.
        foreign_segs=[_seg(0.0, 1.0, "Hallo", "de")],
    )
    with patch("app.streaming.get_asr_backend", return_value=fake):
        all_segs, _ = run_asr_dual_channel(s)

    texts = [seg.src for seg in all_segs]
    # Bleed should be filtered — only one segment carries that text.
    assert texts.count("Hallo") <= 1


def test_dual_channel_autodetects_foreign_lang_when_unset() -> None:
    """If session.foreign_lang is None and the foreign channel reports a
    valid Whisper language, we lock it onto the session."""
    s = StreamingSession(src_lang="auto")
    assert s.foreign_lang is None
    s.add_german_audio(_loud_pcm(32000))
    s.add_foreign_audio(_loud_pcm(32000))

    fake = _backend(
        german_segs=[_seg(0.0, 1.0, "Hallo", "de")],
        foreign_segs=[_seg(0.0, 1.0, "Hello", "fr")],
    )
    with patch("app.streaming.get_asr_backend", return_value=fake):
        run_asr_dual_channel(s)

    assert s.foreign_lang == "fr"


def test_dual_channel_does_not_overwrite_existing_foreign_lang() -> None:
    s = StreamingSession()
    s.foreign_lang = "es"
    s.add_german_audio(_loud_pcm(32000))
    s.add_foreign_audio(_loud_pcm(32000))

    fake = _backend(
        german_segs=[_seg(0.0, 1.0, "Hallo", "de")],
        foreign_segs=[_seg(0.0, 1.0, "Hello", "fr")],  # mistaken
    )
    with patch("app.streaming.get_asr_backend", return_value=fake):
        run_asr_dual_channel(s)

    assert s.foreign_lang == "es"


def test_dual_channel_records_asr_and_tick_metrics() -> None:
    s = StreamingSession()
    s.add_german_audio(_loud_pcm(32000))
    s.add_foreign_audio(_loud_pcm(32000))

    initial_asr = len(streaming._metrics["asr_times"])
    initial_tick = len(streaming._metrics["tick_times"])

    fake = _backend(
        german_segs=[_seg(0.0, 1.0, "Hallo", "de")],
        foreign_segs=[],
    )
    with patch("app.streaming.get_asr_backend", return_value=fake):
        run_asr_dual_channel(s)

    assert len(streaming._metrics["asr_times"]) > initial_asr
    assert len(streaming._metrics["tick_times"]) > initial_tick
