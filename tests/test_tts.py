"""Tests for the Piper TTS module.

We avoid loading any real Piper voice (downloads ONNX models from HF and
spins up onnxruntime sessions) by mocking get_voice with a stub that
emits a small AudioChunk-shaped object. This keeps the tests fast and
hermetic while still exercising _synthesize_pcm16, synthesize_wav, the
WAV header builder, the metrics deque, and the error paths.
"""

from __future__ import annotations

import struct
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from app import tts


@pytest.fixture(autouse=True)
def reset_metrics():
    """Clear cumulative TTS timing metrics between tests."""
    tts._metrics["tts_times"] = deque(maxlen=100)
    yield


def test_supported_langs_excludes_unsupported() -> None:
    # 15 voices configured today; hr/Croatian intentionally absent.
    assert "en" in tts.TTS_SUPPORTED_LANGS
    assert "de" in tts.TTS_SUPPORTED_LANGS
    assert "hr" not in tts.TTS_SUPPORTED_LANGS
    assert tts.TTS_SUPPORTED_LANGS == set(tts.PIPER_VOICES.keys())


def test_make_wav_header_describes_payload() -> None:
    pcm = b"\x00\x00" * 1000  # 1000 silent samples
    wav = tts._make_wav(pcm, sample_rate=16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    # The reported size in the header must match (file_len - 8).
    declared_size = struct.unpack("<I", wav[4:8])[0]
    assert declared_size == len(wav) - 8
    # data chunk size matches PCM payload length.
    data_size = struct.unpack("<I", wav[40:44])[0]
    assert data_size == len(pcm)
    assert wav[44:] == pcm


def test_synthesize_speech_rejects_unknown_lang() -> None:
    with pytest.raises(ValueError, match="not supported"):
        tts.synthesize_speech("hello", lang="xx")


def test_synthesize_wav_rejects_unknown_lang() -> None:
    with pytest.raises(ValueError, match="not supported"):
        tts.synthesize_wav("hello", lang="xx")


def _stub_voice(sample_rate: int = 22050) -> Any:
    chunk = SimpleNamespace(
        sample_rate=sample_rate,
        audio_int16_bytes=b"\x01\x00\x02\x00\x03\x00",  # 3 PCM16 samples
    )
    return SimpleNamespace(synthesize=lambda _text: iter([chunk]))


def test_synthesize_speech_returns_chunks_and_records_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts, "get_voice", lambda lang: _stub_voice())

    pcm = tts.synthesize_speech("hello", lang="en")

    assert pcm == b"\x01\x00\x02\x00\x03\x00"
    assert len(tts._metrics["tts_times"]) == 1
    assert tts._metrics["tts_times"][0] >= 0.0


def test_synthesize_wav_wraps_with_correct_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts, "get_voice", lambda lang: _stub_voice(sample_rate=24000))

    wav = tts.synthesize_wav("hello", lang="de")

    # Sample rate field lives at byte offset 24 in a PCM WAV header.
    sr = struct.unpack("<I", wav[24:28])[0]
    assert sr == 24000


def test_get_tts_metrics_summary_after_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    # No calls yet -> zeroed summary.
    summary = tts.get_tts_metrics()
    assert summary == {"avg_tts_time_ms": 0, "tts_sample_count": 0}

    monkeypatch.setattr(tts, "get_voice", lambda lang: _stub_voice())
    tts.synthesize_speech("a", lang="en")
    tts.synthesize_speech("b", lang="de")

    summary = tts.get_tts_metrics()
    assert summary["tts_sample_count"] == 2
    assert summary["avg_tts_time_ms"] >= 0


def test_synthesize_handles_empty_chunk_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Piper yields no chunks (degenerate input), we still return cleanly."""
    monkeypatch.setattr(
        tts,
        "get_voice",
        lambda lang: SimpleNamespace(synthesize=lambda _t: iter(())),
    )

    wav = tts.synthesize_wav("", lang="en")
    # Header is still well-formed and the data chunk has zero length.
    assert wav[:4] == b"RIFF"
    data_size = struct.unpack("<I", wav[40:44])[0]
    assert data_size == 0
