"""Tests for the default behaviours on the abstract backend base classes.

The base classes provide a small set of default implementations
(transcribe_file fallback, post_process no-op, supports_language always
True, get_language_fallback identity, get_bilingual_prompt None,
supports_language_pair always True). We exercise them through trivial
concrete subclasses so the implementations are exposed without bringing
in faster_whisper or llama_cpp.
"""

from __future__ import annotations

import numpy as np

from app.backends.base import ASRBackend, SummarizationBackend, TranslationBackend
from app.backends.types import ASRResult, ASRSegment


class _DummyASR(ASRBackend):
    """Minimal subclass that exposes the default helpers."""

    def __init__(self) -> None:
        self.last_audio: np.ndarray | None = None

    def load_model(self) -> None:  # pragma: no cover - inert
        return None

    def warmup(self) -> None:  # pragma: no cover - inert
        return None

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        language: str | None = None,  # noqa: ARG002
        initial_prompt: str | None = None,  # noqa: ARG002
    ) -> ASRResult:
        self.last_audio = audio
        return ASRResult(
            segments=[ASRSegment(start=0.0, end=1.0, text="hi", language="en")],
            detected_language="en",
            language_probability=1.0,
        )


class _DummyMT(TranslationBackend):
    def load_model(self) -> None:  # pragma: no cover
        return None

    def warmup(self) -> None:  # pragma: no cover
        return None

    def translate(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        return [f"{src_lang}->{tgt_lang}:{t}" for t in texts]


class _DummySumm(SummarizationBackend):
    def load_model(self) -> None:  # pragma: no cover
        return None

    def warmup(self) -> None:  # pragma: no cover
        return None

    def summarize_bilingual(
        self,
        segments: list[dict],  # noqa: ARG002
        foreign_lang: str,  # noqa: ARG002
    ) -> tuple[str, str]:
        return ("foreign", "german")


def test_default_post_process_is_identity() -> None:
    backend = _DummyASR()
    segs = [ASRSegment(start=0.0, end=1.0, text="x", language="en")]
    assert backend.post_process(segs) == segs


def test_default_supports_language_returns_true_for_anything() -> None:
    backend = _DummyASR()
    assert backend.supports_language("en") is True
    assert backend.supports_language("xx-INVALID") is True


def test_default_get_language_fallback_returns_input() -> None:
    backend = _DummyASR()
    assert backend.get_language_fallback("en") == "en"
    assert backend.get_language_fallback("zz") == "zz"


def test_default_get_bilingual_prompt_is_none() -> None:
    assert _DummyASR().get_bilingual_prompt("en") is None


def test_default_translation_supports_language_pair_is_true() -> None:
    assert _DummyMT().supports_language_pair("en", "de") is True
    assert _DummyMT().supports_language_pair("xx", "yy") is True


def test_translation_dummy_translate_round_trip() -> None:
    out = _DummyMT().translate(["hello", "world"], src_lang="en", tgt_lang="de")
    assert out == ["en->de:hello", "en->de:world"]


def test_summarization_dummy_returns_pair() -> None:
    foreign, german = _DummySumm().summarize_bilingual([{"src_lang": "de", "src": "x"}], "en")
    assert (foreign, german) == ("foreign", "german")


def test_default_transcribe_file_resamples_when_needed(monkeypatch, tmp_path) -> None:
    """ASRBackend.transcribe_file delegates to transcribe() after resampling.

    We mock soundfile.read to return audio at 8 kHz so transcribe_file
    triggers the resampy resample branch.
    """
    import sys
    from types import ModuleType

    fake_sf = ModuleType("soundfile")
    fake_sf.read = lambda _path, dtype="float32": (  # type: ignore[attr-defined]
        np.ones(8000, dtype=np.float32),
        8000,
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    fake_resampy = ModuleType("resampy")
    captured: dict[str, int] = {}

    def fake_resample(audio, src_sr, dst_sr):  # type: ignore[no-untyped-def]
        captured["src_sr"] = src_sr
        captured["dst_sr"] = dst_sr
        # Return any 16k-shaped buffer; the dummy backend doesn't care.
        return np.zeros(16000, dtype=np.float32)

    fake_resampy.resample = fake_resample  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "resampy", fake_resampy)

    backend = _DummyASR()
    result = backend.transcribe_file(str(tmp_path / "x.wav"))

    assert result.detected_language == "en"
    assert captured == {"src_sr": 8000, "dst_sr": 16000}
    # transcribe() received the resampled buffer (16k samples).
    assert backend.last_audio is not None
    assert backend.last_audio.shape == (16000,)


def test_default_transcribe_file_skips_resample_at_16k(monkeypatch, tmp_path) -> None:
    import sys
    from types import ModuleType

    fake_sf = ModuleType("soundfile")
    fake_sf.read = lambda _path, dtype="float32": (  # type: ignore[attr-defined]
        np.ones(16000, dtype=np.float32),
        16000,
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)

    backend = _DummyASR()
    result = backend.transcribe_file(str(tmp_path / "x.wav"))
    assert result.segments[0].text == "hi"
