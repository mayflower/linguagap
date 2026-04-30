"""ASR/MT pipeline functions invoked by the WebSocket handler.

These run synchronously in a ``ThreadPoolExecutor`` because the underlying
backends (faster-whisper, llama-cpp-python) are blocking. The handler
schedules them via ``loop.run_in_executor`` so the event loop stays
responsive to incoming WebSocket frames.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.backends import get_asr_backend, get_translation_backend
from app.languages import LANG_INFO
from app.streaming.session import StreamingSession
from app.streaming_policy import Segment

logger = logging.getLogger(__name__)

# Crosstalk suppression: when both mics share a room, one picks up the
# other's speaker. Suppress the quieter channel only when (a) the louder
# side is clearly active and (b) it is much louder than the other.
_CROSSTALK_MIN_ACTIVE_RMS = 0.02
_CROSSTALK_DOMINANCE_RATIO = 4.0

_executor = ThreadPoolExecutor(max_workers=2)

# Shared metrics dict — populated here and read by ``get_metrics`` /
# the package-level ``_metrics`` re-export. Lives in the package root so
# tests that monkeypatch ``streaming._metrics`` reach the same dict.
from app.streaming._metrics import _metrics  # noqa: E402


def _is_effective_silence(
    audio: np.ndarray, rms_threshold: float = 0.02, peak_threshold: float = 0.05
) -> bool:
    """Detect near-silent buffers that should not be sent to ASR."""
    if len(audio) == 0:
        return True
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    return rms < rms_threshold and peak < peak_threshold


def _build_prompt(backend, language: str | None, finalized_text: str = "") -> str | None:
    """Build ASR prompt from bilingual hint + recent finalized transcript."""
    prompt = backend.get_bilingual_prompt(language) if language else None
    if finalized_text:
        # Whisper prompt is limited to ~224 tokens; keep last ~200 chars
        context = finalized_text[-200:].strip()
        prompt = f"{prompt} {context}" if prompt else context
    return prompt


def _transcribe_channel(
    backend,
    audio: np.ndarray,
    language: str | None,
    speaker_id: str,
    speaker_role: str,
    force_lang: str | None = None,
    finalized_text: str = "",
) -> list[dict]:
    """Transcribe a full channel buffer via the ASR backend."""
    if len(audio) < 1600:
        return []
    if _is_effective_silence(audio):
        return []

    prompt = _build_prompt(backend, language, finalized_text)
    asr_result = backend.transcribe(audio, language=language, initial_prompt=prompt)
    filtered = asr_result.segments

    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "speaker_id": speaker_id,
            "lang": force_lang or seg.language,
            "speaker_role": speaker_role,
        }
        for seg in filtered
    ]


def run_asr_german_channel(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """Run ASR in deterministic desktop mode.

    When the foreign language is known and only desktop audio is active,
    treat desktop input as German-only to prevent role/language drift.
    """
    tick_start = time.time()
    now_sec = session.get_current_time()
    german_audio, german_offset = session.get_german_window_audio()

    if len(german_audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    backend = get_asr_backend()
    asr_start = time.time()

    # Only seed the German prompt with prior German history — leftover
    # foreign-role segments from a past dual-channel phase would bias Whisper
    # the wrong way.
    recent_final = session.segment_tracker.finalized_segments[-10:]
    finalized_text = " ".join(s.src for s in recent_final if s.speaker_role != "foreign")[-500:]
    german_results = _transcribe_channel(
        backend,
        german_audio,
        "de",
        "SPEAKER_00",
        "german",
        force_lang="de",
        finalized_text=finalized_text,
    )

    for seg in german_results:
        seg["start"] += german_offset
        seg["end"] += german_offset

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)
    logger.debug("German-channel ASR: %d segments in %.1fms", len(german_results), asr_time * 1000)

    session.resolve_foreign_lang()

    session.detected_lang = "de"
    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=german_results,
        window_start=0.0,
        now_sec=now_sec,
        src_lang="de",
    )

    tick_time = time.time() - tick_start
    _metrics["tick_times"].append(tick_time)
    return all_segments, newly_finalized


def run_asr_dual_channel(session: StreamingSession) -> tuple[list[Segment], list[Segment]]:
    """Run ASR with separate German and foreign audio channels.

    Used when the viewer is actively sending audio (dual-channel mode).
    Much simpler than run_asr() — no diarization or language detection
    needed since each channel is a known speaker with a known language.
    """
    tick_start = time.time()
    now_sec = session.get_current_time()

    german_audio, german_offset = session.get_german_window_audio()
    foreign_audio, foreign_offset = session.get_foreign_window_audio()

    if len(german_audio) < 1600 and len(foreign_audio) < 1600:
        return list(session.segment_tracker.finalized_segments), []

    backend = get_asr_backend()

    # Why dominance-based, not floor-based: a phone mic across a desk picks up
    # the host at 0.05–0.1 RMS — loud enough to bleed through, not loud enough
    # to trip an absolute floor.
    german_rms = float(np.sqrt(np.mean(german_audio**2))) if len(german_audio) > 0 else 0.0
    foreign_rms = float(np.sqrt(np.mean(foreign_audio**2))) if len(foreign_audio) > 0 else 0.0

    if (
        german_rms > _CROSSTALK_MIN_ACTIVE_RMS
        and german_rms > foreign_rms * _CROSSTALK_DOMINANCE_RATIO
    ):
        logger.debug(
            "Suppressing foreign channel (crosstalk from german, rms %.3f vs %.3f)",
            german_rms,
            foreign_rms,
        )
        foreign_audio = np.array([], dtype=np.float32)
    elif (
        foreign_rms > _CROSSTALK_MIN_ACTIVE_RMS
        and foreign_rms > german_rms * _CROSSTALK_DOMINANCE_RATIO
    ):
        logger.debug(
            "Suppressing german channel (crosstalk from foreign, rms %.3f vs %.3f)",
            foreign_rms,
            german_rms,
        )
        german_audio = np.array([], dtype=np.float32)

    logger.debug(
        "Dual-channel pipeline: german=%d samples (rms=%.3f), foreign=%d samples (rms=%.3f)",
        len(german_audio),
        german_rms,
        len(foreign_audio),
        foreign_rms,
    )

    asr_start = time.time()

    # Channel-local finalized history. Whisper is strongly biased by its
    # initial_prompt: feeding the foreign channel German context early in a
    # session (when only German has finalized) makes it hallucinate a German
    # paraphrase of English audio despite language="en" being forced. Keep
    # each channel's prompt monolingual to prevent cross-language drift.
    recent_final = session.segment_tracker.finalized_segments[-10:]
    german_history = " ".join(s.src for s in recent_final if s.speaker_role == "german")[-500:]
    foreign_history = " ".join(s.src for s in recent_final if s.speaker_role == "foreign")[-500:]

    german_results = _transcribe_channel(
        backend,
        german_audio,
        "de",
        "SPEAKER_00",
        "german",
        force_lang="de",
        finalized_text=german_history,
    )
    foreign_results = _transcribe_channel(
        backend,
        foreign_audio,
        session.foreign_lang,
        "SPEAKER_01",
        "foreign",
        force_lang=session.foreign_lang,
        finalized_text=foreign_history,
    )

    for seg in german_results:
        seg["start"] += german_offset
        seg["end"] += german_offset
    for seg in foreign_results:
        seg["start"] += foreign_offset
        seg["end"] += foreign_offset

    # force_lang on both channels ensures correct labels.
    # Only fall back to Whisper detection if foreign_lang was never configured.
    if session.foreign_lang is None:
        for seg in foreign_results:
            lang = seg.get("lang")
            if lang and lang not in ("unknown", "de") and lang in LANG_INFO:
                session.foreign_lang = lang
                logger.info("Dual-channel: foreign language detected as %s", lang)
                break

    # Filter cross-channel bleed: drop foreign segments that duplicate german ones
    if german_results and foreign_results:
        german_texts = {" ".join(seg["text"].lower().split()) for seg in german_results}
        foreign_results = [
            seg
            for seg in foreign_results
            if " ".join(seg["text"].lower().split()) not in german_texts
        ]

    hyp_segments = german_results + foreign_results
    hyp_segments.sort(key=lambda s: s["start"])

    asr_time = time.time() - asr_start
    _metrics["asr_times"].append(asr_time)

    logger.debug("Dual-channel ASR: %d segments in %.1fms", len(hyp_segments), asr_time * 1000)

    session.detected_lang = session.foreign_lang or "unknown"

    all_segments, newly_finalized = session.segment_tracker.update_from_hypothesis(
        hyp_segments=hyp_segments,
        window_start=0.0,
        now_sec=now_sec,
        src_lang=session.detected_lang,
    )

    tick_time = time.time() - tick_start
    _metrics["tick_times"].append(tick_time)

    return all_segments, newly_finalized


def run_translation(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate a single text from src_lang to tgt_lang."""
    mt_start = time.time()
    mt_backend = get_translation_backend()
    result = mt_backend.translate([text], src_lang=src_lang, tgt_lang=tgt_lang)[0]
    mt_time = time.time() - mt_start
    _metrics["mt_times"].append(mt_time)
    return result
