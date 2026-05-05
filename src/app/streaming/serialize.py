"""Pure helpers that turn Segment objects into wire-format dicts.

These belong on the data-shape boundary between the streaming pipeline
and the WebSocket protocol; they have no I/O of their own and are easy
to test in isolation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from app.languages import LANG_INFO
from app.streaming_policy import Segment

if TYPE_CHECKING:
    from app.streaming.session import StreamingSession


def _role_from_lang(lang: str | None) -> str | None:
    """Map a source language to a semantic speaker role."""
    if lang == "de":
        return "german"
    if lang and lang != "unknown":
        return "foreign"
    return None


def _resolve_segment_role(segment: Segment, dual_channel: bool) -> str | None:
    """Resolve the role for a segment, preferring explicit role metadata."""
    if segment.speaker_role in {"german", "foreign"}:
        return segment.speaker_role

    if dual_channel:
        # In dual-channel mode, speaker IDs are fixed to roles by the pipeline.
        # If we don't have a known SPEAKER_xx, do NOT guess by language —
        # that flips the UI when Whisper misdetects.
        if segment.speaker_id == "SPEAKER_00":
            return "german"
        if segment.speaker_id == "SPEAKER_01":
            return "foreign"
        return None

    return _role_from_lang(segment.src_lang)


def _serialize_segments(session: StreamingSession, segments: list[Segment]) -> list[dict]:
    """Convert Segment objects to dicts with resolved roles and translations."""
    dual_channel = session.is_dual_channel()
    result = []
    for seg in segments:
        seg_dict = asdict(seg)
        speaker_role = _resolve_segment_role(seg, dual_channel)
        seg_dict["speaker_role"] = speaker_role

        # Override src_lang if role is certain, to prevent Whisper's
        # misdetections from breaking the translation logic.
        if speaker_role == "german":
            seg_dict["src_lang"] = "de"
        elif (
            speaker_role == "foreign" and session.foreign_lang and session.foreign_lang in LANG_INFO
        ):
            seg_dict["src_lang"] = session.foreign_lang

        seg_dict["translations"] = session.translations.get(seg.id, {})
        result.append(seg_dict)
    return result


def _resolve_translation_pair(
    segment: Segment,
    role: str | None,
    foreign_lang: str | None,
) -> tuple[str, str] | None:
    """Determine (src_lang, tgt_lang) for a segment, or None to skip translation.

    Returning None means the MT loop will not enqueue a translation for this
    segment — either because we have no foreign language configured, or
    because the source and target are identical (e.g. a German-only
    transcription session where ``foreign_lang == "de"``).
    """
    foreign = foreign_lang if foreign_lang in LANG_INFO else None

    if role == "german":
        if not foreign or foreign == "de":
            return None
        return "de", foreign

    if role == "foreign":
        # Foreign channel always translates to German. Use the session's
        # foreign_lang, not segment.src_lang which may be misdetected.
        if not foreign or foreign == "de":
            return None
        return foreign, "de"

    src = segment.src_lang
    if src == "de" and foreign and foreign != "de":
        return src, foreign
    if src == "de":
        return None
    return src, "de"
