"""Audio buffering + per-session state for the streaming pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.streaming_policy import Segment, SegmentTracker

logger = logging.getLogger(__name__)

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "8.0"))
TICK_SEC = float(os.getenv("TICK_SEC", "0.5"))
MAX_BUFFER_SEC = float(os.getenv("MAX_BUFFER_SEC", "30.0"))


@dataclass
class ChannelBuffer:
    """Audio buffer for a single channel (German or foreign speaker)."""

    audio: deque[bytes] = field(default_factory=deque)
    total_samples: int = 0
    trimmed_samples: int = 0
    start_offset_sec: float = 0.0
    started: bool = False

    def reset(self) -> None:
        """Drop buffered audio so the next utterance starts from a clean slate.

        Used after PTT release: without this, the 8s sliding ASR window keeps
        re-transcribing the just-finalized utterance every tick and the
        trailing words bleed into the *next* PTT press's transcript. Resetting
        ``started`` lets the next add_audio call re-anchor ``start_offset_sec``
        to the current wall-clock — combined with reset counters this keeps
        segment timestamps strictly ascending so SegmentTracker won't merge
        new segments with stale finalized ones.
        """
        self.audio.clear()
        self.total_samples = 0
        self.trimmed_samples = 0
        self.started = False


class StreamingSession:
    """Manages state for a single streaming transcription session.

    Maintains the audio buffer, segment tracking, and per-speaker language
    detection for a WebSocket connection. Each browser tab creates one session.

    Attributes:
        sample_rate: Audio sample rate (typically 16000 Hz)
        src_lang: User-selected source language or "auto" for detection
        audio_buffer: Rolling buffer of PCM16 audio bytes
        detected_lang: Currently detected language from ASR
        foreign_lang: The non-German language in the conversation
        segment_tracker: Tracks segment finalization state
        translations: Cached translations keyed by segment_id and language
    """

    def __init__(self, sample_rate: int = 16000, src_lang: str = "auto"):
        self.sample_rate = sample_rate
        self.src_lang = src_lang  # User-selected source language (or "auto")
        self.audio_buffer: deque[bytes] = deque()
        self.total_samples = 0
        self.start_time = time.time()  # Wall-clock time when session started
        self.detected_lang: str | None = None  # Currently detected language from ASR
        self.foreign_lang: str | None = (
            None  # The non-German language (auto-detected or user-selected)
        )
        self.segment_tracker = SegmentTracker()
        self.dropped_frames = 0
        self.trimmed_samples = 0  # Track total samples trimmed from buffer start
        self.translations: dict[int, dict[str, str]] = {}  # segment_id -> {lang -> translation}

        # Dual-channel buffers (German mic on main page, foreign mic on viewer)
        self.german_channel = ChannelBuffer()
        self.foreign_channel = ChannelBuffer()
        self.viewer_last_audio_time: float = 0.0
        self.dual_channel_locked: bool = False
        self.ptt_mode: bool = False
        # True once at least one viewer has explicitly consented to the
        # bilingual transcript download. Sticky: a later viewer opting out
        # does not revoke an earlier consent from another viewer.
        self.transcript_consent: bool = False
        # Set by WebSocketHandler after session creation so non-handler code
        # (e.g. viewer WebSocket loop) can queue segments for translation.
        self.translation_queue: asyncio.Queue[Segment] | None = None

    def add_audio(self, pcm16_bytes: bytes):
        self.audio_buffer.append(pcm16_bytes)
        self.total_samples += len(pcm16_bytes) // 2
        self._enforce_max_buffer()

    def _enforce_max_buffer(self) -> None:
        trimmed = self._trim_buffer(self.audio_buffer)
        if trimmed > 0:
            self.trimmed_samples += trimmed
            self.dropped_frames += 1

    def _trim_buffer(self, buffer: deque[bytes]) -> int:
        """Trim a buffer to MAX_BUFFER_SEC. Returns number of samples trimmed."""
        max_samples = int(MAX_BUFFER_SEC * self.sample_rate)
        all_bytes = b"".join(buffer)
        total_samples = len(all_bytes) // 2

        if total_samples > max_samples:
            excess_samples = total_samples - max_samples
            trimmed_bytes = all_bytes[excess_samples * 2 :]
            buffer.clear()
            buffer.append(trimmed_bytes)
            return excess_samples
        return 0

    def _add_channel_audio(self, pcm16_bytes: bytes, channel: ChannelBuffer) -> None:
        """Add audio to a channel buffer with first-chunk offset tracking."""
        if not channel.started:
            chunk_sec = (len(pcm16_bytes) // 2) / self.sample_rate
            elapsed = max(0.0, time.time() - self.start_time)
            channel.start_offset_sec = max(0.0, elapsed - chunk_sec)
            channel.started = True
        channel.audio.append(pcm16_bytes)
        channel.total_samples += len(pcm16_bytes) // 2
        trimmed = self._trim_buffer(channel.audio)
        channel.trimmed_samples += trimmed
        self.add_audio(pcm16_bytes)

    def add_german_audio(self, pcm16_bytes: bytes) -> None:
        """Add audio from the main page (German speaker) to the german channel buffer."""
        self._add_channel_audio(pcm16_bytes, self.german_channel)

    def add_foreign_audio(self, pcm16_bytes: bytes) -> None:
        """Add audio from the viewer (foreign speaker) to the foreign channel buffer."""
        self._add_channel_audio(pcm16_bytes, self.foreign_channel)
        self.viewer_last_audio_time = time.time()
        self.dual_channel_locked = True

    def _get_channel_window_audio(self, channel: ChannelBuffer) -> tuple[np.ndarray, float]:
        """Get windowed audio from a channel buffer as float32 array."""
        all_bytes = b"".join(channel.audio)
        if not all_bytes:
            return np.array([], dtype=np.float32), 0.0
        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        max_window_samples = int(WINDOW_SEC * self.sample_rate)
        if len(samples) > max_window_samples:
            samples = samples[-max_window_samples:]
        window_start = channel.start_offset_sec + (
            max(0, channel.total_samples - len(samples)) / self.sample_rate
        )
        return samples, window_start

    def get_german_window_audio(self) -> tuple[np.ndarray, float]:
        """Get the German channel audio buffer as a float32 array."""
        return self._get_channel_window_audio(self.german_channel)

    def get_foreign_window_audio(self) -> tuple[np.ndarray, float]:
        """Get the foreign channel audio buffer as a float32 array."""
        return self._get_channel_window_audio(self.foreign_channel)

    def resolve_foreign_lang(self) -> None:
        """Set foreign_lang from user selection if not yet auto-detected."""
        if self.foreign_lang is None and self.src_lang not in ("auto", "de"):
            self.foreign_lang = self.src_lang
            logger.info("Foreign language set from user selection: %s", self.foreign_lang)

    def is_dual_channel(self) -> bool:
        """Use dual-channel mode once foreign mic audio has been received at least once."""
        return self.dual_channel_locked

    def get_window_audio(self) -> tuple[np.ndarray, float]:
        """Get audio for ASR processing.

        Returns the full audio buffer to ensure complete coverage. Previous
        sliding window logic caused gaps when audio arrived faster than
        real-time (e.g., test scenarios with pre-recorded audio).

        Performance note: The diarization-first pipeline runs ASR only on
        individual speaker segments, not the full audio, so processing time
        scales with speech duration rather than total buffer duration.

        Buffer size is capped at MAX_BUFFER_SEC (default 30s) by
        _enforce_max_buffer(), which trims oldest audio when exceeded.
        """
        all_bytes = b"".join(self.audio_buffer)
        samples = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        window_start = self.trimmed_samples / self.sample_rate
        return samples, window_start

    def get_current_time(self) -> float:
        if self.is_dual_channel():
            german_end = self.german_channel.start_offset_sec + (
                self.german_channel.total_samples / self.sample_rate
            )
            foreign_end = self.foreign_channel.start_offset_sec + (
                self.foreign_channel.total_samples / self.sample_rate
            )
            return max(german_end, foreign_end)
        return self.total_samples / self.sample_rate

    def get_buffered_seconds(self) -> float:
        if self.is_dual_channel():
            german_sec = len(b"".join(self.german_channel.audio)) / 2 / self.sample_rate
            foreign_sec = len(b"".join(self.foreign_channel.audio)) / 2 / self.sample_rate
            return max(german_sec, foreign_sec)
        all_bytes = b"".join(self.audio_buffer)
        return len(all_bytes) / 2 / self.sample_rate
