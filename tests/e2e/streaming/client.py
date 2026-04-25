"""WebSocket streaming client for E2E tests.

Streams audio to linguagap WebSocket API and collects results.
"""

import asyncio
import contextlib
import json
import os
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

# Audio framing: 100 ms PCM16 chunks at the configured sample rate.
# Real production clients send 100 ms frames so the ASR sliding window sees
# steady input. ``REALTIME_FACTOR`` < 1 plays the file faster than realtime.
FRAME_DURATION_MS = 100
# Margin after the audio finishes streaming before requesting the summary.
# ASR runs every 0.5s and each call takes ~1.6s. Worst case the last frame
# arrives mid-call, so wait: current ASR (~1.6s) + next tick (~0.5s) +
# next ASR (~1.6s), with headroom for system load.
POST_STREAM_DRAIN_SEC = 8.0


@dataclass
class StreamingResult:
    """Results collected from a streaming session.

    Attributes:
        segments: List of transcription segments
        translations: Dict mapping segment_id to translations
        summary: Summary result if requested
        errors: List of error messages
        duration_sec: Total audio duration streamed
    """

    segments: list[dict] = field(default_factory=list)
    translations: dict[int, dict[str, str]] = field(default_factory=dict)
    summary: dict | None = None
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def final_segments(self) -> list[dict]:
        """Get only finalized segments."""
        return [s for s in self.segments if s.get("final")]

    @property
    def detected_languages(self) -> set[str]:
        """Get unique languages detected in segments."""
        return {s.get("src_lang", "unknown") for s in self.segments}

    @property
    def detected_speakers(self) -> set[str]:
        """Get unique speaker IDs detected in segments."""
        return {s["speaker_id"] for s in self.segments if s.get("speaker_id")}


class StreamingClient:
    """Client for streaming audio to linguagap WebSocket API."""

    def __init__(
        self,
        ws_url: str | None = None,
        realtime_factor: float = 0.5,
        sample_rate: int = 16000,
    ):
        """Initialize the streaming client.

        Args:
            ws_url: WebSocket URL. If None, uses LINGUAGAP_WS_URL env var.
            realtime_factor: Speed factor (0.5 = 2x faster than realtime)
            sample_rate: Audio sample rate (default 16kHz)
        """
        self.ws_url = ws_url or os.getenv("LINGUAGAP_WS_URL", "ws://localhost:8000/ws")
        self.realtime_factor = realtime_factor
        self.sample_rate = sample_rate

    async def stream_audio_file(
        self,
        audio_path: str | Path,
        src_lang: str = "auto",
        foreign_lang: str | None = None,
        request_summary: bool = True,
        timeout_sec: float = 300.0,
    ) -> StreamingResult:
        """Stream an audio file through the WebSocket API.

        Args:
            audio_path: Path to WAV audio file
            src_lang: Source language hint ("auto" for detection)
            foreign_lang: Optional foreign-language hint for bilingual sessions
            request_summary: Whether to request a summary at the end
            timeout_sec: Maximum time to wait for processing

        Returns:
            StreamingResult with collected data
        """
        import websockets

        result = StreamingResult()

        with wave.open(str(Path(audio_path)), "rb") as wav:
            file_sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_data = wav.readframes(n_frames)
            result.duration_sec = n_frames / file_sample_rate

        samples_per_frame = int(self.sample_rate * FRAME_DURATION_MS / 1000)
        bytes_per_frame = samples_per_frame * 2  # 16-bit audio
        frame_delay = (FRAME_DURATION_MS / 1000) * self.realtime_factor

        config: dict = {
            "type": "config",
            "sample_rate": self.sample_rate,
            "src_lang": src_lang,
            "token": str(uuid.uuid4()),
        }
        if foreign_lang is not None:
            config["foreign_lang"] = foreign_lang

        async with websockets.connect(self.ws_url) as ws:
            await ws.send(json.dumps(config))

            try:
                ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
                ack_data = json.loads(ack)
                if ack_data.get("type") != "config_ack":
                    result.errors.append(f"Unexpected config response: {ack_data}")
            except TimeoutError:
                result.errors.append("Timeout waiting for config acknowledgment")
                return result

            receive_done = asyncio.Event()
            receiver = asyncio.create_task(_receive_messages(ws, result, receive_done))

            for offset in range(0, len(audio_data), bytes_per_frame):
                frame = audio_data[offset : offset + bytes_per_frame]
                if len(frame) < bytes_per_frame:
                    frame = frame + b"\x00" * (bytes_per_frame - len(frame))
                await ws.send(frame)
                await asyncio.sleep(frame_delay)

            await asyncio.sleep(POST_STREAM_DRAIN_SEC)

            if request_summary:
                await ws.send(json.dumps({"type": "request_summary"}))
                try:
                    await asyncio.wait_for(receive_done.wait(), timeout=timeout_sec)
                except TimeoutError:
                    result.errors.append("Timeout waiting for summary")

            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver

        return result

    async def stream_dialogue(
        self,
        audio_path: str | Path,
        request_summary: bool = True,
    ) -> StreamingResult:
        """Stream a dialogue audio file using "auto" language detection."""
        return await self.stream_audio_file(
            audio_path=audio_path,
            src_lang="auto",
            request_summary=request_summary,
        )


async def _receive_messages(ws, result: StreamingResult, receive_done: asyncio.Event) -> None:
    """Consume WS messages and accumulate them into ``result`` until summary/error."""
    try:
        async for message in ws:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "segments":
                result.segments = data.get("segments", [])
            elif msg_type == "translation":
                seg_id = data.get("segment_id")
                if seg_id is not None:
                    result.translations.setdefault(seg_id, {})[data.get("tgt_lang")] = data.get(
                        "text"
                    )
            elif msg_type == "summary":
                result.summary = data
                receive_done.set()
            elif msg_type == "summary_error":
                result.errors.append(f"Summary error: {data.get('error')}")
                receive_done.set()
            elif msg_type == "error":
                result.errors.append(data.get("message", str(data)))
    except Exception as e:
        if "ConnectionClosed" not in str(type(e)):
            result.errors.append(f"Receive error: {e}")
