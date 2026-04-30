"""WebSocketHandler for the host-side streaming session."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Mapping
from typing import Any

from fastapi import WebSocket

from app.backends import get_summarization_backend
from app.languages import LANG_INFO
from app.session_registry import registry
from app.streaming import trace as trace_mod
from app.streaming._metrics import _metrics  # noqa: F401 - re-exported via package
from app.streaming.asr import (
    _executor,
    run_asr_dual_channel,
    run_asr_german_channel,
    run_translation,
)
from app.streaming.broadcast import _maybe_broadcast, broadcast_to_viewers
from app.streaming.serialize import (
    _resolve_segment_role,
    _resolve_translation_pair,
    _serialize_segments,
)
from app.streaming.session import TICK_SEC, StreamingSession
from app.streaming_policy import STABILITY_SEC, Segment

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """Manages the lifecycle of a single WebSocket streaming session.

    Coordinates ASR and MT async loops, handles incoming messages (config,
    audio, request_summary), and broadcasts updates to viewers.

    Message protocol:
        Client → Server:
            - {"type": "config", "sample_rate": 16000, "src_lang": "auto", "token": "..."}
            - Binary PCM16 audio frames (16kHz mono)
            - {"type": "request_summary"} (stops recording, drains translations)

        Server → Client:
            - {"type": "config_ack", "status": "active"}
            - {"type": "segments", "segments": [...], "src_lang": "...", "foreign_lang": "..."}
            - {"type": "translation", "segment_id": N, "tgt_lang": "de", "text": "..."}
    """

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket
        self.session: StreamingSession | None = None
        self.session_token: str | None = None
        self._asr_task: asyncio.Task | None = None
        self._mt_task: asyncio.Task | None = None
        self._running = True
        self._translation_queue: asyncio.Queue[Segment] = asyncio.Queue()
        self._msg_count = 0
        self._bytes_received = 0
        self._host_speaking_off_task: asyncio.Task | None = None
        # Lifecycle tracing state: remember the last text+final seen per
        # segment id so we can distinguish first-emit from update events,
        # and remember enqueue timestamps to report MT queue-wait durations.
        self._trace_seen: dict[int, tuple[str, bool]] = {}
        self._trace_mt_enq: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Accept the WebSocket and run the message loop until disconnect."""
        await self.websocket.accept()
        try:
            while True:
                message = await self.websocket.receive()
                self._msg_count += 1

                if message["type"] == "websocket.disconnect":
                    logger.info(
                        "WebSocket disconnect after %d msgs, %d bytes",
                        self._msg_count,
                        self._bytes_received,
                    )
                    break

                await self._handle_message(message)

        except Exception as e:
            logger.error("WebSocket error: %s", e)
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, message: Mapping[str, Any]) -> None:
        if "text" in message:
            data = json.loads(message["text"])
            msg_type = data.get("type")
            if msg_type == "config":
                await self._handle_config(data)
            elif msg_type == "request_summary":
                await self._handle_request_summary()
            elif msg_type == "ptt_mode":
                await self._handle_ptt_mode(data)
            elif msg_type == "speaking_state":
                await self._handle_host_speaking_state(data)
            elif msg_type == "host_transcript_requested":
                await self._handle_host_transcript_requested(data)
        elif "bytes" in message:
            self._handle_audio(message["bytes"])

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    async def _handle_config(self, data: dict) -> None:
        sample_rate = data.get("sample_rate", 16000)
        src_lang = data.get("src_lang", "auto")
        foreign_lang = data.get("foreign_lang")

        self.session_token = data.get("token")
        if not self.session_token:
            logger.error("No token provided in config message")
            return

        self.session = StreamingSession(sample_rate=sample_rate, src_lang=src_lang)
        self.session.translation_queue = self._translation_queue
        if foreign_lang and foreign_lang not in ("auto", "de", "unknown"):
            self.session.foreign_lang = foreign_lang
        # Honour a pre-activation PTT toggle so the session_active broadcast
        # below already carries the right mode and the viewer doesn't flicker
        # between non-PTT and PTT UI.
        self.session.ptt_mode = bool(data.get("ptt_mode", False))

        self._asr_task = asyncio.create_task(self._asr_loop())
        self._mt_task = asyncio.create_task(self._mt_loop())

        await registry.activate(self.session_token, self.session, self.websocket)
        logger.info(
            "Session activated: sample_rate=%d, src_lang=%s, foreign_lang=%s, session=%s...",
            sample_rate,
            src_lang,
            foreign_lang or "auto",
            self.session_token[:8],
        )

        trace_mod._trace(
            "session_start",
            tok=self._trace_tok(),
            src_lang=src_lang,
            foreign_lang=foreign_lang or "auto",
            ptt=self.session.ptt_mode,
        )
        await self._send_json({"type": "config_ack", "status": "active"})

        entry = await registry.get(self.session_token)
        # A freshly-activated StreamingSession defaults transcript_consent=False.
        # Carry pre-activation viewer consent from the entry onto the new
        # session so the post-stop download button renders for the host.
        # Host notification is best-effort — a failure here must not abort
        # activation or block the viewer-side session_active broadcast.
        if entry and entry.transcript_consent:
            self.session.transcript_consent = True
            try:
                await self._send_json({"type": "transcript_consent", "enabled": True})
            except Exception as exc:  # noqa: BLE001 - best-effort relay
                logger.warning(
                    "transcript_consent host notify failed for token=%s: %s",
                    self.session_token[:8] if self.session_token else "?",
                    exc,
                )
        if entry and entry.viewers:
            await broadcast_to_viewers(
                entry,
                {
                    "type": "session_active",
                    "foreign_lang": self.session.foreign_lang,
                    "segments": [],
                    "ptt_mode": self.session.ptt_mode,
                },
            )

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _handle_audio(self, audio_bytes: bytes) -> None:
        if self.session is not None:
            self._bytes_received += len(audio_bytes)
            self.session.add_german_audio(audio_bytes)
            if self._msg_count % 50 == 0:
                logger.debug(
                    "Audio: %d bytes, %.1fs buffered",
                    self._bytes_received,
                    self.session.get_buffered_seconds(),
                )

    # ------------------------------------------------------------------
    # Push-to-Talk + speaking state
    # ------------------------------------------------------------------

    async def _handle_ptt_mode(self, data: dict) -> None:
        if self.session is None:
            return
        enabled = bool(data.get("enabled", False))
        self.session.ptt_mode = enabled
        logger.info("PTT mode %s", "enabled" if enabled else "disabled")
        await _maybe_broadcast(self.session_token, {"type": "ptt_mode", "enabled": enabled})

    async def _handle_host_transcript_requested(self, data: dict) -> None:
        if self.session_token is None:
            return
        enabled = bool(data.get("enabled", False))
        entry = await registry.get(self.session_token)
        if entry is not None:
            entry.host_transcript_requested = enabled
            # If the host turns the request off, the prior viewer consent
            # becomes irrelevant for this run — reset it so a later toggle-on
            # re-prompts the viewer explicitly.
            if not enabled:
                entry.transcript_consent = False
                if entry.session is not None:
                    entry.session.transcript_consent = False
        logger.info("Host transcript requested: %s", enabled)
        await _maybe_broadcast(
            self.session_token,
            {"type": "host_transcript_requested", "enabled": enabled},
        )

    async def _handle_host_speaking_state(self, data: dict) -> None:
        if self.session is None or self.session_token is None:
            return
        speaking = bool(data.get("speaking", False))
        if speaking:
            # Cancel any pending "speaking off" relay
            if self._host_speaking_off_task and not self._host_speaking_off_task.done():
                self._host_speaking_off_task.cancel()
                self._host_speaking_off_task = None
            await _maybe_broadcast(
                self.session_token,
                {"type": "speaking_state", "party": "host", "speaking": True},
            )
        else:
            delay = STABILITY_SEC + TICK_SEC
            self._host_speaking_off_task = asyncio.create_task(
                self._delayed_speaking_broadcast("host", delay)
            )

    async def _delayed_speaking_broadcast(self, party: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            # In PTT mode, audio stops when the key/button is released, so
            # get_current_time() (audio-based) no longer advances past the last
            # segment's end time. Without this, live segments never meet the
            # STABILITY_SEC finalization threshold and stay stuck as live.
            await self._finalize_pending_segments()
            await _maybe_broadcast(
                self.session_token,
                {"type": "speaking_state", "party": party, "speaking": False},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Failed to broadcast speaking_state off for %s", party, exc_info=True)

    async def _finalize_pending_segments(self) -> None:
        """Force-finalize live segments and queue them for translation.

        Used on PTT release: without new audio, get_current_time() freezes and
        live segments never finalize via the normal time-based threshold.
        Also drains the host (German) channel buffer so the next press starts
        from clean audio — otherwise the 8s sliding window keeps replaying the
        just-released utterance into subsequent transcripts.
        """
        if self.session is None:
            return
        newly_final = self.session.segment_tracker.force_finalize_all()
        await self._enqueue_for_translation(newly_final, reason="ptt_release")
        # Drop any audio still buffered for the host channel — the next PTT
        # press should start from a clean window. Done unconditionally because
        # even if no segment finalized this round, sub-second trailing audio
        # would otherwise bleed into the next utterance.
        self.session.german_channel.reset()
        trace_mod._trace("channel_reset", tok=self._trace_tok(), party="host")
        if newly_final:
            all_segments = list(self.session.segment_tracker.finalized_segments)
            segments_data = _serialize_segments(self.session, all_segments)
            segments_msg = {
                "type": "segments",
                "t": self.session.get_current_time(),
                "src_lang": self.session.foreign_lang or "unknown",
                "foreign_lang": self.session.foreign_lang,
                "dual_channel": self.session.is_dual_channel(),
                "segments": segments_data,
            }
            await self._send_and_broadcast(segments_msg)

    # ------------------------------------------------------------------
    # Request summary (stop recording + summarize)
    # ------------------------------------------------------------------

    async def _handle_request_summary(self) -> None:
        trace_mod._trace(
            "stop_requested", tok=self._trace_tok(), qsize=self._translation_queue.qsize()
        )
        await self._cancel_task("_asr_task")
        logger.debug("ASR loop stopped for stop request")

        if self.session is None:
            return

        now_sec = self.session.get_current_time()
        all_segs, time_finalized = self.session.segment_tracker.update_from_hypothesis(
            [], 0.0, now_sec, "unknown"
        )

        tok = self._trace_tok()
        if time_finalized:
            logger.debug("Queuing %d time-finalized segments", len(time_finalized))
            await self._enqueue_for_translation(time_finalized, reason="stop_time")

        live_segs = [s for s in all_segs if not s.final]
        if live_segs:
            logger.debug("Force-finalizing %d live segments", len(live_segs))
            newly_final = self.session.segment_tracker.force_finalize_all()
            await self._enqueue_for_translation(newly_final, reason="stop_force")

        trace_mod._trace("stop_drain_begin", tok=tok, qsize=self._translation_queue.qsize())
        try:
            await asyncio.wait_for(self._translation_queue.join(), timeout=60)
            trace_mod._trace("stop_drain_done", tok=tok)
        except TimeoutError:
            logger.warning(
                "Translation queue join timed out, %d items remaining",
                self._translation_queue.qsize(),
            )
            trace_mod._trace(
                "stop_drain_timeout",
                tok=tok,
                remaining=self._translation_queue.qsize(),
            )

        # Send final segments with all translations. Broadcast to viewers FIRST
        # so their transcript download state is complete even if the host has
        # already disconnected mid-stop; then tolerate a host-send failure.
        finalized = self.session.segment_tracker.finalized_segments
        if finalized:
            segments_data = _serialize_segments(self.session, list(finalized))
            segments_msg = {
                "type": "segments",
                "t": self.session.get_current_time(),
                "src_lang": self.session.detected_lang or "unknown",
                "foreign_lang": self.session.foreign_lang or "en",
                "segments": segments_data,
            }
            await _maybe_broadcast(self.session_token, segments_msg)
            try:
                await self._send_json(segments_msg)
            except Exception as exc:  # noqa: BLE001 - host may be mid-disconnect
                logger.warning(
                    "Final segments host send failed for token=%s: %s",
                    self.session_token[:8] if self.session_token else "?",
                    exc,
                )

        await self._run_summarization(finalized)

        self._running = False
        logger.info("Recording stopped, closing connection")
        await self.websocket.close()

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    async def _run_summarization(self, finalized: list[Segment]) -> None:
        summ_backend = get_summarization_backend()
        if summ_backend is None or not finalized:
            return

        assert self.session is not None
        foreign_lang = self.session.foreign_lang or "en"
        summ_segments = [
            {
                "src": seg.src,
                "src_lang": seg.src_lang,
                "translations": self.session.translations.get(seg.id, {}),
            }
            for seg in finalized
        ]

        await self._send_json(
            {
                "type": "summary_progress",
                "step": "summarize",
                "message": "Generating summaries...",
            }
        )

        try:
            loop = asyncio.get_event_loop()
            foreign_summary, german_summary = await loop.run_in_executor(
                _executor,
                summ_backend.summarize_bilingual,
                summ_segments,
                foreign_lang,
            )
            summary_msg = {
                "type": "summary",
                "foreign_summary": foreign_summary,
                "german_summary": german_summary,
                "foreign_lang": foreign_lang,
            }
            await self._send_json(summary_msg)
            await _maybe_broadcast(self.session_token, summary_msg)
        except Exception as e:
            logger.error("Summarization error: %s", e)

    # ------------------------------------------------------------------
    # ASR loop
    # ------------------------------------------------------------------

    async def _asr_loop(self) -> None:
        """Process audio windows every TICK_SEC and send segment updates."""
        tick_count = 0
        last_segment_hash = None
        while self._running:
            await asyncio.sleep(TICK_SEC)
            if self.session is None or not self._running:
                continue

            tick_count += 1
            if tick_count <= 3 or tick_count % 20 == 0:
                logger.debug(
                    "ASR tick #%d: %.1fs buffered",
                    tick_count,
                    self.session.get_buffered_seconds(),
                )
            loop = asyncio.get_event_loop()
            try:
                if self.session.is_dual_channel():
                    asr_fn = run_asr_dual_channel
                else:
                    # Before the remote viewer joins, the web UI is the ONLY source of audio.
                    # Since the web UI is STRICTLY the German speaker's device, we must
                    # process this audio exclusively as German. Falling back to `run_asr`
                    # with diarization would cause the German speaker to be misidentified
                    # as 'foreign' if Whisper misdetects the language.
                    asr_fn = run_asr_german_channel

                all_segments, newly_finalized = await loop.run_in_executor(
                    _executor, asr_fn, self.session
                )

                if not self._running:
                    continue

                tok = self._trace_tok()
                for seg in all_segments:
                    prev = self._trace_seen.get(seg.id)
                    key = (seg.src, seg.final)
                    if prev is None:
                        trace_mod._trace(
                            "asr_emit",
                            tok=tok,
                            seg=seg.id,
                            final=seg.final,
                            src_lang=seg.src_lang,
                            role=seg.speaker_role,
                            t=f"{seg.abs_start:.2f}-{seg.abs_end:.2f}",
                            text=seg.src,
                        )
                    elif prev != key:
                        trace_mod._trace(
                            "asr_update",
                            tok=tok,
                            seg=seg.id,
                            final=seg.final,
                            src_lang=seg.src_lang,
                            text=seg.src,
                        )
                    self._trace_seen[seg.id] = key

                segments_data = _serialize_segments(self.session, all_segments)

                current_hash = tuple(
                    (
                        s["id"],
                        s["src"],
                        s.get("src_lang"),
                        s.get("speaker_role"),
                        s["final"],
                        tuple(sorted(s["translations"].items())),
                    )
                    for s in segments_data
                )
                if current_hash != last_segment_hash:
                    last_segment_hash = current_hash
                    segments_msg = {
                        "type": "segments",
                        "t": self.session.get_current_time(),
                        "src_lang": self.session.detected_lang or "unknown",
                        "foreign_lang": self.session.foreign_lang,
                        "dual_channel": self.session.is_dual_channel(),
                        "segments": segments_data,
                    }
                    trace_mod._trace(
                        "ws_segments",
                        tok=tok,
                        count=len(segments_data),
                        ids=",".join(str(s["id"]) for s in segments_data),
                    )
                    await self._send_and_broadcast(segments_msg)

                if newly_finalized:
                    logger.debug("Queuing %d segments for translation", len(newly_finalized))
                    await self._enqueue_for_translation(newly_finalized)

                if all_segments:
                    final_count = sum(1 for s in all_segments if s.final)
                    live_count = len(all_segments) - final_count
                    logger.debug("Segments: %d final, %d live", final_count, live_count)

            except Exception as e:
                logger.error("ASR tick error: %s", e)

    # ------------------------------------------------------------------
    # MT loop
    # ------------------------------------------------------------------

    async def _mt_loop(self) -> None:
        """Consume the translation queue and send updates when ready."""
        while self._running:
            try:
                segment = await asyncio.wait_for(
                    self._translation_queue.get(),
                    timeout=0.5,
                )
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(
                    "MT loop queue error (%s): %r",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                await asyncio.sleep(0.1)
                continue

            if not self._running or self.session is None:
                self._translation_queue.task_done()
                break

            tok = self._trace_tok()
            enq = self._trace_mt_enq.pop(segment.id, None)
            wait_ms = int((time.time() - enq) * 1000) if enq is not None else -1
            trace_mod._trace("mt_get", tok=tok, seg=segment.id, wait_ms=wait_ms)

            tgt_lang: str | None = None
            try:
                role = _resolve_segment_role(segment, self.session.is_dual_channel())
                pair = _resolve_translation_pair(
                    segment,
                    role,
                    self.session.foreign_lang,
                )

                if pair is None:
                    trace_mod._trace(
                        "mt_skip", tok=tok, seg=segment.id, reason="no_pair", role=role
                    )
                    continue
                seg_src_lang, tgt_lang = pair

                if seg_src_lang not in LANG_INFO:
                    logger.warning(
                        "Skipping translation for segment %d: unsupported source language '%s'",
                        segment.id,
                        seg_src_lang,
                    )
                    trace_mod._trace(
                        "mt_skip",
                        tok=tok,
                        seg=segment.id,
                        reason="bad_src_lang",
                        src_lang=seg_src_lang,
                    )
                    continue

                if self.session.translations.get(segment.id, {}).get(tgt_lang):
                    trace_mod._trace(
                        "mt_skip",
                        tok=tok,
                        seg=segment.id,
                        reason="cached",
                        tgt_lang=tgt_lang,
                    )
                    continue

                logger.debug(
                    "Translating segment %d (%s→%s): %s",
                    segment.id,
                    seg_src_lang,
                    tgt_lang,
                    segment.src[:50],
                )
                trace_mod._trace(
                    "mt_start",
                    tok=tok,
                    seg=segment.id,
                    src_lang=seg_src_lang,
                    tgt_lang=tgt_lang,
                    text=segment.src,
                )
                mt_t0 = time.time()
                loop = asyncio.get_event_loop()
                translation = await loop.run_in_executor(
                    _executor,
                    run_translation,
                    segment.src,
                    seg_src_lang,
                    tgt_lang,
                )
                mt_dur_ms = int((time.time() - mt_t0) * 1000)
                logger.debug("Translation done %d: %s", segment.id, translation[:50])
                trace_mod._trace(
                    "mt_done",
                    tok=tok,
                    seg=segment.id,
                    tgt_lang=tgt_lang,
                    dur_ms=mt_dur_ms,
                    text=translation,
                )

                if segment.id not in self.session.translations:
                    self.session.translations[segment.id] = {}
                self.session.translations[segment.id][tgt_lang] = translation

                if self._running:
                    translation_msg = {
                        "type": "translation",
                        "segment_id": segment.id,
                        "tgt_lang": tgt_lang,
                        "text": translation,
                    }
                    trace_mod._trace(
                        "ws_translation",
                        tok=tok,
                        seg=segment.id,
                        tgt_lang=tgt_lang,
                    )
                    await self._send_and_broadcast(translation_msg)

            except Exception as e:
                logger.error(
                    "Translation error for segment %d (%s): %r",
                    segment.id,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
                trace_mod._trace(
                    "mt_error",
                    tok=tok,
                    seg=segment.id,
                    tgt_lang=tgt_lang,
                    err=f"{type(e).__name__}: {e}",
                )
                if self._running:
                    error_msg = {
                        "type": "translation_error",
                        "segment_id": segment.id,
                        "tgt_lang": tgt_lang,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    with contextlib.suppress(Exception):
                        await self._send_and_broadcast(error_msg)
            finally:
                self._translation_queue.task_done()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _trace_tok(self) -> str:
        """Short token identifier for trace lines (first 8 chars of session token)."""
        return self.session_token[:8] if self.session_token else "?"

    async def _enqueue_for_translation(
        self, segments: list[Segment], reason: str | None = None
    ) -> None:
        """Trace and queue finalized segments for the MT loop."""
        if not segments:
            return
        tok = self._trace_tok()
        for seg in segments:
            self._trace_mt_enq[seg.id] = time.time()
            fields = {
                "tok": tok,
                "seg": seg.id,
                "src_lang": seg.src_lang,
                "role": seg.speaker_role,
                "text": seg.src,
            }
            if reason is not None:
                fields["reason"] = reason
            trace_mod._trace("asr_final", **fields)
            trace_mod._trace("mt_put", tok=tok, seg=seg.id, qsize=self._translation_queue.qsize())
            await self._translation_queue.put(seg)

    async def _send_json(self, message: dict) -> None:
        await self.websocket.send_text(json.dumps(message))

    async def _send_and_broadcast(self, message: dict) -> None:
        await self._send_json(message)
        await _maybe_broadcast(self.session_token, message)

    async def _cancel_task(self, attr: str) -> None:
        task: asyncio.Task | None = getattr(self, attr)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            setattr(self, attr, None)

    async def _cleanup(self) -> None:
        self._running = False
        await _maybe_broadcast(self.session_token, {"type": "session_ended"})
        if self.session_token is not None:
            await registry.unregister(self.session_token)
        await self._cancel_task("_asr_task")
        await self._cancel_task("_mt_task")
        await self._cancel_task("_host_speaking_off_task")


async def handle_websocket(websocket: WebSocket) -> None:
    """Main WebSocket handler — thin wrapper around WebSocketHandler."""
    handler = WebSocketHandler(websocket)
    await handler.run()
