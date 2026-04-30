"""Read-only viewer WebSocket endpoint + delayed-off helper."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging

from fastapi import WebSocket

from app.session_registry import registry
from app.streaming import trace as trace_mod
from app.streaming.broadcast import broadcast_to_viewers
from app.streaming.serialize import _serialize_segments
from app.streaming.session import TICK_SEC
from app.streaming_policy import STABILITY_SEC

logger = logging.getLogger(__name__)


async def _delayed_viewer_speaking_off(token: str, delay: float) -> None:
    """Send delayed 'viewer stopped speaking' to host, looking up fresh WS from registry.

    Also force-finalizes pending segments — in PTT mode, audio stops on release
    so get_current_time() stops advancing and live segments never finalize.
    """
    try:
        await asyncio.sleep(delay)
        entry = await registry.get(token)
        if not entry:
            return
        # Force-finalize pending segments so they get translated and marked final
        if entry.session is not None:
            newly_final = entry.session.segment_tracker.force_finalize_all()
            # Queue newly-finalized segments for translation — without this,
            # the viewer's PTT-captured foreign speech gets transcribed but
            # never translated to German.
            if newly_final and entry.session.translation_queue is not None:
                for seg in newly_final:
                    await entry.session.translation_queue.put(seg)
            # Drain the foreign channel buffer so the next viewer PTT press
            # starts from clean audio. Without this the 8s sliding window keeps
            # replaying the just-released utterance into the next transcript
            # (observed bleed: "Can you hear me clearly?" reappearing as a
            # prefix of the *next* viewer segment).
            entry.session.foreign_channel.reset()
            trace_mod._trace("channel_reset", tok=token[:8], party="viewer")
            if newly_final and entry.main_ws:
                all_segments = list(entry.session.segment_tracker.finalized_segments)
                segments_data = _serialize_segments(entry.session, all_segments)
                segments_msg = {
                    "type": "segments",
                    "t": entry.session.get_current_time(),
                    "src_lang": entry.session.foreign_lang or "unknown",
                    "foreign_lang": entry.session.foreign_lang,
                    "dual_channel": entry.session.is_dual_channel(),
                    "segments": segments_data,
                }
                msg_json = json.dumps(segments_msg)
                with contextlib.suppress(Exception):
                    await entry.main_ws.send_text(msg_json)
                await broadcast_to_viewers(entry, segments_msg)
        if entry.main_ws:
            trace_mod._trace("viewer_speaking_off_delayed", tok=token[:8])
            await entry.main_ws.send_text(
                json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False})
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Failed to relay viewer speaking_state off to host", exc_info=True)


async def handle_viewer_websocket(websocket: WebSocket, token: str) -> None:
    """Handle a read-only viewer WebSocket connection."""
    await websocket.accept()

    # Add viewer to session (creates pending entry if doesn't exist)
    if not await registry.add_viewer(token, websocket):
        # Token not reserved - reserve it now for the viewer
        await registry.reserve(token)
        await registry.add_viewer(token, websocket)

    logger.info("Viewer connected to session %s...", token[:8])

    try:
        # Check if session is already active
        entry = await registry.get(token)
        if entry and entry.session is not None:
            # Session is active, send current state
            session = entry.session
            live_segments = [cs.segment for cs in session.segment_tracker.cumulative_segments]
            all_segments = list(session.segment_tracker.finalized_segments) + live_segments
            segments_data = _serialize_segments(session, all_segments)

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "init",
                        "status": "active",
                        "foreign_lang": session.foreign_lang,
                        "segments": segments_data,
                        "ptt_mode": session.ptt_mode,
                    }
                )
            )
            # Inform late-joining viewer about an active host transcript request
            if entry.host_transcript_requested:
                await websocket.send_text(
                    json.dumps({"type": "host_transcript_requested", "enabled": True})
                )
        else:
            # Session is pending - tell viewer to wait
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "init",
                        "status": "waiting",
                        "message": "Waiting for recording to start...",
                    }
                )
            )
            if entry and entry.host_transcript_requested:
                await websocket.send_text(
                    json.dumps({"type": "host_transcript_requested", "enabled": True})
                )

        # Cache session reference for audio routing
        cached_session = None
        viewer_speaking_off_task: asyncio.Task | None = None

        # Main loop: handle pong/keepalive, audio frames, and viewer config
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                if message["type"] == "websocket.disconnect":
                    break

                # Handle binary audio from viewer mic
                if "bytes" in message:
                    # Lazily resolve session reference
                    if cached_session is None:
                        entry = await registry.get(token)
                        if entry and entry.session:
                            cached_session = entry.session
                    if cached_session is not None:
                        cached_session.add_foreign_audio(message["bytes"])

                # Handle text messages (pong, viewer_audio_config)
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        msg_type = data.get("type")

                        if msg_type == "viewer_audio_config":
                            # Viewer sends foreign language hint
                            viewer_foreign_lang = data.get("foreign_lang")
                            if viewer_foreign_lang and viewer_foreign_lang not in (
                                "de",
                                "unknown",
                                "auto",
                            ):
                                if cached_session is None:
                                    entry = await registry.get(token)
                                    if entry and entry.session:
                                        cached_session = entry.session
                                if cached_session is not None:
                                    if cached_session.foreign_lang != viewer_foreign_lang:
                                        logger.info(
                                            "Viewer updated foreign language: %s -> %s",
                                            cached_session.foreign_lang,
                                            viewer_foreign_lang,
                                        )
                                        # Reset cached role/lang inferences to re-lock speakers with new hint.
                                    cached_session.foreign_lang = viewer_foreign_lang

                        elif msg_type == "transcript_consent":
                            # Viewer opted in to the bilingual transcript download.
                            # Consent is stored on the SessionEntry (not the
                            # session) so it survives the pre-activation window
                            # where the host hasn't started recording yet.
                            enabled = bool(data.get("enabled", False))
                            entry = await registry.get(token)
                            if entry and enabled:
                                entry.transcript_consent = True
                                if entry.session is not None:
                                    entry.session.transcript_consent = True
                            if entry and entry.main_ws:
                                msg = {
                                    "type": "transcript_consent",
                                    "enabled": entry.transcript_consent,
                                }
                                try:
                                    await entry.main_ws.send_text(json.dumps(msg))
                                except Exception as exc:  # noqa: BLE001 - relay best-effort
                                    logger.warning(
                                        "Failed to relay transcript_consent to host for "
                                        "token=%s: %s",
                                        token[:8],
                                        exc,
                                        exc_info=True,
                                    )

                        elif msg_type == "speaking_state":
                            # Relay viewer speaking state to host
                            entry = await registry.get(token)
                            if entry and entry.main_ws:
                                speaking = bool(data.get("speaking", False))
                                trace_mod._trace(
                                    "viewer_speaking",
                                    tok=token[:8],
                                    speaking=speaking,
                                )
                                msg = {
                                    "type": "speaking_state",
                                    "party": "viewer",
                                    "speaking": speaking,
                                }
                                if speaking:
                                    if (
                                        viewer_speaking_off_task
                                        and not viewer_speaking_off_task.done()
                                    ):
                                        viewer_speaking_off_task.cancel()
                                        viewer_speaking_off_task = None
                                    try:
                                        await entry.main_ws.send_text(json.dumps(msg))
                                    except Exception:
                                        logger.warning(
                                            "Failed to relay viewer speaking_state to host"
                                        )
                                else:
                                    viewer_speaking_off_task = asyncio.create_task(
                                        _delayed_viewer_speaking_off(
                                            token, STABILITY_SEC + TICK_SEC
                                        )
                                    )

                        # pong is implicitly handled (no action needed)
                    except (json.JSONDecodeError, KeyError):
                        pass

            except TimeoutError:
                # Send ping
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break

    except Exception as e:
        logger.error("Viewer websocket error: %s", e)
    finally:
        # Cancel any pending delayed-off task — we send the speaking=false
        # below directly, which makes the delay redundant and prevents the
        # host indicator from latching ON forever if the viewer disconnected
        # mid-PTT-press without releasing.
        if viewer_speaking_off_task and not viewer_speaking_off_task.done():
            viewer_speaking_off_task.cancel()
        entry = await registry.get(token)
        if entry and entry.main_ws:
            trace_mod._trace("viewer_speaking_off_disconnect", tok=token[:8])
            with contextlib.suppress(Exception):
                await entry.main_ws.send_text(
                    json.dumps({"type": "speaking_state", "party": "viewer", "speaking": False})
                )
        await registry.remove_viewer(token, websocket)
        logger.info("Viewer disconnected from session %s...", token[:8])
