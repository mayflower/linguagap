"""Viewer broadcast helpers.

Centralizes the pattern of sending a JSON message to every viewer of a
session and pruning dead WebSocket connections, so callers (handler, MT
loop, viewer-side disconnect cleanup) never roll their own version.
"""

from __future__ import annotations

import json

from fastapi import WebSocket

from app.session_registry import SessionEntry, registry


async def broadcast_to_viewers(entry: SessionEntry, message: dict) -> None:
    """Broadcast JSON message to all viewers, remove dead connections."""
    if not entry.viewers:
        return

    message_json = json.dumps(message)
    dead_viewers: list[WebSocket] = []

    for viewer_ws in entry.viewers:
        try:
            await viewer_ws.send_text(message_json)
        except Exception:
            dead_viewers.append(viewer_ws)

    for viewer_ws in dead_viewers:
        entry.viewers.discard(viewer_ws)


async def _maybe_broadcast(session_token: str | None, message: dict) -> None:
    """Broadcast to viewers if session_token is set and session exists."""
    if session_token:
        entry = await registry.get(session_token)
        if entry:
            await broadcast_to_viewers(entry, message)
