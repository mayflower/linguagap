"""Tests for the SessionRegistry primitive.

The registry is a small in-memory container with an asyncio.Lock — its
correctness is the foundation of viewer fan-out, so we exercise reserve /
activate / get / unregister and the viewer add/remove paths directly.
"""

from __future__ import annotations

import pytest

from app.session_registry import SessionRegistry


class _FakeWS:
    """Tiny stand-in for fastapi.WebSocket.

    SessionEntry.viewers is a WeakSet so its members must support weak
    references — bare ``object()`` instances do not, hence this class.
    """


@pytest.fixture
def fresh_registry() -> SessionRegistry:
    return SessionRegistry()


async def test_generate_token_is_url_safe_and_unique(fresh_registry: SessionRegistry) -> None:
    a = fresh_registry.generate_token()
    b = fresh_registry.generate_token()
    assert a != b
    assert "/" not in a and "+" not in a  # url-safe


async def test_reserve_creates_pending_entry(fresh_registry: SessionRegistry) -> None:
    assert await fresh_registry.reserve("tok-1") is True
    entry = await fresh_registry.get("tok-1")
    assert entry is not None
    assert entry.session is None  # pending, not active
    assert entry.is_active is False
    assert len(entry.viewers) == 0


async def test_reserve_twice_is_rejected(fresh_registry: SessionRegistry) -> None:
    await fresh_registry.reserve("tok-2")
    assert await fresh_registry.reserve("tok-2") is False


async def test_get_unknown_token_returns_none(fresh_registry: SessionRegistry) -> None:
    assert await fresh_registry.get("does-not-exist") is None


async def test_activate_sets_session_and_main_ws(fresh_registry: SessionRegistry) -> None:
    await fresh_registry.reserve("tok-3")
    fake_session = object()
    fake_ws = object()
    assert await fresh_registry.activate("tok-3", fake_session, fake_ws) is True  # type: ignore[arg-type]
    entry = await fresh_registry.get("tok-3")
    assert entry is not None
    assert entry.session is fake_session
    assert entry.main_ws is fake_ws
    assert entry.is_active is True


async def test_activate_creates_entry_when_not_reserved(fresh_registry: SessionRegistry) -> None:
    fake_session = object()
    fake_ws = object()
    await fresh_registry.activate("tok-4", fake_session, fake_ws)  # type: ignore[arg-type]
    entry = await fresh_registry.get("tok-4")
    assert entry is not None
    assert entry.session is fake_session


async def test_unregister_removes_entry(fresh_registry: SessionRegistry) -> None:
    await fresh_registry.reserve("tok-5")
    await fresh_registry.unregister("tok-5")
    assert await fresh_registry.get("tok-5") is None
    # Idempotent — unregister of a missing token must not raise.
    await fresh_registry.unregister("tok-5")


async def test_add_viewer_returns_false_for_unknown_token(
    fresh_registry: SessionRegistry,
) -> None:
    fake_ws = _FakeWS()
    assert await fresh_registry.add_viewer("never-reserved", fake_ws) is False  # type: ignore[arg-type]


async def test_add_remove_viewer_cycle(fresh_registry: SessionRegistry) -> None:
    await fresh_registry.reserve("tok-6")
    fake_ws_a = _FakeWS()
    fake_ws_b = _FakeWS()
    assert await fresh_registry.add_viewer("tok-6", fake_ws_a) is True  # type: ignore[arg-type]
    assert await fresh_registry.add_viewer("tok-6", fake_ws_b) is True  # type: ignore[arg-type]
    entry = await fresh_registry.get("tok-6")
    assert entry is not None
    assert set(entry.viewers) == {fake_ws_a, fake_ws_b}

    await fresh_registry.remove_viewer("tok-6", fake_ws_a)  # type: ignore[arg-type]
    assert set(entry.viewers) == {fake_ws_b}

    # Removing a viewer that is not in the set is a no-op.
    await fresh_registry.remove_viewer("tok-6", fake_ws_a)  # type: ignore[arg-type]
    assert set(entry.viewers) == {fake_ws_b}

    # Removing from an unknown token is also a no-op (no exception).
    await fresh_registry.remove_viewer("never-reserved", fake_ws_b)  # type: ignore[arg-type]
