"""Tests for the Qwen3 summarization backend.

We never load the real GGUF weights; instead we monkeypatch the lazy
_get_llm() to return a stub that produces a canned chat completion. This
exercises the prompt builder, response parser, and <think>...</think>
stripping without requiring llama-cpp at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.backends.summarization.qwen3 import (
    Qwen3SummarizationBackend,
    _strip_think_block,
)

# ---------------------------------------------------------------------------
# _strip_think_block
# ---------------------------------------------------------------------------


def test_strip_think_block_removes_complete_block() -> None:
    text = "<think>reasoning here</think>Final summary."
    assert _strip_think_block(text) == "Final summary."


def test_strip_think_block_removes_incomplete_block() -> None:
    # Truncated <think> with no closing tag — should drop everything from <think>.
    text = "Visible prefix.\n<think>partial reasoning that got cut off"
    assert _strip_think_block(text) == "Visible prefix."


def test_strip_think_block_returns_placeholder_on_empty() -> None:
    text = "<think>only-thinking</think>"
    assert _strip_think_block(text) == "(Summary generation incomplete)"


def test_strip_think_block_passes_through_clean_text() -> None:
    assert _strip_think_block("Just a normal answer.") == "Just a normal answer."


def test_strip_think_block_keeps_prefix_when_close_missing_and_text_empty_after() -> None:
    text = "Useful header.\n<think>cut"
    assert _strip_think_block(text) == "Useful header."


# ---------------------------------------------------------------------------
# summarize_bilingual — fully mocked llama_cpp
# ---------------------------------------------------------------------------


def _make_backend_with_response(response_text: str) -> Qwen3SummarizationBackend:
    captured: dict[str, Any] = {}

    def fake_create(messages: Any, **_kwargs: Any) -> dict:
        captured["messages"] = messages
        return {"choices": [{"message": {"content": response_text}}]}

    backend = Qwen3SummarizationBackend()
    backend._llm = SimpleNamespace(create_chat_completion=fake_create)  # type: ignore[assignment]
    backend._captured = captured  # type: ignore[attr-defined]
    return backend


def test_summarize_bilingual_parses_two_section_response() -> None:
    response = (
        "ENGLISH: The host welcomed the guest and asked about school.\n"
        "GERMAN: Der Gastgeber begrüßte den Gast und fragte nach der Schule."
    )
    backend = _make_backend_with_response(response)

    foreign, german = backend.summarize_bilingual(
        segments=[
            {"src_lang": "de", "src": "Hallo, willkommen."},
            {"src_lang": "en", "src": "Thanks, my child needs to enrol."},
        ],
        foreign_lang="en",
    )

    assert "host welcomed" in foreign.lower()
    assert "gastgeber begrüßte" in german.lower()


def test_summarize_bilingual_strips_think_block_before_parsing() -> None:
    response = (
        "<think>Let me reason about this conversation...</think>"
        "ENGLISH: Quick chat.\n"
        "GERMAN: Kurzer Plausch."
    )
    backend = _make_backend_with_response(response)

    foreign, german = backend.summarize_bilingual(
        segments=[{"src_lang": "de", "src": "Hi"}],
        foreign_lang="en",
    )
    assert foreign == "Quick chat."
    assert german == "Kurzer Plausch."


def test_summarize_bilingual_falls_back_when_format_missing() -> None:
    """When the LLM ignores the format and returns prose, both summaries
    fall back to the raw response so the user still sees something."""
    response = "I could not produce a structured summary."
    backend = _make_backend_with_response(response)

    foreign, german = backend.summarize_bilingual(
        segments=[{"src_lang": "de", "src": "Hi"}],
        foreign_lang="en",
    )
    assert foreign == response
    assert german == response


def test_summarize_bilingual_falls_back_for_unsupported_language() -> None:
    """An unknown foreign_lang should fall back to English and still run."""
    response = "ENGLISH: Hello.\nGERMAN: Hallo."
    backend = _make_backend_with_response(response)

    foreign, german = backend.summarize_bilingual(
        segments=[{"src_lang": "de", "src": "Hi"}],
        foreign_lang="zz",  # invalid
    )
    # Doesn't crash; produces summaries.
    assert foreign and german


def test_summarize_bilingual_accepts_deutsch_label() -> None:
    """The parser should recognize DEUTSCH: as well as GERMAN:."""
    response = "ENGLISH: Hello.\nDEUTSCH: Hallo."
    backend = _make_backend_with_response(response)

    foreign, german = backend.summarize_bilingual(
        segments=[{"src_lang": "de", "src": "Hi"}],
        foreign_lang="en",
    )
    assert foreign == "Hello."
    assert german == "Hallo."


def test_summarize_bilingual_continuation_lines_appended() -> None:
    """Lines following a section header should accumulate into that section."""
    response = "ENGLISH: First line.\nSecond line continues.\nGERMAN: Erste Zeile.\nZweite Zeile."
    backend = _make_backend_with_response(response)

    foreign, german = backend.summarize_bilingual(
        segments=[{"src_lang": "de", "src": "Hi"}],
        foreign_lang="en",
    )
    assert "First line." in foreign and "Second line continues." in foreign
    assert "Erste Zeile." in german and "Zweite Zeile." in german


def test_summarize_bilingual_builds_bilingual_conversation_text() -> None:
    """The conversation block fed to the LLM should label each speaker by language."""
    response = "ENGLISH: x.\nGERMAN: y."
    backend = _make_backend_with_response(response)

    backend.summarize_bilingual(
        segments=[
            {"src_lang": "de", "src": "Wie heißen Sie?"},
            {"src_lang": "en", "src": "My name is Anna."},
        ],
        foreign_lang="en",
    )

    sent_prompt = backend._captured["messages"][0]["content"]  # type: ignore[attr-defined]
    assert "German speaker" in sent_prompt
    assert "Foreign speaker" in sent_prompt
    assert "Wie heißen Sie?" in sent_prompt
    assert "My name is Anna." in sent_prompt
