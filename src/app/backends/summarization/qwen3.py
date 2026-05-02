"""Qwen3-4B summarization backend using llama-cpp-python.

Generates dual-language summaries (foreign + German) of bilingual conversations.
Uses structured prompts with LANGUAGE: [summary] format.

Qwen3 uses <think>...</think> blocks for chain-of-thought reasoning which are
stripped from the final output.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

from app.backends.base import SummarizationBackend

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)

SUMM_MODEL_REPO = os.getenv("SUMM_MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
SUMM_MODEL_FILE = os.getenv("SUMM_MODEL_FILE", "Qwen3-4B-Q4_K_M.gguf")
SUMM_N_GPU_LAYERS = int(os.getenv("SUMM_N_GPU_LAYERS", "-1"))
SUMM_N_CTX = int(os.getenv("SUMM_N_CTX", "4096"))

# Regex to strip Qwen3 <think>...</think> blocks from responses
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_INCOMPLETE_THINK_PATTERN = re.compile(r"<think>.*$", re.DOTALL)


def _strip_think_block(text: str) -> str:
    """Strip Qwen3 thinking blocks from model output, including truncated ones."""
    original = text
    text = _THINK_PATTERN.sub("", text)
    text = _INCOMPLETE_THINK_PATTERN.sub("", text)
    result = text.strip()
    if not result:
        before_think = original.split("<think>")[0].strip()
        if before_think:
            return before_think
        return "(Summary generation incomplete)"
    return result


class Qwen3SummarizationBackend(SummarizationBackend):
    """Qwen3-4B summarization backend via llama-cpp-python."""

    def __init__(self) -> None:
        self._llm: Llama | None = None

    def load_model(self) -> None:
        self._get_llm()

    def _get_llm(self) -> Llama:
        if self._llm is not None:
            return self._llm
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama as _Llama

        logger.info("Downloading summarization model: %s/%s", SUMM_MODEL_REPO, SUMM_MODEL_FILE)
        model_path = hf_hub_download(  # nosec B615
            repo_id=SUMM_MODEL_REPO,
            filename=SUMM_MODEL_FILE,
        )
        logger.info("Loading summarization model from: %s", model_path)
        self._llm = _Llama(
            model_path=model_path,
            n_gpu_layers=SUMM_N_GPU_LAYERS,
            n_ctx=SUMM_N_CTX,
            n_batch=512,
            verbose=False,
            use_mmap=False,
        )
        logger.info("Summarization model loaded")
        return self._llm

    def warmup(self) -> None:
        llm = self._get_llm()
        # Quick warmup: generate a short completion
        llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
        )
        logger.info("  Summarization warmup complete")

    @staticmethod
    def _build_summary_prompt(segments: list[dict], foreign_name: str) -> str:
        lines = []
        for seg in segments:
            speaker = "German speaker" if seg["src_lang"] == "de" else "Foreign speaker"
            lang_label = "German" if seg["src_lang"] == "de" else foreign_name
            lines.append(f"{speaker} ({lang_label}): {seg['src']}")
        conversation_text = "\n".join(lines)
        return (
            f"Summarize this bilingual dialogue. Generate TWO summaries:\n\n"
            f"1. A summary in {foreign_name} (2-3 sentences)\n"
            f"2. The same summary translated to German (2-3 sentences)\n\n"
            f"Both summaries must cover what BOTH speakers said.\n\n"
            f"Conversation:\n{conversation_text}\n\n"
            f"Respond in this exact format:\n"
            f"{foreign_name.upper()}: [summary in {foreign_name}]\n"
            f"GERMAN: [same summary in German]"
        )

    @staticmethod
    def _parse_bilingual_summary(response: str, foreign_name: str) -> tuple[str, str]:
        """Split the LLM response into (foreign, german) summaries."""
        foreign_prefix = foreign_name.upper() + ":"
        foreign_summary = ""
        german_summary = ""
        current_section: str | None = None

        for line in response.split("\n"):
            line_upper = line.upper()
            stripped = line.strip()
            if line_upper.startswith(foreign_prefix):
                current_section = "foreign"
                foreign_summary = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_upper.startswith("GERMAN:") or line_upper.startswith("DEUTSCH:"):
                current_section = "german"
                german_summary = line.split(":", 1)[1].strip() if ":" in line else ""
            elif current_section == "foreign" and stripped:
                foreign_summary += " " + stripped
            elif current_section == "german" and stripped:
                german_summary += " " + stripped

        return foreign_summary, german_summary

    def summarize_bilingual(
        self,
        segments: list[dict],
        foreign_lang: str,
    ) -> tuple[str, str]:
        """Generate both foreign and German summaries in a single LLM call."""
        from app.languages import LANG_NAMES

        if foreign_lang not in LANG_NAMES:
            logger.warning("Unsupported language '%s' for summary, using 'English'", foreign_lang)
            foreign_lang = "en"

        foreign_name = LANG_NAMES.get(foreign_lang, foreign_lang)
        llm = self._get_llm()
        prompt = self._build_summary_prompt(segments, foreign_name)
        output = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.5,
            top_p=0.9,
        )
        response = _strip_think_block(output["choices"][0]["message"]["content"].strip())

        foreign_summary, german_summary = self._parse_bilingual_summary(response, foreign_name)
        # Fallback if parsing failed: return the raw response in both slots
        # rather than dropping content the user might still want to see.
        if not foreign_summary or not german_summary:
            foreign_summary = foreign_summary or response
            german_summary = german_summary or response

        return foreign_summary.strip(), german_summary.strip()
