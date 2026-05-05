"""
Shared language registry for translation and summarization.

Format: lang_code -> (full_name, iso_code)
- full_name: Human-readable name for summarization prompts
- iso_code: BCP-47 style code for TranslateGemma
"""

LANG_INFO: dict[str, tuple[str, str]] = {
    "en": ("English", "en"),
    "de": ("German", "de"),
    "fr": ("French", "fr"),
    "es": ("Spanish", "es"),
    "it": ("Italian", "it"),
    "pl": ("Polish", "pl"),
    "ro": ("Romanian", "ro"),
    "hr": ("Croatian", "hr"),
    "bg": ("Bulgarian", "bg"),
    "sq": ("Albanian", "sq"),
    "tr": ("Turkish", "tr"),
    "ru": ("Russian", "ru"),
    "uk": ("Ukrainian", "uk"),
    "hu": ("Hungarian", "hu"),
    "ar": ("Arabic", "ar"),
    "fa": ("Farsi", "fa"),
    "ku": ("Kurdish", "ku"),
    "sr": ("Serbian", "sr"),
    "zh": ("Chinese", "zh-Hans"),
    "ja": ("Japanese", "ja"),
    "ko": ("Korean", "ko"),
    "pt": ("Portuguese", "pt"),
    "nl": ("Dutch", "nl"),
}

LANG_NAMES: dict[str, str] = {k: v[0] for k, v in LANG_INFO.items()}


# Tier A: well-supported by both Whisper Large-v3-Turbo (low CER) and
# TranslateGemma 12B (evaluated language pairs).
# Tier B: usable but lower quality on at least one of the two; surfaced in
# the UI with a "(beta)" suffix.
# Languages in LANG_INFO but not listed here (sq, ku) are accepted by the MT
# backend but intentionally omitted from speech-language dropdowns.
LANG_TIER_A: tuple[str, ...] = (
    "ar",
    "zh",
    "nl",
    "en",
    "fr",
    "it",
    "ja",
    "ko",
    "pl",
    "pt",
    "ru",
    "es",
    "tr",
)
LANG_TIER_B: tuple[str, ...] = (
    "bg",
    "hr",
    "fa",
    "hu",
    "ro",
    "sr",
    "uk",
)


def _entry(code: str, tier: str) -> dict[str, str]:
    label = LANG_INFO[code][0]
    if tier == "b":
        label = f"{label} (beta)"
    return {"code": code, "label": label, "tier": tier}


def speech_languages() -> list[dict[str, str]]:
    """Languages offered in the live-translation dropdowns.

    German is exposed as the first entry under the special ``host`` tier
    so a host who only wants a German transcription (no cross-language
    translation) can pick it as their guest language too. Foreign tiers
    follow: tier A (well-supported by both Whisper and TranslateGemma)
    sorted alphabetically by display label, then tier B (beta) likewise.
    """
    a = sorted((_entry(c, "a") for c in LANG_TIER_A), key=lambda e: e["label"])
    b = sorted((_entry(c, "b") for c in LANG_TIER_B), key=lambda e: e["label"])
    return [{"code": "de", "label": "Deutsch", "tier": "host"}, *a, *b]


def translation_languages() -> list[dict[str, str]]:
    """Languages for the text-to-text translate page.

    Currently identical to :func:`speech_languages` — German leads either
    way. Kept as a separate entry point so future divergence (e.g. enabling
    text-only languages that aren't in the speech registry) is a one-line
    change.
    """
    return speech_languages()
