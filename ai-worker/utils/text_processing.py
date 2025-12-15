"""Text processing utilities."""

from __future__ import annotations

import re
from typing import Optional


def sanitize_for_font(text: str) -> str:
    """
    Sanitize text for Pillow font rendering.

    Keep European letters (incl. German umlauts) while removing characters that
    tend to cause issues in common manga fonts (emoji, CJK leftovers, etc.).
    """
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("\n", " ").replace("\r", " ")

    text = re.sub(r"[\U0001F300-\U0001FAFF]", "", text)  # emojis
    text = re.sub(r"[\u200d\uFE0E\uFE0F]", "", text)  # ZWJ + variation selectors
    text = re.sub(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]", "", text)  # JP/CN/KR scripts

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_translation_output(raw: str, *, target_language: Optional[str] = None) -> str:
    """
    Clean LLM translation output from thinking tags and common prefixes.
    """
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    lang = (target_language or "").strip().lower()
    lang_prefixes = [
        "english", "en",
        "german", "deutsch", "de",
        "french", "fr",
        "spanish", "es",
        "italian", "it",
        "portuguese", "pt",
        "japanese", "ja",
        "chinese", "zh",
        "korean", "ko",
        "russian", "ru",
    ]
    if lang and lang not in lang_prefixes:
        lang_prefixes.append(lang)

    prefixes = (
        r"(?i)^("
        + "|".join(["text", "translation", "output", "response", "answer", "result", "final"] + lang_prefixes)
        + r")\s*[:\-]?\s*"
    )
    clean = re.sub(prefixes, "", clean)

    clean = clean.strip().strip('"').strip("'")
    return sanitize_for_font(clean)

