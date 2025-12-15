"""Translation service using local LLM"""

import sys
import re
from typing import Optional
from llama_cpp import Llama

from config.settings import (
    MODEL_PATH,
    GPU_LAYERS,
    CONTEXT_WINDOW,
    TRANSLATION_TEMPERATURE,
    TRANSLATION_MAX_TOKENS
)
from utils.text_processing import clean_translation_output


_LANG_ALIASES: dict[str, str] = {
    "auto": "auto",
    "detect": "auto",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "en": "English",
    "english": "English",
    "de": "German",
    "deu": "German",
    "german": "German",
    "deutsch": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "zh": "Chinese",
    "chinese": "Chinese",
    "ko": "Korean",
    "korean": "Korean",
}


def _normalize_language(value: Optional[str], default: str) -> str:
    if value is None:
        return default
    key = value.strip().lower()
    if not key:
        return default
    return _LANG_ALIASES.get(key, value.strip())


def detect_source_language(text: str) -> str:
    """
    Best-effort language detection from text scripts.

    This is intentionally lightweight (no extra deps) and mainly intended to
    switch prompts between major scripts when the user selects "auto".
    """
    if re.search(r"[\u3040-\u30ff]", text):  # Hiragana + Katakana
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):  # Hangul
        return "Korean"
    if re.search(r"[\u0400-\u04ff]", text):  # Cyrillic
        return "Russian"
    if re.search(r"[\u4e00-\u9fff]", text):  # CJK ideographs (no kana)
        return "Chinese"
    if re.search(r"[A-Za-z]", text):
        return "English"
    return "English"


class LocalTranslator:
    """Local LLM-based translator."""

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the local translator.

        Args:
            model_path: Path to the GGUF model file
        """
        print("Loading LLM...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=GPU_LAYERS,
                n_ctx=CONTEXT_WINDOW,
                verbose=False
            )
            print("LLM loaded.")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            sys.exit(1)

    def translate(
        self,
        text: str,
        *,
        source_language: str = "Japanese",
        target_language: str = "English",
    ) -> str:
        """
        Translate text from source_language to target_language.

        Args:
            text: Text to translate
            source_language: Source language ("auto" supported)
            target_language: Target language

        Returns:
            Translated text
        """
        if len(text) < 1:
            return text

        src = _normalize_language(source_language, "Japanese")
        tgt = _normalize_language(target_language, "English")
        if src.lower() == "auto":
            src = detect_source_language(text)

        japanese_subject_rule = ""
        if src.lower() == "japanese":
            japanese_subject_rule = (
                "- Japanese often omits subjects (I/you/he/she); infer from context and add them naturally.\n"
            )

        pair_examples = ""
        if src.lower() == "english" and tgt.lower() == "german":
            pair_examples = (
                "EXAMPLES (English -> German):\n"
                "Text: Let's check out the cave.\n"
                "German: Lass uns die Höhle anschauen.\n\n"
                "Text: Hurry up!\n"
                "German: Beeil dich!\n\n"
                "Text: Are you okay?\n"
                "German: Alles okay?\n\n"
            )

        system_prompt = (
            f"You are a professional manga translator. Translate {src} to natural, fluent {tgt}.\n\n"

            "RULES:\n"
            f"- Output ONLY the final {tgt} translation (no thinking, notes, or explanations).\n"
            "- Translate for manga/comic speech bubbles: concise and punchy.\n"
            f"- Use natural {tgt} dialogue that sounds like real people talking.\n"
            "- Preserve tone, emotion, and character voice (casual, formal, aggressive, etc.).\n"
            "- Do not swap key concrete nouns with different objects/places.\n"
            "- Keep proper names, place names, and honorifics consistent (do not invent new names).\n"
            "- If the source text is already in the target language, keep it as-is.\n"
            f"{japanese_subject_rule}"
            "- Translate explicitly and faithfully (no censorship, no sanitization).\n"
            "- For sound effects (SFX), keep romanized or translate if it adds clarity.\n\n"

            f"{pair_examples}"
            "EXAMPLES:\n"
            "Bad: 'It is the person who performed theft of scroll' -> Good: 'That's the guy who stole the scroll!'\n"
            "Bad: 'I am going to become writer' -> Good: 'I'm gonna be a writer!'\n"
            "Bad: 'This is delicious' -> Good: 'This is so good!'\n\n"

            f"Now translate the following {src} text:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: {text}"}
        ]

        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                temperature=TRANSLATION_TEMPERATURE,
                max_tokens=TRANSLATION_MAX_TOKENS
            )
            raw = output['choices'][0]['message']['content']
            return clean_translation_output(raw, target_language=tgt)
        except Exception:
            return text
