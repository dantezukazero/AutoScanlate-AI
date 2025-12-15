"""OCR service with optional multi-language backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass(frozen=True)
class OcrConfig:
    prefer_japanese_manga_ocr: bool = True


class OCRService:
    def __init__(self, config: Optional[OcrConfig] = None):
        self.config = config or OcrConfig()
        self._manga_ocr = None
        self._easyocr_readers: dict[str, object] = {}

    def extract_text(self, image: Image.Image, *, source_language: str) -> str:
        lang = (source_language or "Japanese").strip().lower()

        if lang in {"ja", "jp", "japanese"}:
            return self._extract_manga_ocr(image)
        if lang in {"ko", "korean"}:
            return self._extract_easyocr(image, "ko")
        if lang in {"zh", "chinese", "cn", "zh-cn", "zh-hans"}:
            return self._extract_easyocr(image, "ch_sim")
        if lang in {"zh-hant", "zh-tw", "zh-hk", "traditional chinese"}:
            return self._extract_easyocr(image, "ch_tra")
        if lang in {"en", "english"}:
            return self._extract_easyocr(image, "en")
        if lang in {"de", "german", "deutsch"}:
            return self._extract_easyocr(image, "de")
        if lang in {"fr", "french"}:
            return self._extract_easyocr(image, "fr")
        if lang in {"es", "spanish"}:
            return self._extract_easyocr(image, "es")
        if lang in {"it", "italian"}:
            return self._extract_easyocr(image, "it")
        if lang in {"pt", "portuguese"}:
            return self._extract_easyocr(image, "pt")
        if lang in {"ru", "russian"}:
            return self._extract_easyocr(image, "ru")
        if lang in {"auto", "detect"}:
            candidates: list[tuple[str, Optional[str]]] = []
            if self.config.prefer_japanese_manga_ocr:
                candidates.append(("manga-ocr", None))
            candidates.extend(
                [
                    ("easyocr", "ko"),
                    ("easyocr", "ch_sim"),
                    ("easyocr", "en"),
                    ("easyocr", "de"),
                ]
            )
            if not self.config.prefer_japanese_manga_ocr:
                candidates.append(("manga-ocr", None))

            for backend, code in candidates:
                if backend == "manga-ocr":
                    text = self._extract_manga_ocr(image)
                else:
                    try:
                        text = self._extract_easyocr(image, str(code))
                    except RuntimeError:
                        text = ""
                if text.strip():
                    return text
            return ""

        return self._extract_easyocr(image, "en")

    def _extract_manga_ocr(self, image: Image.Image) -> str:
        if self._manga_ocr is None:
            from manga_ocr import MangaOcr

            self._manga_ocr = MangaOcr()

        try:
            return str(self._manga_ocr(image))
        except Exception:
            return ""

    def _extract_easyocr(self, image: Image.Image, lang_code: str) -> str:
        reader = self._easyocr_readers.get(lang_code)
        if reader is None:
            try:
                import torch
                import easyocr
            except Exception as exc:
                raise RuntimeError(
                    "EasyOCR is required for this source language. "
                    "Install it with: pip install -r requirements-ocr-extra.txt"
                ) from exc

            reader = easyocr.Reader([lang_code], gpu=torch.cuda.is_available())
            self._easyocr_readers[lang_code] = reader

        try:
            import numpy as np

            rgb = image.convert("RGB")
            bgr = np.asarray(rgb)[:, :, ::-1]
            results = reader.readtext(bgr, detail=0, paragraph=True)
        except Exception:
            return ""

        if isinstance(results, str):
            return results
        if not results:
            return ""
        if isinstance(results, list):
            return " ".join([str(x).strip() for x in results if str(x).strip()]).strip()
        return str(results).strip()
