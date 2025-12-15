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

    def extract_boxes_and_text(
        self,
        image: Image.Image,
        *,
        source_language: str,
        min_confidence: float = 0.10,
    ) -> list[tuple[list[int], str]]:
        """
        Return text boxes + text for page-level OCR (best for non-Japanese sources).

        Output boxes are [x1, y1, x2, y2] in image coordinates.
        """
        lang = (source_language or "English").strip().lower()
        if lang in {"ja", "jp", "japanese"}:
            return []

        if lang in {"zh", "chinese", "cn", "zh-cn", "zh-hans"}:
            lang_code = "ch_sim"
        elif lang in {"zh-hant", "zh-tw", "zh-hk", "traditional chinese"}:
            lang_code = "ch_tra"
        elif lang in {"ko", "korean"}:
            lang_code = "ko"
        elif lang in {"de", "german", "deutsch"}:
            lang_code = "de"
        elif lang in {"fr", "french"}:
            lang_code = "fr"
        elif lang in {"es", "spanish"}:
            lang_code = "es"
        elif lang in {"it", "italian"}:
            lang_code = "it"
        elif lang in {"pt", "portuguese"}:
            lang_code = "pt"
        elif lang in {"ru", "russian"}:
            lang_code = "ru"
        else:
            lang_code = "en"

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
            primary = dict(
                detail=1,
                paragraph=True,
                canvas_size=4096,
                mag_ratio=1.5,
                text_threshold=0.6,
                low_text=0.3,
                link_threshold=0.4,
                add_margin=0.1,
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )
            fallback = dict(
                detail=1,
                paragraph=True,
                canvas_size=5120,
                mag_ratio=2.0,
                text_threshold=0.5,
                low_text=0.2,
                link_threshold=0.4,
                add_margin=0.15,
                contrast_ths=0.05,
                adjust_contrast=0.7,
            )

            results = list(reader.readtext(bgr, **primary) or [])
            if not results:
                results = list(reader.readtext(bgr, **fallback) or [])
            else:
                results.extend(list(reader.readtext(bgr, **fallback) or []))
        except Exception:
            return []

        def _iou(a: list[int], b: list[int]) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            return float(inter) / float(a_area + b_area - inter)

        out: list[tuple[list[int], str, float]] = []
        for item in results or []:
            try:
                bbox, text, conf = item
            except Exception:
                continue
            if not text or not str(text).strip():
                continue
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.0
            if conf_f < float(min_confidence):
                continue
            try:
                xs = [int(round(float(p[0]))) for p in bbox]
                ys = [int(round(float(p[1]))) for p in bbox]
                x1, x2 = max(0, min(xs)), max(0, max(xs))
                y1, y2 = max(0, min(ys)), max(0, max(ys))
                if x2 <= x1 or y2 <= y1:
                    continue
            except Exception:
                continue
            box = [x1, y1, x2, y2]
            text_s = str(text).strip()

            replaced = False
            for i, (prev_box, prev_text, prev_conf) in enumerate(out):
                if _iou(prev_box, box) >= 0.7:
                    if conf_f > prev_conf and len(text_s) >= len(prev_text):
                        out[i] = (box, text_s, conf_f)
                    replaced = True
                    break
            if not replaced:
                out.append((box, text_s, conf_f))

        return [(b, t) for (b, t, _) in out]

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
