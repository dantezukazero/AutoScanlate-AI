"""Main pipeline for manga translation processing"""

import os
import sys
import shutil
import zipfile
import torch
from typing import Optional
from PIL import Image
from ultralytics import YOLO

from config.settings import (
    YOLO_MODEL_NAME,
    YOLO_CONFIDENCE_THRESHOLD,
    OUTPUT_QUALITY,
    TEMP_DIR,
    MODEL_PATH,
    FONT_PATH,
    OCR_CROP_PAD_PCT,
    OCR_UPSCALE_FACTOR,
    OLLAMA_HOST,
)
from services.ocr import OCRService
from services.translation import create_translator
from services.typesetting import Typesetter
from utils.box_processing import consolidate_boxes


class MangaPipeline:
    """Main pipeline for processing manga images and ZIP files."""

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        translation_backend: str = "llama_cpp",
        ollama_host: str = OLLAMA_HOST,
        ollama_model: Optional[str] = None,
    ):
        """Initialize the manga translation pipeline."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device.upper()}")

        if not os.path.exists(YOLO_MODEL_NAME):
            print(f"Missing YOLO model: {YOLO_MODEL_NAME}")
            sys.exit(1)

        self.detector = YOLO(YOLO_MODEL_NAME)
        self.ocr = OCRService()
        self.translator = create_translator(
            model_path=model_path or MODEL_PATH,
            backend=translation_backend,
            ollama_host=(ollama_host or OLLAMA_HOST),
            ollama_model=ollama_model,
        )
        self.typesetter = Typesetter(FONT_PATH)
        print("Pipeline Ready (V10 - Stable | Masked Inpainting).")

    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        *,
        source_language: str = "Japanese",
        target_language: str = "English",
    ) -> Optional[str]:
        """
        Process a single manga image.

        Args:
            image_path: Path to input image
            output_path: Optional path for output image

        Returns:
            Path to saved output image, or None if processing failed
        """
        try:
            with Image.open(image_path) as img_src:
                img_src.load()
                original_img = img_src.convert("RGB")
        except Exception as e:
            print(f"Skipped invalid image: {image_path} ({e})")
            return None

        print(f"   Processing: {os.path.basename(image_path)}")

        src_lang_key = (source_language or "Japanese").strip().lower()
        boxes_text: list[tuple[list[int], str]] = []
        if src_lang_key not in {"ja", "jp", "japanese"}:
            try:
                boxes_text = self.ocr.extract_boxes_and_text(original_img, source_language=source_language)
            except Exception:
                boxes_text = []

        if boxes_text:
            for box, src_text in boxes_text:
                if not src_text.strip():
                    continue
                fr_text = self.translator.translate(
                    src_text,
                    source_language=source_language,
                    target_language=target_language,
                )
                self.typesetter.clean_box(original_img, box)
                self.typesetter.draw_text(original_img, fr_text, box)
        else:
            # Detect text boxes (YOLO bubble detector)
            results = self.detector(original_img, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            boxes = []
            for r in results:
                if r.boxes:
                    for box in r.boxes.xyxy.cpu().numpy():
                        boxes.append(list(map(int, box)))

            boxes = consolidate_boxes(boxes)

            # Process each text box
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    w = max(1, x2 - x1)
                    h = max(1, y2 - y1)
                    pad_x = int(round(w * float(OCR_CROP_PAD_PCT)))
                    pad_y = int(round(h * float(OCR_CROP_PAD_PCT)))

                    x1p = max(0, x1 - pad_x)
                    y1p = max(0, y1 - pad_y)
                    x2p = min(original_img.width, x2 + pad_x)
                    y2p = min(original_img.height, y2 + pad_y)

                    crop = original_img.crop((x1p, y1p, x2p, y2p))
                    if int(OCR_UPSCALE_FACTOR) > 1:
                        crop = crop.resize(
                            (crop.width * int(OCR_UPSCALE_FACTOR), crop.height * int(OCR_UPSCALE_FACTOR)),
                            resample=Image.Resampling.LANCZOS,
                        )

                    src_text = self.ocr.extract_text(crop, source_language=source_language)

                    if not src_text.strip():
                        continue

                    fr_text = self.translator.translate(
                        src_text,
                        source_language=source_language,
                        target_language=target_language,
                    )
                    self.typesetter.clean_box(original_img, box)
                    self.typesetter.draw_text(original_img, fr_text, box)

        # Save output
        if output_path:
            base, _ = os.path.splitext(output_path)
            save_path = base + ".jpg"
        else:
            base = os.path.splitext(os.path.basename(image_path))[0]
            save_path = f"translated_{base}.jpg"

        original_img.save(save_path, "JPEG", quality=OUTPUT_QUALITY)
        return save_path

    def process_zip(self, zip_path: str, *, source_language: str = "Japanese", target_language: str = "English") -> None:
        """
        Process a ZIP file containing manga images.

        Args:
            zip_path: Path to input ZIP file
        """
        print(f"\nZIP Detected: {zip_path}")
        temp_dir = TEMP_DIR

        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        os.makedirs(temp_dir)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process all images
        count = 0
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    full_input_path = os.path.join(root, file)
                    new_jpg_path = self.process_image(
                        full_input_path,
                        output_path=full_input_path,
                        source_language=source_language,
                        target_language=target_language,
                    )

                    if new_jpg_path and os.path.normpath(new_jpg_path) != os.path.normpath(full_input_path):
                        try:
                            os.remove(full_input_path)
                        except Exception:
                            pass
                    count += 1

        print(f" -> Processed {count} images.")

        # Create output ZIP
        output_zip = os.path.splitext(zip_path)[0] + "_translated"
        shutil.make_archive(output_zip, 'zip', temp_dir)

        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

        print(f"Created: {output_zip}.zip")

    def run(self, input_path: str, *, source_language: str = "Japanese", target_language: str = "English") -> None:
        """
        Run the pipeline on an input file (image or ZIP).

        Args:
            input_path: Path to input file
        """
        if input_path.lower().endswith('.zip'):
            self.process_zip(input_path, source_language=source_language, target_language=target_language)
        else:
            self.process_image(input_path, source_language=source_language, target_language=target_language)
