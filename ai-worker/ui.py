"""Gradio UI for the manga translator (local)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gradio as gr

from core.pipeline import MangaPipeline


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _natural_key(path: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def _iter_images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            yield path


@dataclass
class SessionResult:
    session_dir: Path
    original_paths: list[Path]
    translated_paths: list[Path]
    translated_zip: Optional[Path] = None


_PIPELINE: Optional[MangaPipeline] = None


def _get_pipeline() -> MangaPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = MangaPipeline()
    return _PIPELINE


def _make_session_dir() -> Path:
    base_dir = Path(__file__).resolve().parent / "temp_ui"
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="session_", dir=str(base_dir)))


def _translate_single_image(
    pipeline: MangaPipeline,
    input_path: Path,
    session_dir: Path,
    *,
    source_language: str,
    target_language: str,
) -> SessionResult:
    orig_dir = session_dir / "original"
    out_dir = session_dir / "translated"
    orig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_input = orig_dir / input_path.name
    shutil.copy2(input_path, copied_input)

    save_basename = copied_input.stem
    save_path_hint = out_dir / f"{save_basename}.jpg"
    translated_path = pipeline.process_image(
        str(copied_input),
        output_path=str(save_path_hint),
        source_language=source_language,
        target_language=target_language,
    )
    if not translated_path:
        raise RuntimeError("Image processing failed.")

    return SessionResult(
        session_dir=session_dir,
        original_paths=[copied_input],
        translated_paths=[Path(translated_path).resolve()],
    )


def _translate_zip(
    pipeline: MangaPipeline,
    input_path: Path,
    session_dir: Path,
    *,
    source_language: str,
    target_language: str,
) -> SessionResult:
    orig_dir = session_dir / "original"
    out_dir = session_dir / "translated"
    orig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_path, "r") as zip_ref:
        zip_ref.extractall(orig_dir)

    original_paths = sorted(
        _iter_images(orig_dir),
        key=lambda p: _natural_key(str(p.relative_to(orig_dir)).replace("\\", "/")),
    )
    if not original_paths:
        raise RuntimeError("No images found in the ZIP.")

    translated_paths: list[Path] = []
    for page in original_paths:
        rel = page.relative_to(orig_dir)
        out_hint = out_dir / rel
        out_hint.parent.mkdir(parents=True, exist_ok=True)
        translated_path = pipeline.process_image(
            str(page),
            output_path=str(out_hint),
            source_language=source_language,
            target_language=target_language,
        )
        if translated_path:
            translated_paths.append(Path(translated_path).resolve())

    if not translated_paths:
        raise RuntimeError("No pages were translated.")

    zip_out = session_dir / f"{input_path.stem}_translated.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for translated in translated_paths:
            arcname = translated.relative_to(out_dir).as_posix()
            zf.write(translated, arcname=arcname)

    return SessionResult(
        session_dir=session_dir,
        original_paths=original_paths,
        translated_paths=translated_paths,
        translated_zip=zip_out,
    )


def translate(
    file_obj,
    source_language: str,
    target_language: str,
) -> tuple[SessionResult, gr.Slider, Optional[str], Optional[str], Optional[str], str]:
    if file_obj is None:
        raise gr.Error("Please drop an image or a ZIP file.")

    input_path = Path(getattr(file_obj, "name", "")).resolve()
    if not input_path.exists():
        raise gr.Error("Uploaded file path does not exist.")

    pipeline = _get_pipeline()
    session_dir = _make_session_dir()
    logs: list[str] = [f"Session: {session_dir}", f"Languages: {source_language} -> {target_language}"]

    try:
        if input_path.suffix.lower() == ".zip":
            result = _translate_zip(
                pipeline,
                input_path,
                session_dir,
                source_language=source_language,
                target_language=target_language,
            )
            logs.append(f"Translated pages: {len(result.translated_paths)}")
        elif input_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            result = _translate_single_image(
                pipeline,
                input_path,
                session_dir,
                source_language=source_language,
                target_language=target_language,
            )
            logs.append("Translated: 1 image")
        else:
            raise gr.Error("Unsupported file type. Use an image or a .zip.")
    except Exception as exc:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise gr.Error(str(exc)) from exc

    page_count = len(result.translated_paths)
    slider = gr.Slider(
        minimum=1,
        maximum=page_count,
        value=1,
        step=1,
        interactive=page_count > 1,
        label=f"Page (1/{page_count})",
    )

    orig0 = str(result.original_paths[0])
    trans0 = str(result.translated_paths[0])
    zip_out = str(result.translated_zip) if result.translated_zip else None
    return result, slider, orig0, trans0, zip_out, "\n".join(logs)


def show_page(result: SessionResult, page: int) -> tuple[Optional[str], Optional[str], gr.Slider]:
    if result is None:
        return None, None, gr.Slider(minimum=1, maximum=1, value=1, step=1, interactive=False)

    index = max(1, int(page)) - 1
    index = min(index, len(result.translated_paths) - 1)

    page_count = len(result.translated_paths)
    slider = gr.Slider(
        minimum=1,
        maximum=page_count,
        value=index + 1,
        step=1,
        interactive=page_count > 1,
        label=f"Page ({index + 1}/{page_count})",
    )
    return str(result.original_paths[index]), str(result.translated_paths[index]), slider


def cleanup(result: SessionResult) -> tuple[None, None, None, gr.Slider, str]:
    if result and result.session_dir.exists():
        shutil.rmtree(result.session_dir, ignore_errors=True)
    slider = gr.Slider(minimum=1, maximum=1, value=1, step=1, interactive=False, label="Page")
    return None, None, None, slider, "Cleaned up session files."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AutoScanlate AI - Local UI") as demo:
        gr.Markdown("# AutoScanlate AI - Local UI")
        gr.Markdown(f"UI file: `{Path(__file__).resolve()}`")
        gr.Markdown("Drop an image or a ZIP. Left = original, right = translated.")

        state = gr.State(None)

        with gr.Row():
            upload = gr.File(label="Input (Image or ZIP)")
            translate_btn = gr.Button("Translate", variant="primary")
            cleanup_btn = gr.Button("Cleanup Session")

        with gr.Row():
            source_lang = gr.Dropdown(
                label="Source language",
                choices=["auto", "Japanese", "English", "German", "French", "Spanish", "Italian", "Portuguese", "Russian", "Chinese", "Korean"],
                value="Japanese",
            )
            target_lang = gr.Dropdown(
                label="Target language",
                choices=["English", "German", "French", "Spanish", "Italian", "Portuguese"],
                value="German",
            )

        page = gr.Slider(minimum=1, maximum=1, value=1, step=1, interactive=False, label="Page")

        with gr.Row():
            orig_img = gr.Image(label="Original", interactive=False)
            trans_img = gr.Image(label="Translated", interactive=False)

        zip_out = gr.File(label="Translated ZIP (if input was ZIP)")
        logs = gr.Textbox(label="Logs", lines=6, interactive=False)

        translate_btn.click(
            fn=translate,
            inputs=[upload, source_lang, target_lang],
            outputs=[state, page, orig_img, trans_img, zip_out, logs],
        )
        page.change(fn=show_page, inputs=[state, page], outputs=[orig_img, trans_img, page])
        cleanup_btn.click(fn=cleanup, inputs=[state], outputs=[state, orig_img, trans_img, page, logs])

    return demo


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")
    parser = argparse.ArgumentParser(description="AutoScanlate AI - local Gradio UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    build_app().launch(server_name=args.host, server_port=args.port, share=args.share)
