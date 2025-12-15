"""Gradio UI for the manga translator (local)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gradio as gr

from core.pipeline import MangaPipeline
from config.settings import MODEL_PATH, OLLAMA_HOST


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _natural_key(path: str) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _safe_slug(value: str, *, max_len: int = 80) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    if not value:
        value = "job"
    return value[:max_len]


def _sha256_file(path: Path, *, progress=None) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress is not None and size > 0:
                progress(min(0.05, 0.05 * (done / size)), desc="Hashing input…")
    return h.hexdigest()


def _job_dir_for(
    input_path: Path,
    *,
    source_language: str,
    target_language: str,
    model_path: str,
    progress=None,
) -> Path:
    base_dir = Path(__file__).resolve().parent / "temp_ui" / "jobs"
    base_dir.mkdir(parents=True, exist_ok=True)
    sha = _sha256_file(input_path, progress=progress)
    slug = _safe_slug(input_path.stem)
    src = _safe_slug(source_language, max_len=16)
    tgt = _safe_slug(target_language, max_len=16)
    model_slug = _safe_slug(Path(model_path).stem, max_len=24)
    return base_dir / f"{slug}_{sha[:10]}_{src}_to_{tgt}_{model_slug}"


def _discover_models() -> list[str]:
    base = Path(__file__).resolve().parent
    models: set[str] = set()

    model_dir = base / "models"
    if model_dir.exists():
        models |= {str(p.resolve()) for p in model_dir.rglob("*.gguf")}

    # Also detect Ollama's internal GGUF blobs (if present). Many Ollama models are stored as a
    # GGUF file under a sha256-* filename without an extension.
    ollama_blobs = Path.home() / ".ollama" / "models" / "blobs"
    if ollama_blobs.exists():
        for blob in ollama_blobs.glob("sha256-*"):
            try:
                if not blob.is_file():
                    continue
                if blob.stat().st_size < 256 * 1024 * 1024:
                    continue
                with blob.open("rb") as f:
                    if f.read(4) != b"GGUF":
                        continue
                models.add(str(blob.resolve()))
            except Exception:
                continue

    return sorted(models, key=str.lower)


def _resolve_model_path(model_choice: str, model_custom: str) -> str:
    base = Path(__file__).resolve().parent

    def _resolve(value: str) -> Path:
        p = Path(value).expanduser()
        if p.is_absolute():
            return p
        return (base / p).resolve()

    model_custom = (model_custom or "").strip()
    if model_custom:
        return str(_resolve(model_custom))
    model_choice = (model_choice or "").strip()
    if model_choice:
        return str(_resolve(model_choice))
    return str(_resolve(MODEL_PATH))


def _load_checkpoint(job_dir: Path) -> dict:
    path = job_dir / "checkpoint.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_checkpoint(job_dir: Path, data: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "checkpoint.json"
    tmp = job_dir / "checkpoint.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _expected_translated_path(translated_root: Path, rel_original: Path) -> Path:
    return (translated_root / rel_original).with_suffix(".jpg")


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


_PIPELINES: dict[str, MangaPipeline] = {}


def _get_pipeline(
    *,
    backend: str,
    model_path: str,
    ollama_host: str,
    ollama_model: str,
) -> MangaPipeline:
    key = f"{backend}|{Path(model_path).resolve()}|{ollama_host}|{ollama_model}"
    pipeline = _PIPELINES.get(key)
    if pipeline is None:
        pipeline = MangaPipeline(
            model_path=model_path,
            translation_backend=backend,
            ollama_host=ollama_host,
            ollama_model=ollama_model or None,
        )
        _PIPELINES[key] = pipeline
    return pipeline


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
    progress=None,
) -> SessionResult:
    orig_dir = session_dir / "original"
    out_dir = session_dir / "translated"
    orig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_input = orig_dir / input_path.name
    shutil.copy2(input_path, copied_input)

    if progress is not None:
        progress(0, desc="Processing image…")

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

    if progress is not None:
        progress(1, desc="Done")

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
    model_path: str,
    progress=None,
    resume: bool = True,
) -> SessionResult:
    orig_dir = session_dir / "original"
    out_dir = session_dir / "translated"
    orig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(_iter_images(orig_dir))
    if not existing or not resume:
        if not resume:
            shutil.rmtree(orig_dir, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            try:
                (session_dir / "checkpoint.json").unlink(missing_ok=True)
            except Exception:
                pass
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

    checkpoint = _load_checkpoint(session_dir)
    checkpoint.setdefault("input_name", input_path.name)
    checkpoint.setdefault("source_language", source_language)
    checkpoint.setdefault("target_language", target_language)
    checkpoint.setdefault("model_path", str(Path(model_path).resolve()))
    checkpoint["total_pages"] = len(original_paths)
    checkpoint.setdefault("done", {})
    _save_checkpoint(session_dir, checkpoint)

    total = len(original_paths)
    already_done = 0
    for page in original_paths:
        rel = page.relative_to(orig_dir)
        expected = _expected_translated_path(out_dir, rel)
        if expected.exists():
            already_done += 1
            checkpoint["done"][rel.as_posix()] = expected.relative_to(out_dir).as_posix()
    checkpoint["done_pages"] = already_done
    _save_checkpoint(session_dir, checkpoint)

    if progress is not None:
        progress(already_done / total, desc=f"Translating {already_done}/{total}…")

    translated_paths: list[Path] = []
    durations: list[float] = []
    for idx, page in enumerate(original_paths, start=1):
        start_page = time.perf_counter()

        rel = page.relative_to(orig_dir)
        expected_out = _expected_translated_path(out_dir, rel)
        if resume and expected_out.exists():
            translated_paths.append(expected_out.resolve())
            continue

        out_hint = expected_out
        out_hint.parent.mkdir(parents=True, exist_ok=True)
        translated_path = pipeline.process_image(
            str(page),
            output_path=str(out_hint),
            source_language=source_language,
            target_language=target_language,
        )
        if translated_path:
            translated_paths.append(Path(translated_path).resolve())
            checkpoint["done"][rel.as_posix()] = Path(translated_path).resolve().relative_to(out_dir).as_posix()
            checkpoint["done_pages"] = len(checkpoint["done"])
            _save_checkpoint(session_dir, checkpoint)

        dt = time.perf_counter() - start_page
        durations.append(dt)
        avg = sum(durations) / len(durations)
        done_now = len(checkpoint.get("done", {}))
        remaining = avg * max(0, total - done_now)
        if progress is not None:
            progress(
                min(1.0, done_now / total),
                desc=f"Translating {done_now}/{total} • ETA {_format_eta(remaining)}",
            )

    if not translated_paths:
        raise RuntimeError("No pages were translated.")

    translated_paths = [
        _expected_translated_path(out_dir, page.relative_to(orig_dir)).resolve()
        for page in original_paths
        if _expected_translated_path(out_dir, page.relative_to(orig_dir)).exists()
    ]

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
    backend: str,
    model_choice: str,
    model_custom: str,
    ollama_host: str,
    ollama_model: str,
    resume: bool,
    progress=gr.Progress(),
) -> tuple[SessionResult, gr.Slider, Optional[str], Optional[str], Optional[str], str]:
    if file_obj is None:
        raise gr.Error("Please drop an image or a ZIP file.")

    input_path = Path(getattr(file_obj, "name", "")).resolve()
    if not input_path.exists():
        raise gr.Error("Uploaded file path does not exist.")

    backend_key = (backend or "").strip().lower()
    if backend_key == "ollama":
        model_path = "ollama"
        ollama_host = (ollama_host or OLLAMA_HOST).strip()
        ollama_model = (ollama_model or "").strip()
        if not ollama_model:
            raise gr.Error("Please set an Ollama model name (e.g. qwen2.5:14b-instruct).")
    else:
        model_path = _resolve_model_path(model_choice, model_custom)
        if not Path(model_path).exists():
            raise gr.Error(f"Model not found: {model_path}")
        ollama_host = ""
        ollama_model = ""

    pipeline = _get_pipeline(
        backend=backend_key or "llama_cpp",
        model_path=model_path,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )
    session_dir = _job_dir_for(
        input_path,
        source_language=source_language,
        target_language=target_language,
        model_path=model_path,
        progress=progress,
    )
    logs: list[str] = [
        f"Job: {session_dir}",
        f"Languages: {source_language} -> {target_language}",
        f"Backend: {backend_key or 'llama_cpp'}",
        f"Model: {ollama_model if backend_key == 'ollama' else Path(model_path).name}",
        f"Resume: {resume}",
    ]

    if resume:
        checkpoint = _load_checkpoint(session_dir)
        cp_model = str(checkpoint.get("model_path") or "").strip()
        if cp_model and Path(cp_model).resolve() != Path(model_path).resolve():
            raise gr.Error(
                "This job was created with a different model. "
                "Disable Resume or cleanup the job."
            )

    try:
        if input_path.suffix.lower() == ".zip":
            result = _translate_zip(
                pipeline,
                input_path,
                session_dir,
                source_language=source_language,
                target_language=target_language,
                model_path=model_path,
                progress=progress,
                resume=resume,
            )
            logs.append(f"Translated pages: {len(result.translated_paths)}")
        elif input_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            result = _translate_single_image(
                pipeline,
                input_path,
                session_dir,
                source_language=source_language,
                target_language=target_language,
                progress=progress,
            )
            logs.append("Translated: 1 image")
        else:
            raise gr.Error("Unsupported file type. Use an image or a .zip.")
    except Exception as exc:
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
        gr.Markdown("Translation shows a progress bar with ETA.")
        gr.Markdown(
            "For non-Japanese source OCR (EN/DE/FR/ES/IT/PT/RU/KO/ZH), install: "
            "`pip install -r requirements-ocr-extra.txt`"
        )

        state = gr.State(None)

        with gr.Row():
            upload = gr.File(label="Input (Image or ZIP)")
            translate_btn = gr.Button("Translate", variant="primary")
            pause_btn = gr.Button("Pause", variant="stop")
            resume_toggle = gr.Checkbox(label="Resume (continue an existing job)", value=True)
            cleanup_btn = gr.Button("Cleanup Job")

        available_models = _discover_models()
        default_model_path = _resolve_model_path("", "")
        model_choices = available_models if available_models else [default_model_path]
        default_choice = default_model_path if default_model_path in model_choices else model_choices[0]

        with gr.Row():
            backend = gr.Dropdown(
                label="Translation backend",
                choices=["llama_cpp", "ollama"],
                value="llama_cpp",
                interactive=True,
            )
            model_choice = gr.Dropdown(
                label="Model (GGUF)",
                choices=model_choices,
                value=default_choice,
                interactive=True,
            )
            model_custom = gr.Textbox(
                label="Custom model path (overrides dropdown)",
                placeholder="Leave empty to use the dropdown model",
            )

        with gr.Row():
            ollama_host = gr.Textbox(label="Ollama host", value=OLLAMA_HOST)
            ollama_model = gr.Textbox(label="Ollama model", placeholder="e.g. qwen2.5:14b-instruct")

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

        translate_evt = translate_btn.click(
            fn=translate,
            inputs=[upload, source_lang, target_lang, backend, model_choice, model_custom, ollama_host, ollama_model, resume_toggle],
            outputs=[state, page, orig_img, trans_img, zip_out, logs],
        )
        pause_btn.click(fn=None, cancels=[translate_evt])
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
