"""Manga Translator - Main Entry Point"""

import os
import argparse

from core.pipeline import MangaPipeline


def main():
    """Main entry point for the manga translator."""
    parser = argparse.ArgumentParser(
        description="Translate manga images (image or ZIP) between languages"
    )
    parser.add_argument(
        "input",
        help="Path to image or ZIP file to process"
    )
    parser.add_argument(
        "--source",
        default="Japanese",
        help="Source language (e.g. Japanese, English, Korean, Chinese, auto). Default: Japanese",
    )
    parser.add_argument(
        "--target",
        default="English",
        help="Target language (e.g. English, German). Default: English",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional GGUF model path for translation (overrides config MODEL_PATH).",
    )
    parser.add_argument(
        "--backend",
        default="llama_cpp",
        choices=["llama_cpp", "ollama"],
        help="Translation backend: llama_cpp (GGUF) or ollama (HTTP). Default: llama_cpp",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama host URL (default from config OLLAMA_HOST).",
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Ollama model name (required if --backend ollama). Example: qwen2.5:14b-instruct",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File not found.")
        return

    pipeline = MangaPipeline(
        model_path=args.model,
        translation_backend=args.backend,
        ollama_host=args.ollama_host,
        ollama_model=args.ollama_model,
    )
    pipeline.run(args.input, source_language=args.source, target_language=args.target)


if __name__ == "__main__":
    main()
