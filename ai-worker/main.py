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
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print("File not found.")
        return

    pipeline = MangaPipeline()
    pipeline.run(args.input, source_language=args.source, target_language=args.target)


if __name__ == "__main__":
    main()
