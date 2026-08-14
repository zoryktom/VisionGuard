#!/usr/bin/env python3
"""Annotate a video file or image folder."""

from __future__ import annotations

import argparse

from visionguard.pipeline import RealTimePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionGuard file inference")
    parser.add_argument("source", help="Video path or image directory")
    parser.add_argument("-o", "--output", default="outputs/annotated.mp4")
    parser.add_argument("-b", "--backend", default="dummy")
    args = parser.parse_args()
    pipeline = RealTimePipeline.from_files(
        config_path="configs/inference.yaml",
        zones_path="configs/zones.example.yaml",
        backend=args.backend,
    )
    pipeline.run_file(source=args.source, output=args.output, display=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
