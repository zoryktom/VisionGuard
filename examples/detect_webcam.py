#!/usr/bin/env python3
"""Open the default webcam and overlay VisionGuard detections."""

from visionguard.pipeline import RealTimePipeline


def main() -> None:
    pipeline = RealTimePipeline.from_files(
        config_path="configs/inference.yaml",
        zones_path="configs/zones.example.yaml",
        backend="dummy",  # switch to "ultralytics" when weights are available
    )
    pipeline.run_file(source="0", display=True)


if __name__ == "__main__":
    main()
