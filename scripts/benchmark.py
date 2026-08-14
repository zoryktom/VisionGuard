"""Micro-benchmark dummy vs (optional) YOLO backends."""

from __future__ import annotations

import statistics
import time

import typer
from rich.console import Console
from rich.table import Table

from visionguard.capture import synthetic_frame
from visionguard.config import load_config
from visionguard.hazards.taxonomy import default_taxonomy
from visionguard.inference import build_detector
from visionguard.utils.device import describe_device, resolve_device

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    backend: str = typer.Option("dummy"),
    frames: int = typer.Option(30),
    warmup: int = typer.Option(3),
) -> None:
    """Print mean / p95 inference latency."""

    config = load_config(overrides={"inference": {"backend": backend}})
    device = resolve_device(config.system.device)
    det = build_detector(config.inference, default_taxonomy(), device)
    frame = synthetic_frame()
    for _ in range(warmup):
        det.predict(frame)
    samples: list[float] = []
    for i in range(frames):
        img = synthetic_frame(seed=i)
        t0 = time.perf_counter()
        det.predict(img)
        samples.append((time.perf_counter() - t0) * 1000.0)
    table = Table(title=f"VisionGuard benchmark ({backend} / {describe_device(device)})")
    table.add_column("Metric")
    table.add_column("ms", justify="right")
    table.add_row("mean", f"{statistics.mean(samples):.2f}")
    table.add_row("p50", f"{statistics.median(samples):.2f}")
    table.add_row("p95", f"{sorted(samples)[int(0.95 * (len(samples) - 1))]:.2f}")
    table.add_row("fps", f"{1000.0 / statistics.mean(samples):.1f}")
    console.print(table)


if __name__ == "__main__":
    app()
