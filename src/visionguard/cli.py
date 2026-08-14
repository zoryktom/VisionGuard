"""VisionGuard command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from visionguard.__about__ import __app_name__, __version__
from visionguard.logging_utils import setup_logging
from visionguard.pipeline import RealTimePipeline

app = typer.Typer(help="VisionGuard — real-time visual hazard detection", no_args_is_help=True)
console = Console()


def _banner() -> None:
    console.print(f"[bold green]{__app_name__}[/] [dim]v{__version__}[/]")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
) -> None:
    """Global options."""

    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def detect(
    source: str = typer.Option(
        "0", "--source", "-s", help="Webcam index, video file, RTSP URL, or image folder"
    ),
    config: Path | None = typer.Option(None, "--config", "-c"),
    backend: str = typer.Option("dummy", "--backend", "-b", help="ultralytics | vgnet | dummy"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Annotated video path"),
    zones: Path | None = typer.Option(None, "--zones"),
    display: bool = typer.Option(False, "--display", "-d"),
    headless: bool = typer.Option(False, "--headless"),
    max_frames: int | None = typer.Option(None, "--max-frames"),
) -> None:
    """Run the real-time detection pipeline on a video source."""

    _banner()
    pipeline = RealTimePipeline.from_files(config, zones, backend=backend)
    results = pipeline.run_file(
        source=source,
        output=str(output) if output else None,
        display=display and not headless,
        max_frames=max_frames,
    )
    events = sum(len(r.events) for r in results)
    table = Table(title="Run summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Frames", str(len(results)))
    table.add_row("Events", str(events))
    if results:
        table.add_row("Mean FPS", f"{sum(r.fps for r in results) / len(results):.1f}")
        table.add_row("Peak risk", f"{max(r.risk_score for r in results):.1f}")
        table.add_row(
            "Mean inference (ms)", f"{sum(r.inference_ms for r in results) / len(results):.1f}"
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    backend: str = typer.Option("dummy", "--backend"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Launch the FastAPI dashboard and inference API."""

    import os

    import uvicorn

    os.environ["VISIONGUARD_MODEL_BACKEND"] = backend
    _banner()
    console.print(f"Dashboard → [bold]http://{host}:{port}[/]")
    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/training.yaml"), "--config", "-c"),
    engine: str = typer.Option(
        "vgnet", "--engine", help="vgnet (from scratch) or yolo (Ultralytics)"
    ),
    images: Path = typer.Option(Path("data/datasets/synthetic/images"), "--images"),
    epochs: int = typer.Option(5, "--epochs"),
) -> None:
    """Train a detector on a YOLO-format dataset."""

    _banner()
    if engine == "yolo":
        from visionguard.training.trainer import train_ultralytics

        path = train_ultralytics(config)
    else:
        from visionguard.training.trainer import train_vgnet

        path = train_vgnet(images_dir=images, epochs=epochs)
    console.print(f"[green]Checkpoint[/] {path}")


@app.command()
def export(
    config: Path = typer.Option(Path("configs/training.yaml"), "--config", "-c"),
    format: str = typer.Option("onnx", "--format"),
    vgnet: Path | None = typer.Option(None, "--vgnet", help="VGNet .pt checkpoint"),
) -> None:
    """Export a trained checkpoint to ONNX / TorchScript."""

    from visionguard.training.trainer import export_weights

    paths = export_weights(config, format=format, vgnet_checkpoint=vgnet)
    for path in paths:
        console.print(f"[green]Exported[/] {path}")


@app.command()
def evaluate(
    images: Path = typer.Option(Path("data/datasets/synthetic/images")),
    labels: Path = typer.Option(Path("data/datasets/synthetic/labels")),
    backend: str = typer.Option("dummy"),
) -> None:
    """Evaluate a detector against YOLO-format labels (mAP@0.5)."""

    from visionguard.capture import ImageFolderSource
    from visionguard.training.dataset import parse_yolo_label
    from visionguard.training.evaluate import evaluate_detections

    _banner()
    pipeline = RealTimePipeline.from_files(backend=backend)
    src = ImageFolderSource(images)
    preds = []
    gts = []
    src.open()
    try:
        idx = 0
        while True:
            frame = src.read()
            if frame is None:
                break
            result = pipeline.process_frame(frame)
            stem = sorted(p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".png"})[idx]
            idx += 1
            labels_list = parse_yolo_label(labels / f"{stem.stem}.txt")
            h, w = frame.shape[:2]
            preds.append(
                [(t.bbox, t.detection.class_id, t.ema_confidence) for t in result.tracks]
                or [(d.bbox, d.class_id, d.confidence) for d in result.detections]
            )
            gts.append([(*lb.xyxy(w, h), lb.class_id) for lb in labels_list])
    finally:
        src.close()
    metrics = evaluate_detections(preds, gts)
    table = Table(title="Detection metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.as_dict().items():
        table.add_row(key, f"{value:.3f}")
    console.print(table)


@app.command()
def version() -> None:
    """Print the package version."""

    console.print(__version__)


if __name__ == "__main__":
    app()
