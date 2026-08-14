"""Render a short annotated demo GIF from synthetic frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import typer
from PIL import Image
from rich.console import Console

from visionguard.capture import synthetic_frame
from visionguard.pipeline import RealTimePipeline

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    frames: int = typer.Option(8),
    out: Path = typer.Option(Path("assets/demo.gif")),
) -> None:
    """Write ``assets/demo.gif`` using the dummy backend."""

    pipe = RealTimePipeline.from_files(backend="dummy", zones_path="configs/zones.example.yaml")
    images = []
    for i in range(frames):
        frame = synthetic_frame(480, 270, seed=i)
        result = pipe.process_frame(frame)
        annotated = pipe.annotate(result)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=48)
        images.append(img)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    still = out.with_name("overlay-preview.png")
    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(still, optimize=True)
    console.print(f"[green]Wrote[/] {out} and {still}")


if __name__ == "__main__":
    app()
