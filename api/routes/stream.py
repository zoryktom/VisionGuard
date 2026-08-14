"""MJPEG live stream."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.deps import get_pipeline
from visionguard.capture import open_source, synthetic_frame
from visionguard.utils.image import encode_jpeg

router = APIRouter()

_STOP = threading.Event()


def _frame_bytes() -> Iterator[bytes]:
    """Yield multipart MJPEG frames from webcam, else synthetic demo."""

    pipe = get_pipeline()
    quality = pipe.config.api.mjpeg_quality
    source_id = pipe.config.capture.source
    src = None
    try:
        src = open_source(source_id, pipe.config.capture)
        src.open()
        live = True
    except Exception:
        live = False
        src = None

    idx = 0
    try:
        while not _STOP.is_set():
            if live and src is not None:
                frame = src.read()
                if frame is None:
                    live = False
                    continue
            else:
                frame = synthetic_frame(1280, 720, seed=idx)
                idx += 1
                time.sleep(1 / 15)
            result = pipe.process_frame(frame)
            annotated = pipe.annotate(result)
            payload = encode_jpeg(annotated, quality)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
                + b"\r\n"
            )
    finally:
        if src is not None:
            src.close()


@router.get("/stream/mjpeg")
def mjpeg() -> StreamingResponse:
    """Live annotated MJPEG stream for the dashboard ``<img>`` tag."""

    return StreamingResponse(
        _frame_bytes(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/stream/snapshot")
def snapshot() -> StreamingResponse:
    """Single annotated JPEG (synthetic if no camera)."""

    pipe = get_pipeline()
    frame = synthetic_frame(1280, 720, seed=int(time.time()) % 10_000)
    result = pipe.process_frame(frame)
    annotated = pipe.annotate(result)
    return StreamingResponse(
        iter([encode_jpeg(annotated, pipe.config.api.mjpeg_quality)]),
        media_type="image/jpeg",
    )
