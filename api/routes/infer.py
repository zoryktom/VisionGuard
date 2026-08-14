"""Image inference routes."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from api.deps import get_pipeline
from api.schemas import DetectionOut, EventOut, InferResponse
from visionguard.utils.image import pil_to_bgr

router = APIRouter()


def _to_det(d) -> DetectionOut:  # type: ignore[no-untyped-def]
    return DetectionOut(
        bbox=d.bbox,
        class_id=d.class_id,
        class_name=d.class_name,
        confidence=d.confidence,
        category=d.category.value,
        severity=d.severity.value,
    )


def _to_event(e) -> EventOut:  # type: ignore[no-untyped-def]
    return EventOut(
        event_id=e.event_id,
        name=e.name,
        category=e.category.value,
        severity=e.severity.value,
        confidence=e.confidence,
        track_ids=list(e.track_ids),
        bbox=e.bbox,
        frame_index=e.frame_index,
        timestamp=e.timestamp,
        zone=e.zone,
        message=e.message,
    )


@router.post("/infer/image", response_model=InferResponse)
async def infer_image(file: UploadFile = File(...)) -> InferResponse:
    """Run the pipeline on an uploaded image."""

    try:
        image = Image.open(file.file)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    frame = pil_to_bgr(image)
    if frame.size == 0:
        raise HTTPException(status_code=400, detail="Empty image")
    pipe = get_pipeline()
    result = pipe.process_frame(frame)
    return InferResponse(
        detections=[_to_det(d) for d in result.detections],
        events=[_to_event(e) for e in result.events],
        risk_score=result.risk_score,
        inference_ms=result.inference_ms,
        fps=result.fps,
        frame_index=result.frame_index,
    )


@router.post("/infer/array", response_model=InferResponse)
async def infer_array(payload: dict[str, list[list[list[int]]]]) -> InferResponse:
    """Run inference on a raw HxWx3 RGB list (used by notebooks)."""

    rgb = np.asarray(payload.get("image", []), dtype=np.uint8)
    if rgb.ndim != 3:
        raise HTTPException(status_code=400, detail="image must be HxWx3")
    import cv2

    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = get_pipeline().process_frame(frame)
    return InferResponse(
        detections=[_to_det(d) for d in result.detections],
        events=[_to_event(e) for e in result.events],
        risk_score=result.risk_score,
        inference_ms=result.inference_ms,
        fps=result.fps,
        frame_index=result.frame_index,
    )
