"""Event feed routes."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.deps import get_pipeline
from api.routes.infer import _to_event
from api.schemas import EventOut

router = APIRouter()


@router.get("/events", response_model=list[EventOut])
def list_events(limit: int = 50) -> list[EventOut]:
    """Most recent hazard events (newest last)."""

    events = list(get_pipeline().telemetry.events)[-limit:]
    return [_to_event(e) for e in events]


@router.get("/events/stream")
async def event_sse() -> StreamingResponse:
    """Server-sent events of the telemetry ring buffer."""

    async def gen():  # type: ignore[no-untyped-def]
        last = 0
        while True:
            events = list(get_pipeline().telemetry.events)
            if len(events) > last:
                for event in events[last:]:
                    payload = _to_event(event).model_dump()
                    yield f"data: {json.dumps(payload)}\n\n"
                last = len(events)
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")
