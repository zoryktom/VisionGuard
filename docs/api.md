# HTTP API

Base URL: `http://127.0.0.1:8000`

Interactive docs: `/docs` (Swagger) and `/redoc`.

## Operations

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | Liveness |
| GET | `/api/v1/ready` | Pipeline constructed |
| GET | `/api/v1/stats` | Rolling FPS / latency / risk |
| GET | `/api/v1/metrics` | Prometheus text |
| GET | `/api/v1/events` | Recent hazard events |
| GET | `/api/v1/events/stream` | SSE event stream |
| GET | `/api/v1/stream/mjpeg` | Annotated MJPEG |
| GET | `/api/v1/stream/snapshot` | Single JPEG |
| POST | `/api/v1/infer/image` | Multipart image inference |

## Infer an image

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/infer/image \
  -F "file=@frame.jpg" | jq
```

Response:

```json
{
  "detections": [
    {
      "bbox": [100.0, 80.0, 240.0, 400.0],
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "category": "behavior",
      "severity": "low"
    }
  ],
  "events": [],
  "risk_score": 12.4,
  "inference_ms": 9.2,
  "fps": 48.1,
  "frame_index": 17
}
```

## Auth

The stock server is **open**. Put it behind a reverse proxy with mTLS or a token gateway before any non-local deployment.
