# Architecture

VisionGuard is a **frame-synchronous streaming system**, not a notebook that happens to call YOLO.

```
┌────────────┐   BGR frame    ┌──────────────┐   detections   ┌────────────┐
│  Capture   │ ─────────────► │   Detector   │ ─────────────► │  Tracker   │
│ webcam/file│                │ YOLO / VGNet │                │ Kalman+IoU │
│  RTSP/dir  │                │    dummy     │                └─────┬──────┘
└────────────┘                └──────────────┘                      │ tracks
                                                                    ▼
┌────────────┐   HUD frame    ┌──────────────┐   events+risk  ┌────────────┐
│  Overlay   │ ◄───────────── │    Fusion    │ ◄───────────── │   Zones    │
│  renderer  │                │ persist/prox │                │  geofence  │
└─────┬──────┘                └──────┬───────┘                └────────────┘
      │                              │
      ▼                              ▼
 FastAPI MJPEG / CLI video     SSE event bus + Prometheus
```

## Why fusion exists

A single-frame detector is a **proposal generator**. False positives flicker. A forklift 200 px from a pedestrian for 200 ms is not the same event as a 3-second near-miss. `HazardFusion` requires persistence, applies cooldown, and emits interaction events from a proximity graph.

## Detector backends

| Backend | When to use | License note |
|---|---|---|
| `ultralytics` | Production accuracy / COCO zero-shot demo | AGPL-3.0 (Ultralytics). Use only if that is acceptable. |
| `vgnet` | From-scratch training, ONNX export, interviews | MIT (this repo) |
| `dummy` | CI, UI demos, no GPU | MIT |

## Device policy

`utils.device.resolve_device("auto")` prefers **CUDA → MPS → CPU**. FP16 is enabled only on CUDA.

## Latency budget (720p, YOLOv8n)

| Stage | Typical |
|---|---|
| Capture + drop stale | < 3 ms |
| Inference (CUDA T4) | 8–15 ms |
| Track + fusion + overlay | 2–4 ms |
| End-to-end | ~15–25 ms (40–60 FPS) |

Dummy backend on CPU is typically >60 FPS and is what CI measures.
