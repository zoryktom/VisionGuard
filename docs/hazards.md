# Hazard taxonomy

Three families of hazards, plus **interaction** events that are not detector classes:

| Category | Examples | Typical severity |
|---|---|---|
| Behavior | no helmet, phone use, fall, restricted entry | medium–critical |
| Environment | fire, smoke, spill, blocked exit | high–critical |
| Object | vehicle, knife, chemical, weapon, cable | medium–critical |
| Interaction | pedestrian–vehicle near-miss, person near fire | high–critical |

COCO-pretrained YOLO is mapped through `coco_aliases` so a laptop webcam demo works on day one (`person`, `cell phone`, `knife`, `truck`, …). PPE, fire, and smoke require a **custom fine-tune**.

Temporal rules (see `HazardFusion`):

- Persist *N* frames before emitting (default 8).
- Cooldown 4 s per hypothesis so the timeline is readable.
- Near-miss if two rule-matched centroids are within `near_miss_px`.
- Zone multipliers: restricted > exit > PPE > vehicle lane.

This is **not** a SIL-rated safety function. Treat events as operator-assist signals.
