# Contributing to VisionGuard

Thanks for helping improve a real-time visual safety stack.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make install-dev
```

## Checks before a PR

```bash
make format
make lint
make test
```

## Design notes

- Keep detector backends behind `visionguard.inference.Detector`.
- Temporal logic belongs in `HazardFusion`, not in the overlay or API.
- Tests must run with `VISIONGUARD_MODEL_BACKEND=dummy` (no GPU, no weight download).
- Do not commit checkpoints (`.pt`, `.onnx`) or raw videos.

## Safety

VisionGuard is a research / portfolio system. Changes that imply certified functional-safety behavior need an explicit disclaimer in the PR.
