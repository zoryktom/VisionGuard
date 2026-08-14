#!/usr/bin/env python3
"""Train VGNet on the bundled synthetic dataset."""

from visionguard.training import train_vgnet


def main() -> None:
    ckpt = train_vgnet(
        images_dir="data/datasets/synthetic/images",
        labels_dir="data/datasets/synthetic/labels",
        epochs=8,
        imgsz=320,
        batch_size=8,
        output="models/checkpoints/vgnet.pt",
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
