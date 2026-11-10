"""Generate synthetic data for testing training and prediction commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _sequence(shape: tuple[int, int, int], seed: int):
    rng = np.random.default_rng(seed)
    coordinates = np.indices(shape, dtype=np.float32)
    centre = (np.asarray(shape) - 1) / 2 + rng.uniform(-2, 2, size=3)
    radii = np.asarray(shape) * rng.uniform(0.30, 0.38, size=3)
    interior = sum(
        ((coordinates[axis] - centre[axis]) / radii[axis]) ** 2
        for axis in range(3)
    ) < 1
    levels = np.asarray([0.2, 0.55, 0.85], dtype=np.float32)
    for time_step in range(3):
        front = centre[2] + (time_step - 1) * shape[2] * 0.12
        labels = np.zeros(shape, dtype=np.int16)
        labels[interior] = 2
        labels[interior & (coordinates[2] < front)] = 1
        noise = rng.normal(0, 0.025, size=shape).astype(np.float32)
        images = np.clip(levels[labels] + noise, 0, 1).astype(np.float32)
        yield images, labels


def generate_demo(output: str | Path) -> Path:
    """Write independent synthetic sequences and a small training configuration."""
    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("The output must be a new or empty directory.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("The output directory is not empty; existing files will not be overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    shape = (32, 32, 64)
    manifest = {}
    for split, seed in (("training", 17), ("validation", 29), ("prediction", 43)):
        split_directory = output / split
        split_directory.mkdir()
        sequence = {"id": split, "images": [], "labels": {}}
        for index, (images, labels) in enumerate(_sequence(shape, seed)):
            image_path = Path(split) / f"image_{index:03d}.npy"
            np.save(output / image_path, images, allow_pickle=False)
            sequence["images"].append(image_path.as_posix())
            if split != "prediction" and index in (0, 2):
                label_path = Path(split) / f"label_{index:03d}.npy"
                np.save(output / label_path, labels, allow_pickle=False)
                sequence["labels"][str(index)] = label_path.as_posix()
        manifest[split] = [sequence]

    config = {
        "model": {
            "in_channels": 1,
            "out_channels": 3,
            "temporal_size": 3,
            "feature_size": 12,
            "depths": [1, 1, 1, 1],
            "num_heads": [3, 6, 12, 24],
            "use_checkpoint": False,
        },
        "data": {
            "patch_size": list(shape),
            "temporal_strides": [1],
            "samples_per_epoch": 2,
            "cache_size": 3,
            "normalization": {"mode": "fixed", "lower": 0, "upper": 1},
            "ignore_index": -1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "warmup_epochs": 0,
            "min_learning_rate": 1e-7,
            "validation_interval": 1,
            "num_workers": 0,
            "seed": 42,
            "precision": "float32",
            "augmentation": {"spatial_flip": True, "temporal_reverse": True},
            "gradient_clip": 1.0,
        },
        "loss": {
            "dice_weight": 1,
            "ce_weight": 1,
            "label_smoothing": 0.05,
            "class_weights": None,
        },
        "inference": {"overlap": 0.5, "sw_batch_size": 1, "stitch_device": "cpu"},
    }
    for name, content in (("manifest.json", manifest), ("config.json", config)):
        (output / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small synthetic dataset to try the training and prediction workflow."
    )
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args()
    try:
        output = generate_demo(args.output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Synthetic data: {output}")
    print(f"Manifest: {output / 'manifest.json'}")
    print(f"Configuration: {output / 'config.json'}")


if __name__ == "__main__":
    main()
