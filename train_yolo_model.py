#!/usr/bin/env python3
"""Thin wrapper around Ultralytics YOLO training for the Trill Sparrow dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise SystemExit(
        "Ultralytics is not installed. Install it with 'pip install ultralytics' before training."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO model on the generated dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("yolo-dataset/dataset.yaml"),
        help="Path to the dataset YAML produced by augment_yolo_dataset.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model checkpoint or YAML definition. Defaults to the lightweight YOLO11n weights.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch", type=float, default=16, help="Batch size or auto batch fraction (see docs).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size fed to the network.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device spec, e.g. '0' (GPU), '0,1', 'cpu', or 'mps'. Leave unset for auto.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--project", type=str, default="runs/trills", help="Training output directory.")
    parser.add_argument("--name", type=str, default=None, help="Run name (subdirectory under --project).")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run directory.")
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataloader shuffles.")
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        choices=("ram", "disk"),
        help="Cache images in RAM or on disk to speed up training (optional).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from the weights passed via --model.")
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="Disable validation after each epoch (not recommended unless debugging).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable Ultralytics' built-in data augmentations (use if the dataset is already augmented).",
    )
    return parser.parse_args()


def build_train_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    data_yaml = args.dataset.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    # Ultralytics lets you pass fractional batches (<1) for auto-scaling, but torch's
    # dataloader requires an int when the value represents an actual batch size.
    batch = int(round(args.batch)) if args.batch >= 1 else args.batch

    kwargs: Dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "batch": batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": args.project,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "seed": args.seed,
        "val": not args.skip_val,
    }
    if args.name:
        kwargs["name"] = args.name
    if args.device:
        kwargs["device"] = args.device
    if args.cache:
        kwargs["cache"] = args.cache

    if args.no_augment:
        kwargs.update(
            {
                "augment": False,
                "auto_augment": None,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "scale": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "translate": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "erasing": 0.0,
            }
        )
    return kwargs


def main() -> None:
    args = parse_args()
    train_kwargs = build_train_kwargs(args)

    model = YOLO(args.model)
    results = model.train(resume=args.resume, **train_kwargs)
    if results is None:
        print("Training finished without returning metrics.", file=sys.stderr)


if __name__ == "__main__":
    main()
