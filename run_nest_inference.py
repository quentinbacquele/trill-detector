#!/usr/bin/env python3
"""Run YOLO inference over Nest Recordings and save per-file merged outputs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
from ultralytics import YOLO

from augment_yolo_dataset import (
    info_for_audio,
    load_audio_window,
    render_spectrogram,
    draw_annotations,
    YoloBox,
    write_image,
)
from infer_trills import (
    convert_box_to_physical,
    generate_window_starts,
    merge_overlapping,
    sanitize_component,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference over Nest Recordings; saves merged CSV + merged spectrograms per WAV."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/Volumes/My Passport/2024_SAVS/2024_SAVS_SMM_Recordings/Nest Recordings"),
        help="Root directory containing Nest Recording WAVs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Nest Detections"),
        help="Where to mirror the folder structure with detections/spectrograms.",
    )
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"), help="Path to YOLO checkpoint.")
    parser.add_argument("--slice-seconds", type=float, default=3.0, help="Spectrogram window length (s).")
    parser.add_argument("--hop-seconds", type=float, default=1.0, help="Stride between windows (s).")
    parser.add_argument("--confidence", type=float, default=0.75, help="Confidence threshold.")
    parser.add_argument("--max-detections", type=int, default=2, help="Max detections per window.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model.predict().")
    parser.add_argument("--figure-width", type=float, default=6.0, help="Spectrogram figure width.")
    parser.add_argument("--figure-height", type=float, default=4.0, help="Spectrogram figure height.")
    parser.add_argument("--dpi", type=int, default=200, help="Spectrogram DPI.")
    parser.add_argument("--post-padding", type=float, default=0.25, help="Seconds to pad merged detections.")
    parser.add_argument("--device", type=str, default="mps", help="Computation device (e.g., 'mps', 'cpu', '0').")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N windows per WAV (set 0 to disable).",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (>=1). Each worker loads its own model.",
    )
    return parser.parse_args()


def list_wavs(root: Path) -> List[Path]:
    return [
        p
        for p in sorted(root.rglob("*.wav"))
        if p.is_file() and "spacer" not in p.name.lower()
    ]


def render_post_spectrogram(
    audio_path: Path,
    detection: Dict[str, float],
    padding: float,
    figure_size: Tuple[float, float],
    dpi: int,
) -> Tuple[str, object]:
    samplerate = int(detection["samplerate"])
    freq_max = float(detection["freq_max"])
    start = max(0.0, detection["start_time"] - padding)
    end = detection["end_time"] + padding
    window_len = max(end - start, 1e-3)
    samples = load_audio_window(audio_path, samplerate, start, window_len)
    image = render_spectrogram(samples, samplerate, window_len, freq_max, figure_size, dpi)

    det_start_local = max(detection["start_time"] - start, 0.0)
    det_end_local = min(detection["end_time"] - start, window_len)
    width_time = max(det_end_local - det_start_local, 1e-6)
    x_center = (det_start_local + det_end_local) / 2.0 / window_len

    low = max(0.0, min(detection["low_freq"], freq_max))
    high = max(0.0, min(detection["high_freq"], freq_max))
    low, high = (min(low, high), max(low, high))
    y_center = 1.0 - ((low + high) / 2.0) / freq_max
    y_height = max(high - low, 1e-6) / freq_max

    box = YoloBox(class_id=0, x_center=x_center, y_center=y_center, width=width_time / window_len, height=y_height)
    annotated = draw_annotations(image, [box])

    audio_component = sanitize_component(audio_path.stem or audio_path.name)
    filename = (
        f"{audio_component}"
        f"_det{int(detection['detection_index']):04d}"
        f"_start{detection['start_time']:07.2f}s"
        f"_end{detection['end_time']:07.2f}s"
        f"_conf{detection['confidence']:.2f}.png"
    )
    return filename, annotated


def process_file(
    audio_path: Path,
    input_root: Path,
    output_root: Path,
    model_path: Path,
    device: str,
    slice_seconds: float,
    hop_seconds: float,
    confidence: float,
    max_detections: int,
    imgsz: int,
    batch_size: int,
    fig_size: Tuple[float, float],
    dpi: int,
    post_padding: float,
    progress_every: int,
) -> Tuple[Path, int, int, Optional[Path]]:
    model = YOLO(model_path)
    audio_cache: Dict[Path, tuple[int, float]] = {}

    rel = audio_path.relative_to(input_root)
    out_dir = output_root / rel.parent / audio_path.stem
    csv_path = out_dir / f"{audio_path.stem}_detections.csv"

    try:
        samplerate, duration = info_for_audio(audio_path, audio_cache)
    except Exception as exc:
        print(f"Warning: failed to read {audio_path}: {exc}")
        return audio_path, 0, 0, None
    if duration < slice_seconds:
        print(f"Warning: skipping {audio_path} (duration < slice length).")
        return audio_path, 0, 0, None

    freq_max = samplerate / 2.0
    windows = list(generate_window_starts(duration, slice_seconds, hop_seconds))
    if not windows:
        return

    fieldnames = [
        "audio_path",
        "window_index",
        "window_start",
        "window_end",
        "detection_index",
        "start_time",
        "end_time",
        "duration",
        "low_freq",
        "high_freq",
        "confidence",
    ]

    detections: List[Dict[str, float]] = []
    batch_images: List = []
    batch_rgb: List = []
    batch_meta: List[Dict[str, float]] = []
    raw_dets = 0

    def flush_batch() -> None:
        nonlocal raw_dets
        if not batch_images:
            return
        results = model.predict(
            source=batch_images,
            verbose=False,
            conf=confidence,
            max_det=max_detections,
            imgsz=imgsz,
            device=device,
            batch=batch_size,
        )
        for res, rgb, meta in zip(results, batch_rgb, batch_meta):
            boxes = res.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for det_idx, (coords, conf) in enumerate(zip(xywhn, confs), start=1):
                x_center, y_center, width, height = coords
                physical = convert_box_to_physical(
                    float(x_center),
                    float(y_center),
                    float(width),
                    float(height),
                    meta["window_start"],
                    slice_seconds,
                    freq_max,
                )
                raw_dets += 1
                detections.append(
                    {
                        "audio_path": str(audio_path),
                        "window_index": meta["window_index"],
                        "window_start": meta["window_start"],
                        "window_end": meta["window_start"] + slice_seconds,
                        "detection_index": det_idx,
                        "start_time": physical["start_time"],
                        "end_time": physical["end_time"],
                        "duration": physical["duration"],
                        "low_freq": physical["low_freq"],
                        "high_freq": physical["high_freq"],
                        "confidence": float(conf),
                        "freq_max": freq_max,
                        "samplerate": samplerate,
                    }
                )
        batch_images.clear()
        batch_rgb.clear()
        batch_meta.clear()

    for window_idx, window_start in enumerate(windows):
        samples = load_audio_window(audio_path, samplerate, window_start, slice_seconds)
        if samples.size == 0:
            continue
        image = render_spectrogram(
            samples,
            samplerate,
            slice_seconds,
            freq_max,
            fig_size,
            dpi,
        )
        batch_images.append(image[..., ::-1])
        batch_rgb.append(image)
        batch_meta.append({"window_index": window_idx, "window_start": window_start})
        if len(batch_images) >= batch_size:
            flush_batch()
        if progress_every > 0 and ((window_idx + 1) % progress_every == 0 or window_idx == len(windows) - 1):
            print(
                f"  [{audio_path.name}] window {window_idx + 1}/{len(windows)}"
                f" | raw so far: {raw_dets}",
                flush=True,
            )
    flush_batch()

    final_detections = merge_overlapping(detections, 0.0)

    # Only create output directory when we have something to write
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = out_dir
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for det_idx, det in enumerate(final_detections, start=1):
            det["detection_index"] = det_idx
            writer.writerow({k: det.get(k, "") for k in fieldnames})
            # Save merged spectrogram with bbox
            fname, annotated = render_post_spectrogram(
                audio_path,
                det,
                post_padding,
                fig_size,
                dpi,
            )
            write_image(merged_dir / fname, annotated)

    print(f"[{audio_path.name}] raw={raw_dets} merged={len(final_detections)} -> {csv_path}")
    return audio_path, raw_dets, len(final_detections), csv_path


def main() -> None:
    args = parse_args()
    wavs = list_wavs(args.input_root)
    if not wavs:
        raise SystemExit("No WAV files found under input-root.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    worker_fn = partial(
        process_file,
        input_root=args.input_root,
        output_root=args.output_root,
        model_path=args.model,
        device=device,
        slice_seconds=args.slice_seconds,
        hop_seconds=args.hop_seconds,
        confidence=args.confidence,
        max_detections=args.max_detections,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        fig_size=(args.figure_width, args.figure_height),
        dpi=args.dpi,
        post_padding=args.post_padding,
        progress_every=args.progress_every,
    )

    workers = max(1, args.file_workers)
    total = len(wavs)
    completed = 0

    if workers == 1:
        for wav in wavs:
            _, raw_dets, merged, csv_path = worker_fn(wav)
            completed += 1
            print(f"[{completed}/{total}] {wav.name}: raw={raw_dets} merged={merged} csv={csv_path}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(worker_fn, wav): wav for wav in wavs}
            try:
                for fut in as_completed(futures):
                    audio_path, raw_dets, merged, csv_path = fut.result()
                    completed += 1
                    name = audio_path.name if isinstance(audio_path, Path) else str(audio_path)
                    print(f"[{completed}/{total}] {name}: raw={raw_dets} merged={merged} csv={csv_path}")
            except KeyboardInterrupt:
                print("Ctrl+C received; terminating workers...", file=sys.stderr)
                for fut in futures:
                    fut.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()
