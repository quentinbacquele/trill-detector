#!/usr/bin/env python3
"""Run trained YOLO detections on WAV files or directories of WAV files and export a CSV."""

from __future__ import annotations

import argparse
import csv
import math
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import os

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="mplcache_"))

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise SystemExit("Ultralytics is not installed. Install it with 'pip install ultralytics'.") from exc

from augment_yolo_dataset import (  # noqa: E402  - reuse helpers
    info_for_audio,
    load_audio_window,
    draw_annotations,
    render_spectrogram,
    write_image,
    YoloBox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slide a trained YOLO model across WAV recordings to detect trills and export detections."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="WAV file(s) or directory roots to scan for WAVs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained Ultralytics YOLO checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("trill_detections.csv"),
        help="Destination CSV for aggregated detections.",
    )
    parser.add_argument(
        "--slice-seconds",
        type=float,
        default=3.0,
        help="Window length (seconds) used when creating spectrogram slices (match training).",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="Stride (seconds) between consecutive windows.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for keeping a detection.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=2,
        help="Maximum number of detections to retain per window.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size passed to YOLO.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device string understood by Ultralytics (e.g., '0', 'cpu', 'mps').",
    )
    parser.add_argument("--figure-width", type=float, default=6.0, help="Spectrogram figure width (inches).")
    parser.add_argument("--figure-height", type=float, default=4.0, help="Spectrogram figure height (inches).")
    parser.add_argument("--dpi", type=int, default=200, help="Spectrogram DPI.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when scanning folders.")
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=None,
        help="Optional directory to write spectrogram PNGs for each retained detection.",
    )
    parser.add_argument(
        "--duration-range",
        type=float,
        nargs=2,
        metavar=("MIN_DURATION", "MAX_DURATION"),
        help="Only keep detections whose durations (seconds) fall inside this inclusive interval.",
    )
    parser.add_argument(
        "--merge-overlaps",
        action="store_true",
        help="Merge overlapping detections within each audio file before writing the CSV.",
    )
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=0.0,
        help="Tolerance (seconds) when deciding if two detections overlap for merging.",
    )
    parser.add_argument(
        "--post-spectrogram-dir",
        type=Path,
        default=None,
        help="Optional directory to write spectrograms after merging, centered on each final detection with a bounding box.",
    )
    parser.add_argument(
        "--post-spectrogram-padding",
        type=float,
        default=0.25,
        help="Extra seconds added before/after each merged detection when rendering post-run spectrograms.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of spectrograms to send to YOLO in a single predict() call.",
    )
    parser.add_argument(
        "--auto-name-outputs",
        action="store_true",
        help="Name the output CSV and spectrogram directory using slice/hop/conf parameters.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress update every N windows (per file). Set to 0 to disable.",
    )
    return parser.parse_args()


def iter_audio_files(inputs: Sequence[str], recursive: bool) -> List[Path]:
    targets: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            print(f"Warning: input path does not exist, skipping: {path}")
            continue
        if path.is_file():
            if path.suffix.lower() == ".wav":
                targets.append(path)
            else:
                print(f"Warning: file is not a WAV, skipping: {path}")
        elif path.is_dir():
            matches = sorted(path.rglob("*.wav")) if recursive else sorted(path.glob("*.wav"))
            targets.extend(match for match in matches if match.is_file())
        else:
            print(f"Warning: unsupported path type, skipping: {path}")
    return targets


def sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("_") or "audio"


def spectrogram_path_for_detection(
    base_dir: Path,
    audio_path: Path,
    window_index: int,
    detection_index: int,
    physical: Dict[str, float],
    confidence: float,
) -> Path:
    audio_component = sanitize_component(audio_path.stem or audio_path.name)
    filename = (
        f"{audio_component}"
        f"_win{window_index:05d}"
        f"_det{detection_index:02d}"
        f"_start{physical['start_time']:07.2f}s"
        f"_end{physical['end_time']:07.2f}s"
        f"_conf{confidence:.2f}.png"
    )
    return base_dir / filename


def generate_window_starts(duration: float, window: float, hop: float) -> Iterator[float]:
    if duration < window:
        return
    start = 0.0
    last_start = duration - window
    while start <= last_start + 1e-6:
        yield start
        start += hop
    if not math.isclose((start - hop), last_start, rel_tol=1e-6, abs_tol=1e-6) and last_start > 0:
        yield last_start


def convert_box_to_physical(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    window_start: float,
    slice_seconds: float,
    freq_max: float,
) -> Dict[str, float]:
    time_center = x_center * slice_seconds
    time_width = width * slice_seconds
    call_start = window_start + time_center - time_width / 2.0
    call_end = window_start + time_center + time_width / 2.0
    call_start = max(window_start, call_start)
    call_end = min(window_start + slice_seconds, call_end)

    freq_center = (1.0 - y_center) * freq_max
    freq_height = height * freq_max
    freq_low = max(0.0, freq_center - freq_height / 2.0)
    freq_high = min(freq_max, freq_center + freq_height / 2.0)

    return {
        "start_time": call_start,
        "end_time": call_end,
        "duration": max(0.0, call_end - call_start),
        "low_freq": freq_low,
        "high_freq": freq_high,
    }


def resolve_device(preferred: str | None) -> str:
    """Choose an inference device, preferring user input then GPU/MPS if available."""
    if preferred:
        return preferred
    try:
        import torch

        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Warning: unable to query torch devices: {exc}")
    return "cpu"


def param_suffix(args: argparse.Namespace) -> str:
    suffix = f"s{args.slice_seconds:.2f}_h{args.hop_seconds:.2f}_c{args.confidence:.2f}"
    return sanitize_component(suffix)


def merge_overlapping(
    detections: List[Dict[str, float]],
    tolerance: float,
) -> List[Dict[str, float]]:
    """Merge overlapping detections within a single audio file."""
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["start_time"])
    merged: List[Dict[str, float]] = []
    current = sorted_dets[0].copy()
    for det in sorted_dets[1:]:
        if det["start_time"] <= current["end_time"] + tolerance:
            current["end_time"] = max(current["end_time"], det["end_time"])
            current["duration"] = max(0.0, current["end_time"] - current["start_time"])
            current["low_freq"] = min(current["low_freq"], det["low_freq"])
            current["high_freq"] = max(current["high_freq"], det["high_freq"])
            current["confidence"] = max(current["confidence"], det["confidence"])
            current["window_index"] = min(current["window_index"], det["window_index"])
            current["window_start"] = min(current["window_start"], det["window_start"])
            current["window_end"] = max(current["window_end"], det["window_end"])
        else:
            merged.append(current)
            current = det.copy()
    merged.append(current)
    return merged


def render_post_spectrogram(
    audio_path: Path,
    detection: Dict[str, float],
    padding: float,
    figure_size: tuple[float, float],
    dpi: int,
) -> tuple[Path, object]:
    """Render a spectrogram centered on a merged detection with a bounding box overlay."""
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

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    y_height = min(y_height, 1.0)
    width_time = min(width_time / window_len if window_len > 0 else 1.0, 1.0)

    box = YoloBox(class_id=0, x_center=x_center, y_center=y_center, width=width_time, height=y_height)
    annotated = draw_annotations(image, [box])

    audio_component = sanitize_component(audio_path.stem or audio_path.name)
    filename = (
        f"{audio_component}"
        f"_det{int(detection['detection_index']):04d}"
        f"_start{detection['start_time']:07.2f}s"
        f"_end{detection['end_time']:07.2f}s"
        f"_conf{detection['confidence']:.2f}.png"
    )
    return Path(filename), annotated


def run_inference(args: argparse.Namespace) -> None:
    audio_files = iter_audio_files(args.inputs, args.recursive)
    if not audio_files:
        raise SystemExit("No WAV files found in the provided inputs.")

    base_dir = Path(os.path.commonpath([str(p.parent) for p in audio_files]))
    device = resolve_device(args.device)
    model = YOLO(args.model)
    audio_cache: Dict[Path, tuple[int, float]] = {}
    if args.output_csv == Path("trill_detections.csv"):
        args.output_csv = base_dir / args.output_csv.name
    if args.auto_name_outputs:
        suffix = param_suffix(args)
        args.output_csv = args.output_csv.with_name(f"{args.output_csv.stem}_{suffix}{args.output_csv.suffix}")
    if args.spectrogram_dir is None:
        dir_name = f"spectrograms_{param_suffix(args)}" if args.auto_name_outputs else "spectrograms"
        args.spectrogram_dir = base_dir / dir_name
    spectrogram_dir = args.spectrogram_dir
    if args.post_spectrogram_dir is not None and not args.post_spectrogram_dir.is_absolute():
        args.post_spectrogram_dir = base_dir / args.post_spectrogram_dir
    if spectrogram_dir is not None:
        spectrogram_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving detection spectrograms to: {spectrogram_dir}")
    else:
        print("Spectrogram saving disabled; pass --spectrogram-dir DIR to save detection crops.")
    if args.post_spectrogram_dir is not None:
        args.post_spectrogram_dir.mkdir(parents=True, exist_ok=True)
        print(f"Post-merge spectrograms will be written to: {args.post_spectrogram_dir}")

    print(f"Processing {len(audio_files)} WAV file(s) with model {args.model} on device '{device}'.")

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

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_detections = 0
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for file_idx, audio_path in enumerate(audio_files, start=1):
            try:
                samplerate, duration = info_for_audio(audio_path, audio_cache)
            except Exception as exc:
                print(f"Warning: failed to read {audio_path}: {exc}")
                continue

            if duration < args.slice_seconds:
                print(f"Warning: skipping {audio_path} (duration < slice length).")
                continue

            freq_max = samplerate / 2.0
            windows = list(generate_window_starts(duration, args.slice_seconds, args.hop_seconds))
            if not windows:
                continue
            print(
                f"[{file_idx}/{len(audio_files)}] {audio_path.name}: {duration:.1f}s audio -> {len(windows)} windows",
                flush=True,
            )

            raw_detections = 0
            detections: List[Dict[str, float]] = []
            batch_images: List = []
            batch_rgb: List = []
            batch_meta: List[Dict[str, float]] = []

            def flush_batch() -> None:
                nonlocal raw_detections
                if not batch_images:
                    return
                results = model.predict(
                    source=batch_images,
                    verbose=False,
                    conf=args.confidence,
                    max_det=args.max_detections,
                    imgsz=args.imgsz,
                    device=device,
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
                            args.slice_seconds,
                            freq_max,
                        )
                        if args.duration_range is not None:
                            min_duration, max_duration = args.duration_range
                            duration = physical["duration"]
                            if duration < min_duration or duration > max_duration:
                                continue
                        raw_detections += 1
                        detections.append(
                            {
                                "audio_path": str(audio_path),
                                "window_index": meta["window_index"],
                                "window_start": meta["window_start"],
                                "window_end": meta["window_start"] + args.slice_seconds,
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
                        if spectrogram_dir is not None:
                            image_path = spectrogram_path_for_detection(
                                spectrogram_dir,
                                audio_path,
                                meta["window_index"],
                                det_idx,
                                physical,
                                float(conf),
                            )
                            write_image(image_path, rgb)
                batch_images.clear()
                batch_rgb.clear()
                batch_meta.clear()

            for window_idx, window_start in enumerate(windows):
                samples = load_audio_window(audio_path, samplerate, window_start, args.slice_seconds)
                if samples.size > 0:
                    image = render_spectrogram(
                        samples,
                        samplerate,
                        args.slice_seconds,
                        freq_max,
                        (args.figure_width, args.figure_height),
                        args.dpi,
                    )
                    batch_images.append(image[..., ::-1])
                    batch_rgb.append(image)
                    batch_meta.append({"window_index": window_idx, "window_start": window_start})
                    if len(batch_images) >= args.batch_size:
                        flush_batch()
                else:
                    # No samples; nothing to batch for prediction
                    pass

                if args.progress_every > 0 and ((window_idx + 1) % args.progress_every == 0 or window_idx == len(windows) - 1):
                    print(
                        f"  processed {window_idx + 1}/{len(windows)} windows, detections so far: {raw_detections}",
                        flush=True,
                    )

            flush_batch()

            final_detections = (
                merge_overlapping(detections, args.merge_tolerance) if args.merge_overlaps else detections
            )
            for det_idx, det in enumerate(final_detections, start=1):
                det["detection_index"] = det_idx
                writer.writerow({k: det.get(k, "") for k in fieldnames})
                total_detections += 1
                if args.post_spectrogram_dir is not None:
                    rel_path, annotated = render_post_spectrogram(
                        audio_path,
                        det,
                        args.post_spectrogram_padding,
                        (args.figure_width, args.figure_height),
                        args.dpi,
                    )
                    write_image(args.post_spectrogram_dir / rel_path, annotated)

            merged_count = len(final_detections)
            if merged_count == 0:
                print(f"  No detections retained for {audio_path}.")
            else:
                if args.merge_overlaps:
                    print(
                        f"  Finished {audio_path}: {merged_count} merged detections (raw={raw_detections}).",
                        flush=True,
                    )
                else:
                    print(f"  Finished {audio_path}: {merged_count} detections.", flush=True)

    print(f"Done. Wrote {total_detections} detections to {args.output_csv}.")


def main() -> None:
    args = parse_args()
    if args.slice_seconds <= 0:
        raise SystemExit("--slice-seconds must be positive.")
    if args.hop_seconds <= 0:
        raise SystemExit("--hop-seconds must be positive.")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be zero or positive.")
    if args.merge_tolerance < 0:
        raise SystemExit("--merge-tolerance must be non-negative.")
    if args.post_spectrogram_padding < 0:
        raise SystemExit("--post-spectrogram-padding must be non-negative.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.duration_range is not None:
        min_duration, max_duration = args.duration_range
        if min_duration < 0 or max_duration < 0:
            raise SystemExit("--duration-range values must be non-negative.")
        if min_duration > max_duration:
            raise SystemExit("--duration-range minimum must be <= maximum.")
    run_inference(args)


if __name__ == "__main__":
    main()
