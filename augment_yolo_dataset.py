#!/usr/bin/env python3
"""Build and augment a YOLO detection dataset directly from processed annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import os

import matplotlib  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="mplcache_"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402


@dataclass
class AnnotationRecord:
    index: int
    soundfile: str
    channel: int
    left: float
    right: float
    top_freq: float
    bottom_freq: float
    comment: str
    notes: str
    folder_name: str = ""
    metafile_name: str = ""
    annotation_file_name: str = ""
    annotation_name: str = ""
    audio_path: str = ""

    @property
    def duration(self) -> float:
        return max(self.right - self.left, 0.0)


@dataclass
class DatasetStats:
    count: int
    mean_duration: float
    std_duration: float
    slice_duration: float


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an augmented YOLO dataset from processed_annotations.csv and source WAV files."
    )
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=Path("processed_annotations.csv"),
        help="CSV produced by process_trill_metadata.py.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("."),
        help="Directory containing the source WAV files referenced in the annotations CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("yolo-dataset"),
        help="Destination directory for the generated dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove any existing --output-dir before writing the dataset.",
    )
    parser.add_argument(
        "--slice-seconds",
        type=float,
        help="Override slice duration (seconds). When omitted the value is mean + 4*std of annotation durations.",
    )
    parser.add_argument(
        "--min-slice-seconds",
        type=float,
        default=1.0,
        help="Lower bound enforced on the computed slice duration.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=6.0,
        help="Width of the spectrogram figure (inches).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.0,
        help="Height of the spectrogram figure (inches).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI used when rasterising spectrograms.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of annotations assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of annotations assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of annotations assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed used for split assignment and augmentation sampling.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class id to assign to every annotation.",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="call",
        help="Human-readable name for the single class.",
    )
    parser.add_argument(
        "--shift-variants",
        type=int,
        default=2,
        help="Number of additional time-shifted variants (per annotation) generated for the training split.",
    )
    parser.add_argument(
        "--shift-min-seconds",
        type=float,
        default=0.0,
        help="Minimum absolute shift (seconds) applied when sampling a shift variant.",
    )
    parser.add_argument(
        "--shift-step-seconds",
        type=float,
        default=0.1,
        help="Resolution (seconds) used when enumerating candidate shift positions.",
    )
    parser.add_argument(
        "--shift-max-seconds",
        type=float,
        default=5.0,
        help="Maximum absolute shift (seconds) applied when sampling a shift variant.",
    )
    parser.add_argument(
        "--horizontal-flip",
        action="store_true",
        help="Generate a horizontally flipped variant for each (shifted) training sample.",
    )
    parser.add_argument(
        "--dropout-variants",
        type=int,
        default=1,
        help="Number of coarse dropout variants created per (shifted) training sample.",
    )
    parser.add_argument(
        "--dropout-rects",
        type=int,
        default=2,
        help="Number of coarse dropout rectangles used in each dropout variant.",
    )
    parser.add_argument(
        "--dropout-size-range",
        type=float,
        nargs=2,
        default=(0.05, 0.2),
        metavar=("MIN", "MAX"),
        help="Relative width/height range for coarse dropout rectangles.",
    )
    parser.add_argument(
        "--xy-mask-variants",
        type=int,
        default=1,
        help="Number of XY masking variants created per (shifted) training sample.",
    )
    parser.add_argument(
        "--xy-mask-max-frac",
        type=float,
        default=0.1,
        help="Maximum relative thickness of horizontal/vertical masks.",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if not (0.0 < train < 1.0 and 0.0 <= val < 1.0 and 0.0 <= test < 1.0):
        raise ValueError("Train/val/test ratios must each lie between 0 and 1.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")


def read_annotations(path: Path) -> List[AnnotationRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {path}")

    records: List[AnnotationRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "soundfile",
            "channel",
            "lefttimesec",
            "righttimesec",
            "topfreqhz",
            "bottomfreqhz",
            "comment",
            "notes",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotations CSV missing columns: {', '.join(sorted(missing))}")

        for idx, row in enumerate(reader, start=1):
            try:
                record = AnnotationRecord(
                    index=idx,
                    soundfile=row["soundfile"].strip(),
                    channel=int(row["channel"]),
                    left=float(row["lefttimesec"]),
                    right=float(row["righttimesec"]),
                    top_freq=float(row["topfreqhz"]),
                    bottom_freq=float(row["bottomfreqhz"]),
                    comment=(row.get("comment") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                    folder_name=(row.get("FolderName") or "").strip(),
                    metafile_name=(row.get("MetafileName") or "").strip(),
                    annotation_file_name=(row.get("AnnotationFileName") or "").strip(),
                    annotation_name=(row.get("AnnotationName") or "").strip(),
                    audio_path=(row.get("AudioPath") or "").strip(),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row at index {idx}: {row!r}") from exc
            records.append(record)

    if not records:
        raise ValueError("Annotations CSV is empty.")
    return records


def build_audio_lookup(
    audio_root: Path, annotations: Sequence[AnnotationRecord]
) -> Dict[int, Optional[Path]]:
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root directory not found: {audio_root}")

    resolved_root = audio_root.resolve()
    lookup: Dict[int, Optional[Path]] = {}
    missing: List[Tuple[int, str, Optional[str], Optional[str]]] = []

    for record in annotations:
        candidates: List[Path] = []
        raw_audio_path = record.audio_path.strip()
        if raw_audio_path:
            audio_path_obj = Path(raw_audio_path)
            if audio_path_obj.is_absolute():
                candidates.append(audio_path_obj)
            else:
                if record.folder_name:
                    candidates.append(resolved_root / Path(record.folder_name) / audio_path_obj)
                candidates.append(resolved_root / audio_path_obj)
        if record.folder_name:
            candidates.append(resolved_root / Path(record.folder_name) / record.soundfile)
        candidates.append(resolved_root / record.soundfile)

        resolved: Optional[Path] = None
        seen: List[Path] = []
        for attempt in candidates:
            attempt_resolved = attempt.resolve()
            if attempt_resolved in seen:
                continue
            seen.append(attempt_resolved)
            if attempt_resolved.exists():
                resolved = attempt_resolved
                break

        if resolved is not None:
            lookup[record.index] = resolved
        else:
            lookup[record.index] = None
            missing.append(
                (
                    record.index,
                    record.soundfile,
                    record.folder_name or None,
                    raw_audio_path or None,
                )
            )

    if not missing:
        return lookup

    grouped: Dict[str, List[Tuple[int, str, Optional[str], Optional[str]]]] = {}
    for entry in missing:
        _, soundfile, _, audio_hint = entry
        base_name = Path(audio_hint).name if audio_hint else Path(soundfile).name
        grouped.setdefault(base_name, []).append(entry)

    for base_name, items in grouped.items():
        matches = [path.resolve() for path in resolved_root.rglob(base_name)]
        if not matches:
            for index, _, _, _ in items:
                lookup[index] = None
            continue

        for index, original, folder, audio_hint in items:
            candidates = matches
            hint_parts: List[str] = []
            if folder:
                hint_parts.extend(Path(folder).parts)
            if audio_hint and not Path(audio_hint).is_absolute():
                hint_parts.extend(Path(audio_hint).parent.parts)

            if hint_parts:
                filtered = [
                    path for path in candidates if all(part in path.parts for part in hint_parts if part)
                ]
                if len(filtered) == 1:
                    lookup[index] = filtered[0]
                    continue
                if filtered:
                    candidates = filtered

            if len(candidates) == 1:
                lookup[index] = candidates[0]
                continue

            exact = [path for path in candidates if str(path).endswith(original)]
            if len(exact) == 1:
                lookup[index] = exact[0]
            else:
                chosen = candidates[0]
                lookup[index] = chosen
                print(
                    f"Warning: multiple audio files found for {original!r}; using {chosen}"
                )

    return lookup


def compute_statistics(records: Sequence[AnnotationRecord], min_slice_seconds: float) -> DatasetStats:
    durations = [rec.duration for rec in records if rec.duration > 0.0]
    if not durations:
        raise ValueError("No positive-duration annotations found.")

    mean_duration = float(np.mean(durations))
    std_duration = float(np.std(durations))
    derived_slice = mean_duration + 4.0 * std_duration
    rounded = max(min_slice_seconds, float(max(1, int(round(derived_slice)))))
    return DatasetStats(
        count=len(durations),
        mean_duration=mean_duration,
        std_duration=std_duration,
        slice_duration=rounded,
    )


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    for sub in [
        path / "images" / "train",
        path / "images" / "val",
        path / "images" / "test",
        path / "labels" / "train",
        path / "labels" / "val",
        path / "labels" / "test",
        path / "images-annotated" / "train",
        path / "images-annotated" / "val",
        path / "images-annotated" / "test",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def trim_white_edges(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    if image.ndim != 3:
        return image
    mask = np.any(image < threshold, axis=2)
    non_empty_rows = np.where(mask.any(axis=1))[0]
    non_empty_cols = np.where(mask.any(axis=0))[0]
    if non_empty_rows.size == 0 or non_empty_cols.size == 0:
        return image
    top = int(non_empty_rows[0])
    bottom = int(non_empty_rows[-1]) + 1
    left = int(non_empty_cols[0])
    right = int(non_empty_cols[-1]) + 1
    return image[top:bottom, left:right, :]


def assign_splits(
    annotations: Sequence[AnnotationRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[AnnotationRecord]]:
    rng = random.Random(seed)
    shuffled = list(annotations)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        val_count = max(0, total - train_count)
    splits = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
    }
    splits["test"] = shuffled[train_count + val_count :]
    return splits


def info_for_audio(path: Path, cache: Dict[Path, Tuple[int, float]]) -> Tuple[int, float]:
    if path in cache:
        return cache[path]
    info = sf.info(str(path))
    samplerate = info.samplerate
    duration = info.frames / float(info.samplerate)
    cache[path] = (samplerate, duration)
    return samplerate, duration


def compute_window_start(
    annotation: AnnotationRecord,
    slice_seconds: float,
    audio_duration: float,
    shift_seconds: float,
) -> Optional[float]:
    min_start = max(0.0, annotation.right - slice_seconds)
    max_start = min(annotation.left, audio_duration - slice_seconds)
    if min_start > max_start:
        return None

    annotation_center = (annotation.left + annotation.right) / 2.0
    desired_center = annotation_center + shift_seconds
    tentative = desired_center - slice_seconds / 2.0
    start = max(min_start, min(max_start, tentative))
    if not (min_start - 1e-6 <= start <= max_start + 1e-6):
        return None
    return max(0.0, min(start, audio_duration - slice_seconds))


def load_audio_window(
    path: Path,
    samplerate: int,
    start: float,
    duration: float,
) -> np.ndarray:
    start_frame = max(int(math.floor(start * samplerate)), 0)
    frame_count = max(int(math.ceil(duration * samplerate)), 1)
    with sf.SoundFile(path) as wave_file:
        if start_frame >= len(wave_file):
            return np.zeros(1, dtype="float32")
        wave_file.seek(start_frame)
        samples = wave_file.read(frame_count, dtype="float32", always_2d=True)
    if samples.size == 0:
        return np.zeros(1, dtype="float32")
    mono = samples.mean(axis=1)
    if len(mono) < frame_count:
        padding = np.zeros(frame_count - len(mono), dtype=mono.dtype)
        mono = np.concatenate([mono, padding])
    return mono


def to_yolo_box(
    annotation: AnnotationRecord,
    window_start: float,
    slice_seconds: float,
    freq_max: float,
    class_id: int,
) -> YoloBox:
    left = annotation.left - window_start
    right = annotation.right - window_start
    width = max(right - left, 0.0)
    time_center = left + width / 2.0

    bottom = min(annotation.bottom_freq, annotation.top_freq)
    top = max(annotation.bottom_freq, annotation.top_freq)
    freq_height = max(top - bottom, 0.0)
    freq_center = bottom + freq_height / 2.0

    x_center = np.clip(time_center / slice_seconds, 0.0, 1.0)
    # Spectrogram renders keep 0 Hz at the bottom while YOLO assumes (0, 0) at the top-left,
    # so flip the vertical axis when normalising.
    y_center = np.clip(1.0 - (freq_center / freq_max), 0.0, 1.0)
    width_norm = np.clip(width / slice_seconds, 0.0, 1.0)
    height_norm = np.clip(freq_height / freq_max, 0.0, 1.0)

    return YoloBox(class_id, float(x_center), float(y_center), float(width_norm), float(height_norm))


def compute_center_bounds(
    annotation: AnnotationRecord,
    slice_seconds: float,
    audio_duration: float,
) -> Optional[Tuple[float, float]]:
    min_start = max(0.0, annotation.right - slice_seconds)
    max_start = min(annotation.left, audio_duration - slice_seconds)
    if min_start > max_start:
        return None
    center_min = min_start + slice_seconds / 2.0
    center_max = max_start + slice_seconds / 2.0
    return center_min, center_max


def generate_shift_offsets(
    annotation: AnnotationRecord,
    center_bounds: Tuple[float, float],
    step_seconds: float,
    min_shift_seconds: float,
    variants: int,
    rng: np.random.Generator,
) -> List[float]:
    annotation_center = (annotation.left + annotation.right) / 2.0
    center_min, center_max = center_bounds
    if center_max - center_min <= 1e-6 or variants <= 0:
        return []

    offset_min = center_min - annotation_center
    offset_max = center_max - annotation_center
    span = offset_max - offset_min
    step = step_seconds if step_seconds > 0 else max(span / max(variants, 1), 1e-3)

    grid = np.arange(center_min, center_max + step * 0.5, step)
    if grid.size == 0 or grid[0] > center_min + 1e-6:
        grid = np.insert(grid, 0, center_min)
    if grid[-1] < center_max - 1e-6:
        grid = np.append(grid, center_max)

    offsets = grid - annotation_center

    eps = 1e-6
    left_offsets = offsets[offsets < -eps]
    right_offsets = offsets[offsets > eps]

    if offset_min < -eps and left_offsets.size == 0:
        left_offsets = np.array([offset_min])
    if offset_max > eps and right_offsets.size == 0:
        right_offsets = np.array([offset_max])

    min_shift = max(0.0, min_shift_seconds)
    if min_shift > 0:
        left_filtered = left_offsets[np.abs(left_offsets) >= min_shift - eps]
        right_filtered = right_offsets[np.abs(right_offsets) >= min_shift - eps]
        if left_offsets.size and left_filtered.size == 0:
            left_filtered = left_offsets
        if right_offsets.size and right_filtered.size == 0:
            right_filtered = right_offsets
        left_offsets = left_filtered
        right_offsets = right_filtered

    pools: List[Tuple[str, List[float]]] = []
    if left_offsets.size:
        pools.append(("neg", left_offsets.tolist()))
    if right_offsets.size:
        pools.append(("pos", right_offsets.tolist()))

    if not pools:
        return []

    offsets_out: List[float] = []
    for idx in range(variants):
        pool = pools[idx % len(pools)][1]
        if not pool:
            continue
        choice_idx = rng.integers(0, len(pool))
        offsets_out.append(float(pool[choice_idx]))
    return offsets_out


def collect_boxes_for_window(
    anchor: AnnotationRecord,
    window_start: float,
    slice_seconds: float,
    freq_max: float,
    class_id: int,
    audio_path: Path,
    grouped_annotations: Dict[Path, List[AnnotationRecord]],
) -> List[YoloBox]:
    candidates = grouped_annotations.get(audio_path)
    if not candidates:
        return []
    window_end = window_start + slice_seconds
    tolerance = 1e-6
    boxes: List[YoloBox] = []
    for candidate in candidates:
        if candidate.channel != anchor.channel:
            continue
        if candidate.duration <= 0.0:
            continue
        if candidate.left < window_start - tolerance or candidate.right > window_end + tolerance:
            continue
        freq_bottom = min(candidate.bottom_freq, candidate.top_freq)
        freq_top = max(candidate.bottom_freq, candidate.top_freq)
        if freq_bottom < -tolerance or freq_top > freq_max + tolerance:
            continue
        boxes.append(to_yolo_box(candidate, window_start, slice_seconds, freq_max, class_id))
    return boxes


def render_spectrogram(
    samples: np.ndarray,
    samplerate: int,
    slice_seconds: float,
    freq_max: float,
    figure_size: Tuple[float, float],
    dpi: int,
) -> np.ndarray:
    freqs, times, spec = spectrogram(
        samples,
        fs=samplerate,
        nperseg=1024,
        noverlap=768,
        scaling="spectrum",
    )
    spec_db = 10.0 * np.log10(spec + 1e-12)

    vmin = np.percentile(spec_db, 5)
    vmax = np.percentile(spec_db, 99)

    fig = plt.figure(figsize=figure_size, dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    mesh = ax.pcolormesh(times, freqs, spec_db, shading="auto", cmap="magma")
    mesh.set_clim(vmin, vmax)
    ax.set_xlim(0.0, slice_seconds)
    ax.set_ylim(0.0, freq_max)
    ax.set_axis_off()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bbox_pixels = ax.get_window_extent().transformed(fig.dpi_scale_trans)
    x0, y0, width, height = bbox_pixels.bounds
    x0 = max(0, int(math.floor(x0)))
    y0 = max(0, int(math.floor(y0)))
    x1 = min(rgba.shape[1], int(math.ceil(x0 + width)))
    y1 = min(rgba.shape[0], int(math.ceil(y0 + height)))
    cropped = rgba[y0:y1, x0:x1, :3]
    image = trim_white_edges(np.ascontiguousarray(cropped))
    plt.close(fig)
    return image


def draw_annotations(image: np.ndarray, boxes: Sequence[YoloBox]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(annotated)
    ax.set_axis_off()
    for box in boxes:
        box_w = box.width * width
        box_h = box.height * height
        center_x = box.x_center * width
        center_y = box.y_center * height
        rect = Rectangle(
            (center_x - box_w / 2.0, center_y - box_h / 2.0),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="cyan",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            center_x - box_w / 2.0,
            max(center_y - box_h / 2.0 - 5, 0),
            str(box.class_id),
            color="cyan",
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="left",
        )
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    bbox_pixels = ax.get_window_extent().transformed(fig.dpi_scale_trans)
    x0, y0, width, height = bbox_pixels.bounds
    x0 = max(0, int(math.floor(x0)))
    y0 = max(0, int(math.floor(y0)))
    x1 = min(rgba.shape[1], int(math.ceil(x0 + width)))
    y1 = min(rgba.shape[0], int(math.ceil(y0 + height)))
    overlay = trim_white_edges(np.ascontiguousarray(rgba[y0:y1, x0:x1, :3]))
    plt.close(fig)
    return overlay


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image)


def write_labels(path: Path, boxes: Sequence[YoloBox]) -> None:
    if not boxes:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for box in boxes:
            handle.write(
                f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}\n"
            )


def compute_fill_value(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        med = float(np.median(image))
        return np.array(med, dtype=image.dtype)
    med = np.median(image.reshape(-1, image.shape[-1]), axis=0)
    return np.array(med, dtype=image.dtype)


def apply_coarse_dropout(
    image: np.ndarray,
    rng: np.random.Generator,
    rects: int,
    size_range: Tuple[float, float],
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    min_frac, max_frac = size_range
    fill_value = compute_fill_value(result)
    for _ in range(rects):
        rect_w = max(1, int(round(rng.uniform(min_frac, max_frac) * width)))
        rect_h = max(1, int(round(rng.uniform(min_frac, max_frac) * height)))
        if rect_w >= width or rect_h >= height:
            continue
        x0 = rng.integers(0, width - rect_w)
        y0 = rng.integers(0, height - rect_h)
        result[y0 : y0 + rect_h, x0 : x0 + rect_w, ...] = fill_value
    return result


def apply_xy_mask(
    image: np.ndarray,
    rng: np.random.Generator,
    max_frac: float,
) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    fill_value = compute_fill_value(result)

    vertical = max(1, int(round(rng.uniform(0.0, max_frac) * width)))
    horizontal = max(1, int(round(rng.uniform(0.0, max_frac) * height)))

    if vertical < width:
        x0 = rng.integers(0, width - vertical)
        result[:, x0 : x0 + vertical, ...] = fill_value

    if horizontal < height:
        y0 = rng.integers(0, height - horizontal)
        result[y0 : y0 + horizontal, :, ...] = fill_value

    return result


def flip_horizontal(image: np.ndarray, boxes: Sequence[YoloBox]) -> Tuple[np.ndarray, List[YoloBox]]:
    flipped = np.flip(image, axis=1)
    updated: List[YoloBox] = [
        YoloBox(box.class_id, 1.0 - box.x_center, box.y_center, box.width, box.height) for box in boxes
    ]
    return flipped, updated


def generate_variant_name(base_stem: str, suffix: str) -> str:
    return f"{base_stem}{suffix}"


def save_sample(
    stem: str,
    split: str,
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    output_dir: Path,
) -> None:
    image_path = output_dir / "images" / split / f"{stem}.png"
    label_path = output_dir / "labels" / split / f"{stem}.txt"
    annotated_path = output_dir / "images-annotated" / split / f"{stem}.png"

    write_image(image_path, image)
    write_labels(label_path, boxes)
    annotated = draw_annotations(image, boxes) if boxes else image
    write_image(annotated_path, annotated)


def build_dataset(args: argparse.Namespace) -> None:
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    annotations = read_annotations(args.annotations_csv)
    stats = compute_statistics(annotations, args.min_slice_seconds)
    slice_seconds = args.slice_seconds if args.slice_seconds is not None else stats.slice_duration

    print(
        f"Annotation count: {stats.count}, "
        f"mean duration: {stats.mean_duration:.3f}s, "
        f"std: {stats.std_duration:.3f}s, "
        f"slice duration: {slice_seconds:.3f}s"
    )

    ensure_output_dir(args.output_dir, args.overwrite)
    splits = assign_splits(annotations, args.train_ratio, args.val_ratio, args.seed)
    audio_cache: Dict[Path, Tuple[int, float]] = {}
    rng = np.random.default_rng(args.seed)
    audio_lookup = build_audio_lookup(args.audio_root, annotations)
    grouped_annotations: Dict[Path, List[AnnotationRecord]] = {}
    for record in annotations:
        resolved = audio_lookup.get(record.index)
        if resolved is not None:
            grouped_annotations.setdefault(resolved, []).append(record)

    summary_counts = {"train": 0, "val": 0, "test": 0}
    augmentation_counts = {"shift": 0, "flip": 0, "dropout": 0, "xy_mask": 0}

    for split_name, split_annotations in splits.items():
        for annotation in split_annotations:
            audio_path = audio_lookup.get(annotation.index)
            if not audio_path or not audio_path.exists():
                if annotation.audio_path:
                    candidate = Path(annotation.audio_path)
                    if candidate.is_absolute():
                        expected = candidate
                    elif annotation.folder_name:
                        expected = (args.audio_root / Path(annotation.folder_name) / candidate).resolve()
                    else:
                        expected = (args.audio_root / candidate).resolve()
                elif annotation.folder_name:
                    expected = (args.audio_root / Path(annotation.folder_name) / annotation.soundfile).resolve()
                else:
                    expected = (args.audio_root / annotation.soundfile).resolve()
                print(
                    f"Warning: audio file missing for annotation #{annotation.index}: {expected}"
                )
                continue

            samplerate, audio_duration = info_for_audio(audio_path, audio_cache)
            if audio_duration < slice_seconds or annotation.duration > slice_seconds:
                continue

            base_name = f"{Path(annotation.soundfile).stem}_ann_{annotation.index:05d}"
            annotation_center = (annotation.left + annotation.right) / 2.0

            base_start = compute_window_start(annotation, slice_seconds, audio_duration, shift_seconds=0.0)
            if base_start is None:
                continue
            samples = load_audio_window(audio_path, samplerate, base_start, slice_seconds)
            if samples.size == 0:
                continue
            freq_max = samplerate / 2.0
            base_image = render_spectrogram(
                samples,
                samplerate,
                slice_seconds,
                freq_max,
                (args.figure_width, args.figure_height),
                args.dpi,
            )
            base_boxes = collect_boxes_for_window(
                annotation,
                base_start,
                slice_seconds,
                freq_max,
                args.class_id,
                audio_path,
                grouped_annotations,
            )
            if not base_boxes:
                continue
            save_sample(base_name, split_name, base_image, base_boxes, args.output_dir)
            summary_counts[split_name] += 1

            if split_name != "train":
                continue

            variant_bank: List[Tuple[str, np.ndarray, List[YoloBox]]] = [("", base_image, base_boxes)]
            center_bounds = compute_center_bounds(annotation, slice_seconds, audio_duration)
            offsets = []
            if center_bounds is not None and args.shift_max_seconds > 0.0:
                limited_bounds = (
                    max(center_bounds[0], annotation_center - args.shift_max_seconds),
                    min(center_bounds[1], annotation_center + args.shift_max_seconds),
                )
                if limited_bounds[1] - limited_bounds[0] > 1e-6:
                    offsets = generate_shift_offsets(
                        annotation,
                        limited_bounds,
                        args.shift_step_seconds,
                        args.shift_min_seconds,
                        args.shift_variants,
                        rng,
                    )

            for idx, offset in enumerate(offsets, start=1):
                start_center = (annotation.left + annotation.right) / 2.0 + offset
                start = max(0.0, min(start_center - slice_seconds / 2.0, audio_duration - slice_seconds))
                variant_samples = load_audio_window(audio_path, samplerate, start, slice_seconds)
                if variant_samples.size == 0:
                    continue
                variant_image = render_spectrogram(
                    variant_samples,
                    samplerate,
                    slice_seconds,
                    freq_max,
                    (args.figure_width, args.figure_height),
                    args.dpi,
                )
                variant_boxes = collect_boxes_for_window(
                    annotation,
                    start,
                    slice_seconds,
                    freq_max,
                    args.class_id,
                    audio_path,
                    grouped_annotations,
                )
                if not variant_boxes:
                    continue
                slice_center = start + slice_seconds / 2.0
                annotation_center = (annotation.left + annotation.right) / 2.0
                actual_shift = slice_center - annotation_center
                if abs(actual_shift) < max(0.0, args.shift_min_seconds) - 1e-6:
                    continue
                shift_suffix = "__shift{}".format(idx) if actual_shift >= 0 else "__shiftneg{}".format(idx)
                variant_name = generate_variant_name(base_name, shift_suffix)
                save_sample(variant_name, "train", variant_image, variant_boxes, args.output_dir)
                summary_counts["train"] += 1
                augmentation_counts["shift"] += 1
                variant_bank.append((shift_suffix, variant_image, variant_boxes))

            for suffix, variant_image, variant_boxes in variant_bank:
                variant_stem = generate_variant_name(base_name, suffix)

                if args.horizontal_flip:
                    flipped_image, flipped_boxes = flip_horizontal(variant_image, variant_boxes)
                    flip_name = generate_variant_name(variant_stem, "__flip")
                    save_sample(flip_name, "train", flipped_image, flipped_boxes, args.output_dir)
                    summary_counts["train"] += 1
                    augmentation_counts["flip"] += 1

                for drop_idx in range(args.dropout_variants):
                    dropout_image = apply_coarse_dropout(
                        variant_image,
                        rng,
                        args.dropout_rects,
                        tuple(args.dropout_size_range),
                    )
                    drop_name = generate_variant_name(variant_stem, f"__drop{drop_idx+1}")
                    save_sample(drop_name, "train", dropout_image, variant_boxes, args.output_dir)
                    summary_counts["train"] += 1
                    augmentation_counts["dropout"] += 1

                for mask_idx in range(args.xy_mask_variants):
                    mask_image = apply_xy_mask(variant_image, rng, args.xy_mask_max_frac)
                    mask_name = generate_variant_name(variant_stem, f"__mask{mask_idx+1}")
                    save_sample(mask_name, "train", mask_image, variant_boxes, args.output_dir)
                    summary_counts["train"] += 1
                    augmentation_counts["xy_mask"] += 1

    dataset_yaml = args.output_dir / "dataset.yaml"
    with dataset_yaml.open("w") as handle:
        handle.write(f"path: {args.output_dir.resolve()}\n")
        handle.write("train: images/train\n")
        handle.write("val: images/val\n")
        handle.write("test: images/test\n")
        handle.write("names:\n")
        handle.write(f"  {args.class_id}: {args.class_name}\n")

    stats_path = args.output_dir / "dataset_stats.json"
    with stats_path.open("w") as handle:
        json.dump(
            {
                "annotation_count": stats.count,
                "mean_duration_seconds": stats.mean_duration,
                "std_duration_seconds": stats.std_duration,
                "slice_duration_seconds": slice_seconds,
                "splits": summary_counts,
                "augmentations": augmentation_counts,
            },
            handle,
            indent=2,
        )

    print(f"Dataset generated in {args.output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
