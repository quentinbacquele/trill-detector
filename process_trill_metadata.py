#!/usr/bin/env python3
"""Process batch-annotated bird trill recordings into per-file metadata."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set, Mapping

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, spectrogram

# Ensure matplotlib can initialise even when the default cache dir is read-only.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="mplcache_"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


@dataclasses.dataclass
class Segment:
    name: str
    path: Optional[Path]
    start: float
    end: float
    duration: float
    samplerate: Optional[int]
    channels: Optional[int]
    is_audio: bool


@dataclasses.dataclass
class Annotation:
    sequence_index: int
    source_name: str
    source_path: Path
    channel: int
    left_cumulative: float
    right_cumulative: float
    left_relative: float
    right_relative: float
    top_freq: float
    bottom_freq: float
    comment: str
    notes: str
    samplerate: int
    channels: int
    segment_duration: float
    metafile: Path
    global_index: int = 0
    folder_name: str = ""
    metafile_name: str = ""
    annotation_file_name: str = ""
    annotation_name: str = ""
    audio_path: str = ""

    @property
    def duration(self) -> float:
        return self.right_relative - self.left_relative


def parse_spacer_override(value: str) -> Tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected format NAME=SECONDS for spacer override.")
    name, _, seconds = value.partition("=")
    key = name.strip()
    if not key:
        raise argparse.ArgumentTypeError("Metafile name in spacer override cannot be empty.")
    try:
        duration = float(seconds)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid spacer duration '{seconds}' for {key!r}.") from exc
    if duration <= 0.0:
        raise argparse.ArgumentTypeError("Spacer override duration must be positive.")
    return key.lower(), duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-map batch annotations to individual files, "
        "generate spectrograms, and optionally extract clips across one or many experiments."
    )
    parser.add_argument(
        "--meta",
        type=Path,
        nargs="*",
        help="Specific metafile(s) to process. If omitted, all '*metafile.txt' files under --root are used.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory to search for metafiles when --meta is not supplied.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Path to the annotation log. Defaults to the file referenced in the metafile.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory containing the source WAV files (defaults to the metafile's folder).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("processed_annotations.csv"),
        help="Destination CSV path for the per-file annotations.",
    )
    parser.add_argument(
        "--spectrogram-dir",
        type=Path,
        default=Path("spectrograms"),
        help="Directory to write spectrogram images. Use --skip-spectrograms to disable.",
    )
    parser.add_argument(
        "--skip-spectrograms",
        action="store_true",
        help="Disable spectrogram generation.",
    )
    parser.add_argument(
        "--spectrogram-padding",
        type=float,
        default=1.0,
        help="Seconds of padding before and after each annotation window for spectrograms.",
    )
    parser.add_argument(
        "--extract-clips",
        action="store_true",
        help="Enable writing per-annotation WAV clips trimmed in time and frequency.",
    )
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=Path("clips"),
        help="Directory to write extracted clips when --extract-clips is used.",
    )
    parser.add_argument(
        "--clip-padding",
        type=float,
        default=0.0,
        help="Seconds of padding to include before and after each clip.",
    )
    parser.add_argument(
        "--bandpass-margin",
        type=float,
        default=50.0,
        help="Extra Hertz to widen the bandpass filter on each side when extracting clips.",
    )
    parser.add_argument(
        "--default-spacer-seconds",
        type=float,
        default=55.0,
        help="Effective duration (seconds) to subtract for spacer entries.",
    )
    parser.add_argument(
        "--spacer-duration-override",
        action="append",
        default=[],
        type=parse_spacer_override,
        metavar="NAME=SECONDS",
        help="Override spacer duration for specific metafile names. Repeat to supply multiple overrides.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Seconds of tolerance when matching annotations to audio segments.",
    )
    parser.add_argument(
        "--filter-twitter",
        action="store_true",
        help="Only keep annotations whose comment contains an AnnotationName from the Twitter Vocal Signatures sheet.",
    )
    parser.add_argument(
        "--twitter-signatures",
        type=Path,
        default=Path("Twitter Vocal Signatures.csv"),
        help="CSV file listing AnnotationName entries used by --filter-twitter.",
    )
    return parser.parse_args()


def resolve_spacer_override(meta: Path, overrides: Dict[str, float]) -> Optional[float]:
    if not overrides:
        return None

    def trim_suffix_insensitive(value: str, suffix: str) -> Optional[str]:
        if suffix and value.lower().endswith(suffix.lower()):
            return value[: -len(suffix)]
        return None

    candidates: Set[str] = set()
    raw_values = {meta.name, meta.stem}
    if meta.suffix:
        raw_values.add(meta.name[: -len(meta.suffix)])

    for raw in raw_values:
        if not raw:
            continue
        stripped = raw.strip()
        if not stripped:
            continue
        candidates.add(stripped.lower())

        trimmed_txt = trim_suffix_insensitive(stripped, ".txt")
        if trimmed_txt:
            candidates.add(trimmed_txt.strip().lower())

        trimmed_space = trim_suffix_insensitive(stripped, " metafile")
        if trimmed_space:
            candidates.add(trimmed_space.strip().lower())

        trimmed_under = trim_suffix_insensitive(stripped, "_metafile")
        if trimmed_under:
            candidates.add(trimmed_under.strip().lower())

    for candidate in candidates:
        if candidate in overrides:
            return overrides[candidate]
    return None


def read_metafile(
    path: Path, audio_dir: Path, default_spacer: float
) -> Tuple[List[Segment], Path]:
    if not path.exists():
        raise FileNotFoundError(f"Metafile not found: {path}")

    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Metafile is empty: {path}")

    log_ref_line = lines.pop(0)
    if not log_ref_line.lower().startswith("annotationfile="):
        raise ValueError(
            f"First line of metafile must reference the annotation log, got: {log_ref_line}"
        )

    log_path = Path(log_ref_line.split("=", 1)[1].strip())
    cursor = 0.0
    segments: List[Segment] = []

    for entry in lines:
        entry_name = entry.strip()
        entry_lower = entry_name.lower()
        entry_path = (audio_dir / entry_name) if entry_lower.endswith(".wav") else None
        is_spacer = "spacer" in entry_lower
        is_audio = entry_path is not None and entry_path.exists() and not is_spacer
        samplerate: Optional[int] = None
        channels: Optional[int] = None

        if is_audio:
            info = sf.info(entry_path)
            duration = info.frames / float(info.samplerate)
            samplerate = info.samplerate
            channels = info.channels
        else:
            duration = infer_spacer_duration(entry_name, default_spacer)

        segment = Segment(
            name=entry_name,
            path=entry_path if is_audio else None,
            start=cursor,
            end=cursor + duration,
            duration=duration,
            samplerate=samplerate,
            channels=channels,
            is_audio=is_audio,
        )
        segments.append(segment)
        cursor += duration

    return segments, log_path


def infer_spacer_duration(name: str, default: float) -> float:
    return default


def load_annotations(log_path: Path) -> List[Tuple[str, int, float, float, float, float, str, str]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Annotation log not found: {log_path}")

    annotations: List[Tuple[str, int, float, float, float, float, str, str]] = []
    with log_path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if not header:
            raise ValueError(f"Annotation log has no header: {log_path}")

        for row in reader:
            if not row or row[0].strip().lower() not in {"an:", "ft:"}:
                continue
            if len(row) < 8:
                raise ValueError(f"Unexpected row format in {log_path}: {row!r}")

            source = row[1].strip()
            channel = int(row[2])
            left = float(row[3])
            right = float(row[4])
            top = float(row[5])
            bottom = float(row[6])
            comment = row[7].strip() if len(row) > 7 else ""
            notes = row[8].strip() if len(row) > 8 else ""
            annotations.append((source, channel, left, right, top, bottom, comment, notes))

    return annotations


def map_annotations_to_segments(
    annotations: Sequence[Tuple[str, int, float, float, float, float, str, str]],
    segments: Sequence[Segment],
    tolerance: float,
    metafile: Path,
) -> List[Annotation]:
    mapped: List[Annotation] = []

    for index, (source, channel, left, right, top, bottom, comment, notes) in enumerate(
        annotations, start=1
    ):
        remaining_left = left
        remaining_right = right
        target_segment: Optional[Segment] = None
        left_rel = 0.0
        right_rel = 0.0

        for seg in segments:
            duration = seg.duration

            if seg.is_audio:
                if remaining_left <= duration + tolerance:
                    target_segment = seg
                    left_rel = max(remaining_left, 0.0)
                    right_rel = min(remaining_right, duration)
                    break
                remaining_left -= duration
                remaining_right -= duration
            else:
                remaining_left = max(remaining_left - duration, 0.0)
                remaining_right = max(remaining_right - duration, 0.0)

        if target_segment is None:
            raise ValueError(
                f"Could not match annotation #{index} ({left}-{right}s) to an audio segment."
            )
        if target_segment.path is None or target_segment.samplerate is None:
            raise ValueError(
                f"Annotation #{index} maps to non-audio segment '{target_segment.name}'."
            )

        right_rel = min(right_rel, target_segment.duration)

        if right_rel - left_rel > target_segment.duration + tolerance:
            raise ValueError(
                f"Annotation #{index} exceeds duration of file {target_segment.name}."
            )

        mapped.append(
            Annotation(
                sequence_index=index,
                source_name=target_segment.name,
                source_path=target_segment.path,
                channel=channel,
                left_cumulative=left,
                right_cumulative=right,
                left_relative=left_rel,
                right_relative=right_rel,
                top_freq=top,
                bottom_freq=bottom,
                comment=comment,
                notes=notes,
                samplerate=target_segment.samplerate,
                channels=target_segment.channels or 1,
                segment_duration=target_segment.duration,
                metafile=metafile,
            )
        )

    return mapped


def discover_metafiles(meta_args: Optional[Sequence[Path]], root: Path) -> List[Path]:
    if meta_args:
        return sorted(Path(m).resolve() for m in meta_args)
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    return sorted(p.resolve() for p in root.rglob("*metafile.txt"))


def load_twitter_signatures(
    path: Path,
) -> Tuple[Dict[Tuple[str, str], Dict[str, List[str]]], Dict[Tuple[str, str], str]]:
    if not path.exists():
        raise FileNotFoundError(f"Twitter Vocal Signatures CSV not found: {path}")

    annotation_tokens: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    log_targets: Dict[Tuple[str, str], str] = {}
    saw_annotation_name = False
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "AnnotationName" not in reader.fieldnames:
            raise ValueError("Twitter Vocal Signatures CSV must contain 'AnnotationName' column.")
        if "FolderName" not in reader.fieldnames:
            raise ValueError("Twitter Vocal Signatures CSV must contain 'FolderName' column.")
        if "MetafileName" not in reader.fieldnames:
            raise ValueError("Twitter Vocal Signatures CSV must contain 'MetafileName' column.")
        if "AnnotationFileName" not in reader.fieldnames:
            raise ValueError("Twitter Vocal Signatures CSV must contain 'AnnotationFileName' column.")
        for row in reader:
            annotation_name = (row.get("AnnotationName") or "").strip()
            folder = (row.get("FolderName") or "").strip()
            meta = (row.get("MetafileName") or "").strip()
            log = (row.get("AnnotationFileName") or "").strip()

            if folder and meta and log:
                key = (folder, meta)
                existing_log = log_targets.get(key)
                if existing_log is not None and existing_log != log:
                    raise ValueError(
                        f"Conflicting AnnotationFileName entries for folder {folder!r} "
                        f"and metafile {meta!r} in Twitter Vocal Signatures CSV."
                    )
                log_targets[key] = log

                if annotation_name:
                    saw_annotation_name = True
                    token_map = annotation_tokens.setdefault(key, {})
                    for token in build_annotation_match_tokens(annotation_name):
                        token_map.setdefault(token, [])
                        if annotation_name not in token_map[token]:
                            token_map[token].append(annotation_name)
            elif annotation_name:
                saw_annotation_name = True
    if not saw_annotation_name:
        raise ValueError("Twitter Vocal Signatures CSV does not contain any AnnotationName entries.")
    if not log_targets:
        raise ValueError(
            "Twitter Vocal Signatures CSV does not contain combined Folder/Metafile/Annotation entries."
        )
    return annotation_tokens, log_targets


def build_annotation_match_tokens(annotation_name: str) -> List[str]:
    base = annotation_name.strip().lower()
    tokens: List[str] = []
    seen: Set[str] = set()

    def add_token(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            tokens.append(value)
            seen.add(value)

    add_token(base)
    normalized = base.replace("-", "_")
    add_token(normalized)
    add_token(normalized.replace("_", " "))
    if "_" in normalized:
        tail = normalized.split("_", 1)[1]
        add_token(tail)
        add_token(tail.replace("_", " "))
    return tokens


def filter_annotations_by_comment(
    annotations: Sequence[Annotation],
    token_map: Mapping[str, Sequence[str]],
) -> List[Annotation]:
    if not token_map:
        return []

    sorted_tokens = sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True)
    result: List[Annotation] = []
    for ann in annotations:
        comment = (ann.comment or "").strip()
        if not comment:
            continue
        comment_lower = comment.lower()
        comment_forms = [
            comment_lower,
            comment_lower.replace("-", "_"),
            comment_lower.replace("_", " "),
            comment_lower.replace(" ", "_"),
        ]
        matched_name: Optional[str] = None
        for token, names in sorted_tokens:
            if any(token in form for form in comment_forms):
                if names:
                    matched_name = names[0]
                break
        if matched_name:
            ann.annotation_name = matched_name
            result.append(ann)
    return result


def sanitize_component(component: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", component)
    return safe.strip("_") or "item"


def build_output_stem(path: Path, relative_to: Path) -> str:
    try:
        relative = path.resolve().relative_to(relative_to.resolve())
    except ValueError:
        relative = Path(path.name)
    parts = list(relative.parts)
    sanitized = [sanitize_component(p.rsplit(".", 1)[0]) for p in parts]
    return "__".join(sanitized)


def top_level_folder(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
        if relative.parts:
            return relative.parts[0]
    except ValueError:
        pass
    return path.parent.name


def write_annotations_csv(path: Path, annotations: Sequence[Annotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "FolderName",
                "MetafileName",
                "AnnotationFileName",
                "AnnotationName",
                "AudioPath",
                "soundfile",
                "channel",
                "lefttimesec",
                "righttimesec",
                "topfreqhz",
                "bottomfreqhz",
                "comment",
                "notes",
            ]
        )
        for ann in annotations:
            writer.writerow(
                [
                    ann.folder_name,
                    ann.metafile_name,
                    ann.annotation_file_name,
                    ann.annotation_name,
                    ann.audio_path,
                    ann.source_name,
                    ann.channel,
                    f"{ann.left_relative:.6f}",
                    f"{ann.right_relative:.6f}",
                    f"{ann.top_freq:.6f}",
                    f"{ann.bottom_freq:.6f}",
                    ann.comment,
                    ann.notes,
                ]
            )


def generate_spectrograms(
    annotations: Sequence[Annotation],
    output_dir: Path,
    padding: float,
    root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        pad_left = max(ann.left_relative - padding, 0.0)
        pad_right = min(ann.right_relative + padding, ann.segment_duration)
        pad_left = min(pad_left, ann.segment_duration)
        pad_right = min(pad_right, ann.segment_duration)
        if pad_right <= pad_left:
            print(
                f"Skipping spectrogram for annotation #{ann.sequence_index} in {ann.source_name}; "
                "annotation exceeds available audio."
            )
            continue

        data = load_audio_window(ann.source_path, ann.samplerate, pad_left, pad_right)

        freqs, spec_times, spec = spectrogram(
            data,
            fs=ann.samplerate,
            nperseg=1024,
            noverlap=768,
            scaling="spectrum",
        )

        spec_db = 10 * np.log10(spec + 1e-12)
        fig, ax = plt.subplots(figsize=(8, 4))
        mesh = ax.pcolormesh(
            spec_times + pad_left,
            freqs,
            spec_db,
            shading="gouraud",
            cmap="magma",
        )
        fig.colorbar(mesh, ax=ax, label="Power (dB)")
        ax.add_patch(
            Rectangle(
                (ann.left_relative, ann.bottom_freq),
                max(ann.duration, 1e-6),
                max(ann.top_freq - ann.bottom_freq, 1e-6),
                linewidth=1.5,
                edgecolor="cyan",
                facecolor="none",
            )
        )
        ax.set_xlim(pad_left, pad_right)
        ax.set_ylim(0, ann.samplerate / 2.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{ann.source_name} #{ann.sequence_index}: {ann.comment}")
        fig.tight_layout()

        stem = build_output_stem(ann.source_path, root)
        outfile = output_dir / f"{stem}_ann_{ann.global_index:05d}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)


def extract_audio_clips(
    annotations: Sequence[Annotation],
    output_dir: Path,
    padding: float,
    margin: float,
    root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        start = max(ann.left_relative - padding, 0.0)
        end = min(ann.right_relative + padding, ann.segment_duration)
        start = min(start, ann.segment_duration)
        end = min(end, ann.segment_duration)
        if end <= start:
            print(
                f"Skipping clip extraction for annotation #{ann.sequence_index} in {ann.source_name}; "
                "annotation exceeds available audio."
            )
            continue
        data = load_audio_window(ann.source_path, ann.samplerate, start, end, always_mono=False)

        filtered = apply_bandpass(data, ann.samplerate, ann.bottom_freq, ann.top_freq, margin)
        stem = build_output_stem(ann.source_path, root)
        outfile = output_dir / f"{stem}_ann_{ann.global_index:05d}.wav"
        sf.write(outfile, filtered, ann.samplerate)


def load_audio_window(
    path: Path,
    samplerate: int,
    start: float,
    end: float,
    always_mono: bool = True,
) -> np.ndarray:
    with sf.SoundFile(path) as wave_file:
        start_frame = int(math.floor(start * samplerate))
        end_frame = int(math.ceil(end * samplerate))
        frame_count = max(end_frame - start_frame, 1)
        wave_file.seek(start_frame)
        samples = wave_file.read(frame_count, dtype="float32", always_2d=True)

    if always_mono:
        samples = samples.mean(axis=1)
    else:
        samples = samples if samples.shape[1] > 1 else samples[:, 0:1]

    return samples


def apply_bandpass(
    data: np.ndarray,
    samplerate: int,
    low_freq: float,
    high_freq: float,
    margin: float,
) -> np.ndarray:
    if data.ndim == 1:
        working = data
    else:
        working = data.copy()

    nyquist = samplerate / 2.0
    low = max(low_freq - margin, 0.0)
    high = min(high_freq + margin, nyquist)

    if low <= 0.0 and high >= nyquist:
        return working

    if high <= low + 1.0:
        return working

    if low <= 0.0:
        sos = butter(4, high / nyquist, btype="lowpass", output="sos")
    elif high >= nyquist:
        sos = butter(4, low / nyquist, btype="highpass", output="sos")
    else:
        sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")

    return sosfiltfilt(sos, working, axis=0)


def main() -> None:
    args = parse_args()
    spacer_overrides: Dict[str, float] = dict(args.spacer_duration_override)

    metafiles = discover_metafiles(args.meta, args.root)
    if not metafiles:
        raise FileNotFoundError("No metafile.txt files found to process.")

    if args.log is not None and len(metafiles) != 1:
        raise ValueError("--log can only be used when processing a single metafile.")

    signature_tokens: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
    signature_targets: Dict[Tuple[str, str], str] = {}
    if args.filter_twitter:
        signature_tokens, signature_targets = load_twitter_signatures(args.twitter_signatures)
        allowed_folders = {folder for folder, _ in signature_targets.keys()}

        metafiles = [
            meta
            for meta in metafiles
            if top_level_folder(meta, args.root) in allowed_folders
        ]
        if not metafiles:
            raise FileNotFoundError(
                "No metafiles found within folders referenced by Twitter Vocal Signatures CSV."
            )

    all_annotations: List[Annotation] = []
    for meta in metafiles:
        override_value = resolve_spacer_override(meta, spacer_overrides)
        effective_spacer = override_value if override_value is not None else args.default_spacer_seconds

        target_key: Optional[Tuple[str, str]] = None

        if args.filter_twitter:
            folder_found = top_level_folder(meta, args.root)
            key_candidates = [
                (folder_found, meta.stem),
                (folder_found, meta.name),
                (folder_found, meta.stem + ".txt"),
            ]
            target_log_name: Optional[str] = None
            for key in key_candidates:
                if key in signature_targets:
                    target_key = key
                    target_log_name = signature_targets[key]
                    break
            if target_key is None or target_log_name is None:
                continue

        audio_dir = args.audio_dir if args.audio_dir is not None else meta.parent
        segments, inferred_log = read_metafile(meta, audio_dir, effective_spacer)
        log_path = args.log if args.log is not None else meta.parent / inferred_log

        if args.filter_twitter:
            log_candidates = []
            if target_log_name:
                log_candidates.extend(
                    [
                        target_log_name,
                        f"{target_log_name}.log",
                        f"{target_log_name}.txt",
                    ]
                )
            for candidate in log_candidates:
                candidate_path = meta.parent / candidate
                if candidate_path.exists():
                    log_path = candidate_path
                    break
            else:
                print(
                    f"Warning: annotation log '{target_log_name}' not found for {meta}. Skipping."
                )
                continue

        annotations_raw = load_annotations(log_path)
        mapped = map_annotations_to_segments(annotations_raw, segments, args.tolerance, meta)

        if args.filter_twitter:
            token_map = signature_tokens.get(target_key, {}) if target_key is not None else {}
            mapped = filter_annotations_by_comment(mapped, token_map)
            if not mapped:
                continue

        folder_path = meta.parent
        try:
            relative_folder = folder_path.resolve().relative_to(args.root.resolve())
            folder_name = str(relative_folder) if str(relative_folder) != "." else folder_path.name
        except ValueError:
            folder_name = folder_path.name
        metafile_name = meta.stem
        annotation_file_name = Path(log_path).stem
        for ann in mapped:
            ann.folder_name = folder_name
            ann.metafile_name = metafile_name
            ann.annotation_file_name = annotation_file_name
            if not args.filter_twitter:
                ann.annotation_name = ann.comment
            resolved_audio = ann.source_path.resolve()
            ann.audio_path = str(resolved_audio)
        all_annotations.extend(mapped)

    for idx, ann in enumerate(all_annotations, start=1):
        ann.global_index = idx

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_annotations_csv(args.output_csv, all_annotations)

    if not args.skip_spectrograms and all_annotations:
        args.spectrogram_dir.mkdir(parents=True, exist_ok=True)
        common_root = (
            args.root.resolve()
            if not args.meta
            else Path(os.path.commonpath([str(meta.parent.resolve()) for meta in metafiles]))
        )
        generate_spectrograms(
            all_annotations,
            args.spectrogram_dir,
            args.spectrogram_padding,
            common_root,
        )

    if args.extract_clips and all_annotations:
        args.clip_dir.mkdir(parents=True, exist_ok=True)
        common_root = (
            args.root.resolve()
            if not args.meta
            else Path(os.path.commonpath([str(meta.parent.resolve()) for meta in metafiles]))
        )
        extract_audio_clips(
            all_annotations,
            args.clip_dir,
            args.clip_padding,
            args.bandpass_margin,
            common_root,
        )


if __name__ == "__main__":
    main()
