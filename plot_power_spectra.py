#!/usr/bin/env python3
"""Plot power spectra for short slices around detections."""

from pathlib import Path
import os

import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr
from matplotlib.patches import Rectangle

os.environ.setdefault("MPLCONFIGDIR", str((Path("inf") / "matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


DEFAULT_SLICE_MS = 20
NUM_SLICES = 20
PAD_MS = 50
FLOOR_PERCENTILE = 10
FLOOR_MARGIN = 1.12
MIN_FLOOR_BINS = 2
FLOOR_SMOOTH_BINS = 5
FLOOR_MAX_DELTA_HZ = 1200
TRILL_FREQ_THRESHOLD = 6000
TRILL_POWER_RELATIVE = 0.05
DEFAULT_NOISE_REDUCTION = True


def load_segment(audio_path: Path, request_start_s: float, request_end_s: float):
    """Load a slice of audio that covers the requested window."""
    info = sf.info(audio_path)
    sr = info.samplerate
    duration_s = info.frames / sr

    actual_start = max(0.0, request_start_s)
    actual_end = min(duration_s, request_end_s)
    start_frame = int(round(actual_start * sr))
    frames = int(round((actual_end - actual_start) * sr))

    audio, _ = sf.read(audio_path, start=start_frame, frames=frames, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio, sr, actual_start


def extract_slice(
    audio: np.ndarray,
    sr: int,
    segment_start_s: float,
    slice_start_s: float,
    slice_duration_s: float,
) -> np.ndarray:
    """Return a 1-D slice padded with zeros if it falls outside the loaded segment."""
    slice_frames = int(round(slice_duration_s * sr))
    offset_frames = int(round((slice_start_s - segment_start_s) * sr))

    pad_left = max(0, -offset_frames)
    read_start = max(0, offset_frames)
    read_end = read_start + max(0, slice_frames - pad_left)

    slice_audio = audio[read_start:read_end]
    pad_right = max(0, slice_frames - pad_left - slice_audio.shape[0])

    if pad_left or pad_right:
        slice_audio = np.pad(slice_audio, (pad_left, pad_right))

    return slice_audio


def power_spectrum(signal: np.ndarray, sr: int):
    """Compute a simple power spectrum of the provided signal."""
    if signal.size == 0:
        return np.array([]), np.array([])

    window = np.hanning(signal.size)
    windowed = signal * window
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1 / sr)
    return freqs, power


def prepare_detection_data(row, enable_noise_reduction: bool, slice_ms: float):
    """Pre-compute slice information and spectrogram data for a detection."""
    audio_path = Path(row["audio_path"])
    center_time = 0.5 * (row["start_time"] + row["end_time"])
    slice_duration_s = slice_ms / 1000

    detection_window_start = row["start_time"] - PAD_MS / 1000
    detection_window_end = row["end_time"] + PAD_MS / 1000

    audio, sr, segment_start = load_segment(audio_path, detection_window_start, detection_window_end)

    total_slices = int(np.ceil((detection_window_end - detection_window_start) / slice_duration_s))
    center_idx = total_slices // 2
    half = NUM_SLICES // 2
    start_idx = max(0, center_idx - half)
    end_idx = min(total_slices, start_idx + NUM_SLICES)
    if end_idx - start_idx < NUM_SLICES:
        start_idx = max(0, end_idx - NUM_SLICES)

    noise_slice_start = detection_window_start
    min_rms = float("inf")
    noise_clip = None

    for slice_idx in range(total_slices):
        slice_start = detection_window_start + slice_idx * slice_duration_s
        slice_audio = extract_slice(audio, sr, segment_start, slice_start, slice_duration_s)
        if slice_audio.size == 0:
            continue
        rms = float(np.sqrt(np.mean(slice_audio**2)))
        if rms < min_rms:
            min_rms = rms
            noise_slice_start = slice_start
            noise_clip = slice_audio.copy()

    if enable_noise_reduction and noise_clip is not None and noise_clip.size > 0:
        try:
            audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, stationary=False)
        except Exception as exc:
            print(f"Noise reduction failed for detection {row['detection_index']}: {exc}")

    spec_pxx, spec_freqs, spec_bins, _ = plt.specgram(
        audio,
        NFFT=512,
        Fs=sr,
        noverlap=256,
        detrend="mean",
        scale="dB",
        cmap="magma",
    )
    spec_pxx_db = 10 * np.log10(np.maximum(spec_pxx, 1e-12))
    db_vmax = float(np.nanmax(spec_pxx_db))
    db_vmin = db_vmax - 60.0
    plt.close()

    all_slice_data = []

    for slice_global_idx in range(total_slices):
        slice_start = detection_window_start + slice_global_idx * slice_duration_s
        slice_audio = extract_slice(audio, sr, segment_start, slice_start, slice_duration_s)
        freqs, power = power_spectrum(slice_audio, sr)

        entry = {
            "slice_start": slice_start,
            "freqs_band": np.array([]),
            "power_band": np.array([]),
            "power_floor": None,
            "dom_freq": None,
            "left_freq": None,
            "right_freq": None,
            "classification": "Silence",
            "dom_power": None,
        }

        if freqs.size:
            mask = (freqs >= 4000) & (freqs <= 12000)
            freqs_band = freqs[mask]
            power_band = power[mask]
            entry["freqs_band"] = freqs_band
            entry["power_band"] = power_band

            if freqs_band.size:
                power_floor = max(float(np.percentile(power_band, FLOOR_PERCENTILE)), 1e-12)
                floor_threshold = power_floor * FLOOR_MARGIN

                if FLOOR_SMOOTH_BINS > 1 and power_band.size > 1:
                    kernel = np.ones(min(FLOOR_SMOOTH_BINS, power_band.size), dtype=float)
                    kernel /= kernel.size
                    analysis_power = np.convolve(power_band, kernel, mode="same")
                else:
                    analysis_power = power_band

                dom_idx = int(np.argmax(power_band))
                dom_freq = freqs_band[dom_idx]
                dom_power = float(power_band[dom_idx]) if power_band.size else None

                left_idx = find_floor_crossing(
                    freqs_band,
                    analysis_power,
                    dom_idx,
                    floor_threshold,
                    MIN_FLOOR_BINS,
                    direction="left",
                    max_delta_hz=FLOOR_MAX_DELTA_HZ,
                )
                right_idx = find_floor_crossing(
                    freqs_band,
                    analysis_power,
                    dom_idx,
                    floor_threshold,
                    MIN_FLOOR_BINS,
                    direction="right",
                    max_delta_hz=FLOOR_MAX_DELTA_HZ,
                )

                entry.update(
                    {
                        "power_floor": power_floor,
                        "dom_freq": dom_freq,
                        "left_freq": freqs_band[left_idx],
                        "right_freq": freqs_band[right_idx],
                        "dom_power": dom_power,
                    }
                )

        all_slice_data.append(entry)

    max_dom_power = max((entry["dom_power"] or 0.0 for entry in all_slice_data), default=0.0)
    power_cutoff = max_dom_power * TRILL_POWER_RELATIVE if max_dom_power > 0 else None

    for entry in all_slice_data:
        dom_freq = entry.get("dom_freq")
        dom_power = entry.get("dom_power")
        if dom_freq is not None and dom_freq >= TRILL_FREQ_THRESHOLD:
            if power_cutoff is None or (dom_power is not None and dom_power >= power_cutoff):
                entry["classification"] = "Trill"
            else:
                entry["classification"] = "Silence"
        else:
            entry["classification"] = "Silence"

    plot_slice_data = all_slice_data[start_idx:end_idx]
    if len(plot_slice_data) < NUM_SLICES and all_slice_data:
        plot_slice_data = all_slice_data[max(0, len(all_slice_data) - NUM_SLICES) : len(all_slice_data)]

    return {
        "row": row,
        "audio": audio,
        "sr": sr,
        "segment_start": segment_start,
        "slice_duration_s": slice_duration_s,
        "plot_slice_data": plot_slice_data,
        "all_slice_data": all_slice_data,
        "spec_pxx_db": spec_pxx_db,
        "spec_freqs": spec_freqs,
        "spec_bins": spec_bins,
        "db_vmin": db_vmin,
        "db_vmax": db_vmax,
        "center_time": center_time,
        "window_start": detection_window_start,
        "noise_slice_start": noise_slice_start,
    }


def find_floor_crossing(
    freqs_band: np.ndarray,
    analysis_power: np.ndarray,
    dom_idx: int,
    threshold: float,
    min_bins: int,
    direction: str,
    max_delta_hz: float,
) -> int:
    """
    Return the first index from the dominant peak where power stays near the floor.
    Falls back to the lowest-power bin encountered within the search window.
    """
    n = analysis_power.size
    if n == 0:
        return dom_idx

    min_bins = max(1, min_bins)
    step = -1 if direction == "left" else 1
    idx = dom_idx + step
    if idx < 0 or idx >= n:
        return dom_idx

    center_freq = freqs_band[dom_idx]
    limit_freq = center_freq - max_delta_hz if direction == "left" else center_freq + max_delta_hz

    best_idx = idx
    best_power = analysis_power[idx]
    consecutive = 0

    while 0 <= idx < n:
        freq = freqs_band[idx]
        if (direction == "left" and freq < limit_freq) or (direction == "right" and freq > limit_freq):
            break

        current_power = analysis_power[idx]
        if current_power < best_power:
            best_power = current_power
            best_idx = idx

        if current_power <= threshold:
            consecutive += 1
            if consecutive >= min_bins:
                if step < 0:
                    crossing_idx = idx + (min_bins - 1)
                else:
                    crossing_idx = idx - (min_bins - 1)
                crossing_idx = max(0, min(n - 1, crossing_idx))
                return crossing_idx
        else:
            consecutive = 0

        idx += step

    return best_idx


def compute_detection_stats(det_data):
    """Aggregate slice metrics for a single detection."""
    dom_trill_freqs = []
    low_freqs = []
    high_freqs = []
    total_slices = 0
    trill_slices = 0
    trill_times = []
    trill_dom = []
    trill_low = []
    trill_high = []

    window_start = det_data["window_start"]

    for entry in det_data["all_slice_data"]:
        dom = entry.get("dom_freq")
        low = entry.get("left_freq")
        high = entry.get("right_freq")
        classification = entry.get("classification", "Silence")

        if dom is not None and classification == "Trill":
            dom_trill_freqs.append(dom)
        if low is not None:
            low_freqs.append(low)
        if high is not None:
            high_freqs.append(high)

        if classification == "Trill":
            trill_slices += 1
            trill_dom.append(dom if dom is not None else np.nan)
            trill_low.append(low if low is not None else np.nan)
            trill_high.append(high if high is not None else np.nan)
            trill_times.append(entry.get("slice_start", 0.0) - window_start)
        total_slices += 1

    slice_duration_s = det_data["slice_duration_s"]
    total_time = total_slices * slice_duration_s
    trill_rate_hz = trill_slices / total_time if total_time > 0 else 0.0

    return {
        "dom_trill_freqs": np.array(dom_trill_freqs),
        "low_freqs": np.array(low_freqs),
        "high_freqs": np.array(high_freqs),
        "trill_rate_hz": trill_rate_hz,
        "trill_times": np.array(trill_times),
        "trill_dom": np.array(trill_dom),
        "trill_low": np.array(trill_low),
        "trill_high": np.array(trill_high),
    }


def plot_stats_boxplot(ax, stats):
    """Render boxplots for the slice statistics."""
    data = [
        stats["dom_trill_freqs"] / 1000 if stats["dom_trill_freqs"].size else np.array([]),
        stats["low_freqs"] / 1000 if stats["low_freqs"].size else np.array([]),
        stats["high_freqs"] / 1000 if stats["high_freqs"].size else np.array([]),
    ]
    labels = ["Dom. Trill (kHz)", "Low Floor (kHz)", "High Floor (kHz)"]

    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    present = [i for i, d in enumerate(data) if d.size]
    if not present:
        ax.text(0.5, 0.5, "No slice statistics available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_ylabel("Frequency (kHz)")
        return

    box = ax.boxplot(
        [data[i] for i in present],
        tick_labels=[labels[i] for i in present],
        patch_artist=True,
        widths=0.5,
    )

    colors = ["tab:purple", "tab:red", "tab:orange"]
    for patch_idx, idx in enumerate(present):
        patch = box["boxes"][patch_idx]
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.4)
        mean_val = float(np.mean(data[idx]))
        ax.scatter(patch_idx + 1, mean_val, color="black", marker="D", zorder=5, s=25)
        ax.text(
            patch_idx + 1,
            mean_val,
            f"{mean_val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Slice Statistics")


def plot_trill_time_series(ax, stats, det_data):
    """Plot time-vs-frequency trajectories for trill slices."""
    times = stats["trill_times"]
    dom = stats["trill_dom"] / 1000 if stats["trill_dom"].size else np.array([])
    low = stats["trill_low"] / 1000 if stats["trill_low"].size else np.array([])
    high = stats["trill_high"] / 1000 if stats["trill_high"].size else np.array([])

    if times.size == 0 or dom.size == 0:
        ax.text(0.5, 0.5, "No trill slices detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        return

    order = np.argsort(times)
    times = times[order]
    dom = dom[order]
    low = low[order]
    high = high[order]

    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)

    # Spectrogram background aligned to detection window start
    spec_times = det_data["spec_bins"] + det_data["segment_start"] - det_data["window_start"]
    spec_freqs = det_data["spec_freqs"] / 1000
    ax.pcolormesh(
        spec_times,
        spec_freqs,
        det_data["spec_pxx_db"],
        shading="auto",
        cmap="magma",
        vmin=det_data["db_vmin"],
        vmax=det_data["db_vmax"],
        alpha=0.55,
    )

    valid_dom = np.isfinite(dom)
    valid_low = np.isfinite(low)
    valid_high = np.isfinite(high)
    fill_mask = valid_low & valid_high

    if fill_mask.any():
        ax.fill_between(times[fill_mask], low[fill_mask], high[fill_mask], color="red", alpha=0.15, label="Low–High Range")

    if valid_dom.any():
        ax.plot(times[valid_dom], dom[valid_dom], color="green", lw=1.0, label="Dominant")
        ax.scatter(times[valid_dom], dom[valid_dom], color="green", s=15)
    if valid_low.any():
        ax.plot(times[valid_low], low[valid_low], color="red", lw=0.8, linestyle="--", label="Low Floor")
        ax.scatter(times[valid_low], low[valid_low], color="red", s=12)
    if valid_high.any():
        ax.plot(times[valid_high], high[valid_high], color="red", lw=0.8, linestyle="--", label="High Floor")
        ax.scatter(times[valid_high], high[valid_high], color="red", s=12)

    max_time = float(np.nanmax(times)) if times.size else 0.0
    ax.set_xlim(0, max(0.05, max_time))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Trill Slice Trajectories")
    ax.set_ylim(0, 12)
    ax.legend(loc="upper right")
    ax.text(
        0.01,
        0.95,
        f"Trill rate: {stats['trill_rate_hz']:.2f} Hz",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )


def plot_detection_stats(det_data, stats, output_dir: Path):
    """Generate a stats figure for a single detection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (box_ax, traj_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.35},
    )

    plot_stats_boxplot(box_ax, stats)
    plot_trill_time_series(traj_ax, stats, det_data)

    row = det_data["row"]
    fig.suptitle(
        f"Detection {row['detection_index']} — {Path(row['audio_path']).name}",
        fontsize=12,
    )

    outfile = output_dir / f"detection_{row['detection_index']:03d}_stats.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def plot_detection_slices(det_data, output_dir: Path):
    row = det_data["row"]
    audio_path = Path(row["audio_path"])
    audio = det_data["audio"]
    sr = det_data["sr"]
    segment_start = det_data["segment_start"]
    slice_duration_s = det_data["slice_duration_s"]
    slice_data = det_data["plot_slice_data"]

    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(8, 5, height_ratios=[1] * 8, figure=fig)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(8)])

    for idx, slice_info in enumerate(slice_data):
        row_pair = (idx // 5) * 2
        col = idx % 5

        spec_ax = axes[row_pair, col]
        pow_ax = axes[row_pair + 1, col]

        freqs_band = slice_info["freqs_band"]
        power_band = slice_info["power_band"]
        power_floor = slice_info["power_floor"]
        dom_freq = slice_info["dom_freq"]
        left_freq = slice_info["left_freq"]
        right_freq = slice_info["right_freq"]
        classification = slice_info["classification"]
        slice_start = slice_info["slice_start"]

        if freqs_band.size:
            pow_ax.plot(freqs_band, power_band, lw=0.6, color="tab:blue")
            if power_floor is not None:
                pow_ax.axhline(power_floor, color="purple", linestyle="-", lw=0.8)
            if dom_freq is not None:
                pow_ax.axvline(dom_freq, color="limegreen", linestyle="--", lw=1)
            if left_freq is not None and right_freq is not None:
                pow_ax.axvline(left_freq, color="red", linestyle="-", lw=0.8)
                pow_ax.axvline(right_freq, color="red", linestyle="-", lw=0.8)
                if right_freq > left_freq and power_floor is not None:
                    band_mask = (freqs_band >= left_freq) & (freqs_band <= right_freq)
                    if np.any(band_mask):
                        pow_ax.fill_between(
                            freqs_band[band_mask],
                            power_band[band_mask],
                            power_floor,
                            color="red",
                            alpha=0.15,
                            zorder=0.5,
                        )
            pow_ax.set_xlim(4000, 12000)

        offset_ms = (slice_start - det_data["center_time"]) * 1000
        pow_ax.set_title(f"{offset_ms:+.0f} ms", fontsize=9)
        pow_ax.tick_params(labelsize=8)

        spec_ax.pcolormesh(
            det_data["spec_bins"] + segment_start,
            det_data["spec_freqs"],
            det_data["spec_pxx_db"],
            shading="auto",
            cmap="magma",
            vmin=det_data["db_vmin"],
            vmax=det_data["db_vmax"],
        )
        spec_ax.set_ylim(0, sr / 2)
        slice_end = slice_start + slice_duration_s
        spec_ax.axvspan(slice_start, slice_end, color="tab:blue", alpha=0.20, zorder=3)
        if dom_freq is not None:
            spec_ax.axhline(dom_freq, color="limegreen", linestyle="--", lw=1, zorder=6)
        if dom_freq is not None and left_freq is not None and right_freq is not None:
            lower_band = min(left_freq, right_freq)
            upper_band = max(left_freq, right_freq)
            if upper_band > lower_band:
                rect = Rectangle(
                    (slice_start, lower_band),
                    slice_duration_s,
                    upper_band - lower_band,
                    facecolor="red",
                    edgecolor="none",
                    alpha=0.35,
                    zorder=7,
                )
                spec_ax.add_patch(rect)

        label = "Trill" if classification == "Trill" else "Silence"
        spec_ax.text(
            0.5,
            1.02,
            label,
            transform=spec_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.3, boxstyle="round,pad=0.2"),
            clip_on=False,
        )
        spec_ax.tick_params(labelsize=8)

        if col == 0:
            spec_ax.set_ylabel("Freq (Hz)")
            pow_ax.set_ylabel("Power")
        if row_pair == 6:
            spec_ax.set_xlabel("Time (s)")

    fig.suptitle(
        f"Detection {row['detection_index']} — {audio_path.name} t={row['start_time']:.2f}-{row['end_time']:.2f}s",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"detection_{row['detection_index']:03d}_t{row['start_time']:.2f}s.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

    return outfile


def parse_args():
    parser = argparse.ArgumentParser(description="Plot power spectra for detections.")
    parser.add_argument(
        "--disable-noise-reduction",
        action="store_true",
        help="Disable spectral gating noise reduction.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("inf/trill_detections_s3.00_h1.00_c0.75.csv"),
        help="Path to detections CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("inf/power_spectrum_plots"),
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--slice-ms",
        type=float,
        default=DEFAULT_SLICE_MS,
        help="Slice duration in milliseconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv
    output_dir = args.output
    slice_ms = args.slice_ms
    enable_noise_reduction = not args.disable_noise_reduction if DEFAULT_NOISE_REDUCTION else False

    df = pd.read_csv(csv_path)
    filtered = df[(df["duration"] >= 1.0) & (df["duration"] <= 5.0)].copy()

    if filtered.empty:
        print("No detections with duration between 1s and 5s.")
        return

    slice_plots = []
    stats_plots = []
    for _, row in filtered.iterrows():
        det_data = prepare_detection_data(row, enable_noise_reduction=enable_noise_reduction, slice_ms=slice_ms)
        slice_file = plot_detection_slices(det_data, output_dir)
        slice_plots.append(slice_file)
        print(f"Saved {slice_file}")

        stats = compute_detection_stats(det_data)
        stats_file = plot_detection_stats(det_data, stats, output_dir)
        stats_plots.append(stats_file)
        print(f"Saved stats figure {stats_file}")

    print(
        f"Generated {len(slice_plots)} slice plots and {len(stats_plots)} stats plots in {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
