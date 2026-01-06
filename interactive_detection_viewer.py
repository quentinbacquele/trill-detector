"""
Interactive detection viewer for Jupyter notebooks.

Usage inside a notebook:
```python
from interactive_detection_viewer import create_detection_viewer
ui = create_detection_viewer(
    csv_path="inf/trill_detections_s3.00_h1.00_c0.75.csv",
)
ui
```
Adjust the widgets to explore different detections/parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import noisereduce as nr
from IPython.display import clear_output, display


@dataclass
class ViewerConfig:
    slice_ms: float = 20.0
    num_slices: int = 20
    pad_ms: float = 50.0
    floor_percentile: float = 10.0
    floor_margin: float = 1.12
    min_floor_bins: int = 2
    floor_smooth_bins: int = 5
    floor_max_delta_hz: float = 1200.0
    trill_freq_threshold: float = 6000.0
    trill_power_relative: float = 0.05
    use_noise_reduction: bool = True
    dedupe_trills: bool = False
    dedupe_window_ms: float = 20.0


def load_segment(audio_path: Path, start_s: float, end_s: float) -> Tuple[np.ndarray, int, float]:
    """Load mono audio for the requested window."""
    info = sf.info(str(audio_path))
    sr = info.samplerate
    duration = info.frames / sr
    actual_start = max(0.0, start_s)
    actual_end = min(duration, end_s)
    start_frame = int(round(actual_start * sr))
    frames = int(round((actual_end - actual_start) * sr))
    audio, _ = sf.read(str(audio_path), start=start_frame, frames=frames, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr, actual_start


def extract_slice(audio: np.ndarray, sr: int, segment_start: float, slice_start: float, slice_duration: float):
    slice_frames = int(round(slice_duration * sr))
    offset = int(round((slice_start - segment_start) * sr))
    pad_left = max(0, -offset)
    read_start = max(0, offset)
    read_end = read_start + max(0, slice_frames - pad_left)
    clip = audio[read_start:read_end]
    pad_right = max(0, slice_frames - pad_left - clip.shape[0])
    if pad_left or pad_right:
        clip = np.pad(clip, (pad_left, pad_right))
    return clip


def power_spectrum(signal: np.ndarray, sr: int):
    if signal.size == 0:
        return np.array([]), np.array([])
    window = np.hanning(signal.size)
    windowed = signal * window
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1 / sr)
    return freqs, power


def find_floor_crossing(freqs_band, analysis_power, dom_idx, threshold, min_bins, direction, max_delta_hz):
    n = analysis_power.size
    if n == 0:
        return dom_idx
    step = -1 if direction == "left" else 1
    idx = dom_idx + step
    if idx < 0 or idx >= n:
        return dom_idx

    limit_freq = freqs_band[dom_idx] - max_delta_hz if direction == "left" else freqs_band[dom_idx] + max_delta_hz
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
                crossing_idx = idx + (consecutive - 1) if step < 0 else idx - (consecutive - 1)
                return max(0, min(n - 1, crossing_idx))
        else:
            consecutive = 0
        idx += step

    return best_idx


def load_detection_base(row, config: ViewerConfig):
    audio_path = Path(row["audio_path"])
    detection_window_start = row["start_time"] - config.pad_ms / 1000.0
    detection_window_end = row["end_time"] + config.pad_ms / 1000.0
    audio, sr, segment_start = load_segment(audio_path, detection_window_start, detection_window_end)
    return {
        "audio": audio,
        "sr": sr,
        "segment_start": segment_start,
        "detection_window_start": detection_window_start,
        "detection_window_end": detection_window_end,
    }


def prepare_detection_data(row, base, config: ViewerConfig):
    audio = base["audio"].copy()
    sr = base["sr"]
    segment_start = base["segment_start"]
    detection_window_start = base["detection_window_start"]
    detection_window_end = base["detection_window_end"]
    slice_duration = config.slice_ms / 1000.0

    total_slices = int(np.ceil((detection_window_end - detection_window_start) / slice_duration))
    center_idx = total_slices // 2
    half = config.num_slices // 2
    start_idx = max(0, center_idx - half)
    end_idx = min(total_slices, start_idx + config.num_slices)
    if end_idx - start_idx < config.num_slices:
        start_idx = max(0, end_idx - config.num_slices)

    if config.use_noise_reduction:
        noise_clip = None
        min_rms = float("inf")
        for slice_idx in range(total_slices):
            slice_start = detection_window_start + slice_idx * slice_duration
            clip = extract_slice(audio, sr, segment_start, slice_start, slice_duration)
            if clip.size == 0:
                continue
            rms = float(np.sqrt(np.mean(clip**2)))
            if rms < min_rms:
                min_rms = rms
                noise_clip = clip.copy()
        if noise_clip is not None and noise_clip.size:
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
    for slice_idx in range(total_slices):
        slice_start = detection_window_start + slice_idx * slice_duration
        clip = extract_slice(audio, sr, segment_start, slice_start, slice_duration)
        freqs, power = power_spectrum(clip, sr)
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
                power_floor = max(float(np.percentile(power_band, config.floor_percentile)), 1e-12)
                floor_threshold = power_floor * config.floor_margin
                if config.floor_smooth_bins > 1 and power_band.size > 1:
                    kernel_len = min(config.floor_smooth_bins, power_band.size)
                    kernel = np.ones(kernel_len, dtype=float) / kernel_len
                    analysis_power = np.convolve(power_band, kernel, mode="same")
                else:
                    analysis_power = power_band
                dom_idx = int(np.argmax(power_band))
                dom_freq = freqs_band[dom_idx]
                dom_power = float(power_band[dom_idx])
                left_idx = find_floor_crossing(
                    freqs_band,
                    analysis_power,
                    dom_idx,
                    floor_threshold,
                    config.min_floor_bins,
                    "left",
                    config.floor_max_delta_hz,
                )
                right_idx = find_floor_crossing(
                    freqs_band,
                    analysis_power,
                    dom_idx,
                    floor_threshold,
                    config.min_floor_bins,
                    "right",
                    config.floor_max_delta_hz,
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
    power_cutoff = max_dom_power * config.trill_power_relative if max_dom_power > 0 else None
    for entry in all_slice_data:
        dom_freq = entry.get("dom_freq")
        dom_power = entry.get("dom_power")
        if dom_freq is not None and dom_freq >= config.trill_freq_threshold:
            if power_cutoff is None or (dom_power is not None and dom_power >= power_cutoff):
                entry["classification"] = "Trill"
            else:
                entry["classification"] = "Silence"
        else:
            entry["classification"] = "Silence"

    plot_slice_data = all_slice_data[start_idx:end_idx]
    if len(plot_slice_data) < config.num_slices and all_slice_data:
        plot_slice_data = all_slice_data[max(0, len(all_slice_data) - config.num_slices) :]

    return {
        "row": row,
        "audio": audio,
        "sr": sr,
        "segment_start": segment_start,
        "slice_duration_s": slice_duration,
        "plot_slice_data": plot_slice_data,
        "all_slice_data": all_slice_data,
        "spec_pxx_db": spec_pxx_db,
        "spec_freqs": spec_freqs,
        "spec_bins": spec_bins,
        "db_vmin": db_vmin,
        "db_vmax": db_vmax,
        "center_time": 0.5 * (row["start_time"] + row["end_time"]),
        "window_start": detection_window_start,
    }


def compute_detection_stats(det_data, config: ViewerConfig):
    dom_trill = []
    low_freqs = []
    high_freqs = []
    trill_times = []
    trill_dom = []
    trill_low = []
    trill_high = []
    total_slices = 0
    trill_slices = 0

    for entry in det_data["all_slice_data"]:
        dom = entry.get("dom_freq")
        low = entry.get("left_freq")
        high = entry.get("right_freq")
        classification = entry.get("classification", "Silence")
        if dom is not None and classification == "Trill":
            dom_trill.append(dom)
        if low is not None:
            low_freqs.append(low)
        if high is not None:
            high_freqs.append(high)
        if classification == "Trill":
            trill_slices += 1
            trill_dom.append(dom if dom is not None else np.nan)
            trill_low.append(low if low is not None else np.nan)
            trill_high.append(high if high is not None else np.nan)
            trill_times.append(entry.get("slice_start", 0.0) - det_data["window_start"])
        total_slices += 1

    slice_duration = det_data["slice_duration_s"]
    total_time = total_slices * slice_duration
    trill_rate_hz = trill_slices / total_time if total_time > 0 else 0.0
    stats = {
        "dom_trill_freqs": np.array(dom_trill),
        "low_freqs": np.array(low_freqs),
        "high_freqs": np.array(high_freqs),
        "trill_rate_hz": trill_rate_hz,
        "trill_times": np.array(trill_times),
        "trill_dom": np.array(trill_dom),
        "trill_low": np.array(trill_low),
        "trill_high": np.array(trill_high),
    }
    if config.dedupe_trills and stats["trill_times"].size > 0:
        stats = dedupe_trill_stats(stats, config)
    return stats


def dedupe_trill_stats(stats, config: ViewerConfig):
    times = stats["trill_times"]
    dom = stats["trill_dom"]
    low = stats["trill_low"]
    high = stats["trill_high"]
    window_s = config.dedupe_window_ms / 1000.0

    order = np.argsort(times)
    times = times[order]
    dom = dom[order]
    low = low[order]
    high = high[order]

    mask = dom > 0
    times = times[mask]
    dom = dom[mask]
    low = low[mask]
    high = high[mask]

    kept_idx = []
    group_start = 0
    while group_start < len(times):
        group_end = group_start + 1
        while group_end < len(times) and (times[group_end] - times[group_start]) < window_s:
            group_end += 1
        group_slice = slice(group_start, group_end)
        group_dom = dom[group_slice]
        if group_dom.size:
            best_local_idx = group_start + int(np.nanargmax(group_dom))
            kept_idx.append(best_local_idx)
        group_start = group_end

    if kept_idx:
        kept_idx = np.array(sorted(kept_idx))
        new_times = times[kept_idx]
        new_dom = dom[kept_idx]
        new_low = low[kept_idx]
        new_high = high[kept_idx]
    else:
        new_times = np.array([])
        new_dom = np.array([])
        new_low = np.array([])
        new_high = np.array([])

    slice_duration = config.slice_ms / 1000.0
    total_time = slice_duration * stats["trill_times"].size if stats["trill_times"].size else slice_duration
    trill_rate = new_times.size / max(total_time, 1e-9)
    return {
        "dom_trill_freqs": stats["dom_trill_freqs"],
        "low_freqs": stats["low_freqs"],
        "high_freqs": stats["high_freqs"],
        "trill_rate_hz": trill_rate,
        "trill_times": new_times,
        "trill_dom": new_dom,
        "trill_low": new_low,
        "trill_high": new_high,
    }


def plot_detection_trajectory(det_data, stats):
    row = det_data["row"]
    fig, ax = plt.subplots(figsize=(12, 6))
    times = stats["trill_times"]
    dom = stats["trill_dom"] / 1000 if stats["trill_dom"].size else np.array([])
    low = stats["trill_low"] / 1000 if stats["trill_low"].size else np.array([])
    high = stats["trill_high"] / 1000 if stats["trill_high"].size else np.array([])

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

    if times.size and dom.size:
        order = np.argsort(times)
        times = times[order]
        dom = dom[order]
        low = low[order]
        high = high[order]
        valid_dom = np.isfinite(dom)
        valid_low = np.isfinite(low)
        valid_high = np.isfinite(high)
        fill_mask = valid_low & valid_high
        if fill_mask.any():
            ax.fill_between(times[fill_mask], low[fill_mask], high[fill_mask], color="red", alpha=0.15)
        if valid_dom.any():
            ax.plot(times[valid_dom], dom[valid_dom], color="green", lw=1.2, label="Dominant")
            ax.scatter(times[valid_dom], dom[valid_dom], color="green", s=20)
        if valid_low.any():
            ax.plot(times[valid_low], low[valid_low], color="red", lw=1.0, linestyle="--", label="Low Floor")
            ax.scatter(times[valid_low], low[valid_low], color="red", s=16)
        if valid_high.any():
            ax.plot(times[valid_high], high[valid_high], color="red", lw=1.0, linestyle="--", label="High Floor")
            ax.scatter(times[valid_high], high[valid_high], color="red", s=16)
        ax.set_xlim(0, max(0.05, float(np.nanmax(times))))
    else:
        ax.text(0.5, 0.5, "No trill slices detected", ha="center", va="center", transform=ax.transAxes, color="white")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_ylim(0, 12)
    ax.set_title(
        f"Detection {int(row['detection_index']):03d} — {Path(row['audio_path']).name} "
        f"({row['start_time']:.2f}-{row['end_time']:.2f}s)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2, color="white")
    ax.text(
        0.01,
        0.95,
        f"Trill rate: {stats['trill_rate_hz']:.2f} Hz",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )
    fig.tight_layout()
    return fig


def render_detection(row, config: ViewerConfig, base_cache: Dict[int, dict]):
    det_id = int(row["detection_index"])
    if det_id not in base_cache:
        base_cache[det_id] = load_detection_base(row, config)
    det_data = prepare_detection_data(row, base_cache[det_id], config)
    stats = compute_detection_stats(det_data, config)
    traj_fig = plot_detection_trajectory(det_data, stats)
    return traj_fig


def create_detection_viewer(csv_path: str | Path = "inf/trill_detections_s3.00_h1.00_c0.75.csv"):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df = df[(df["duration"] >= 1.0) & (df["duration"] <= 5.0)].copy()
    if df.empty:
        raise ValueError("No detections found in the specified CSV.")

    detection_options = [
        (
            f"{int(row['detection_index']):03d} — {Path(row['audio_path']).name} "
            f"t={row['start_time']:.2f}-{row['end_time']:.2f}s",
            int(row["detection_index"]),
        )
        for _, row in df.iterrows()
    ]
    detection_lookup = {int(row["detection_index"]): row for _, row in df.iterrows()}
    base_cache: Dict[int, dict] = {}

    defaults = ViewerConfig()
    detection_dropdown = widgets.Dropdown(options=detection_options, description="Detection:")
    slice_slider = widgets.FloatSlider(
        value=defaults.slice_ms,
        min=1.0,
        max=100.0,
        step=1.0,
        description="Slice (ms):",
        continuous_update=False,
    )
    noise_checkbox = widgets.Checkbox(value=True, description="Noise reduction")
    trill_freq_slider = widgets.IntSlider(
        value=int(defaults.trill_freq_threshold),
        min=0,
        max=12000,
        step=100,
        description="Trill Hz:",
        continuous_update=False,
    )
    energy_slider = widgets.FloatSlider(
        value=defaults.trill_power_relative,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Rel energy:",
        continuous_update=False,
    )
    dedupe_checkbox = widgets.Checkbox(value=defaults.dedupe_trills, description="Dedupe trills")
    dedupe_window_slider = widgets.FloatSlider(
        value=defaults.dedupe_window_ms,
        min=1.0,
        max=100.0,
        step=1.0,
        description="Dedupe window (ms):",
        continuous_update=False,
    )
    floor_percentile = widgets.IntSlider(
        value=int(defaults.floor_percentile),
        min=1,
        max=100,
        step=1,
        description="Floor pct:",
        continuous_update=False,
    )
    floor_margin = widgets.FloatSlider(
        value=defaults.floor_margin,
        min=1.0,
        max=2.0,
        step=0.01,
        description="Floor margin:",
        continuous_update=False,
    )
    min_floor_bins = widgets.IntSlider(
        value=defaults.min_floor_bins,
        min=1,
        max=100,
        step=1,
        description="Min floor bins:",
        continuous_update=False,
    )
    smooth_bins = widgets.IntSlider(
        value=defaults.floor_smooth_bins,
        min=1,
        max=101,
        step=1,
        description="Smooth bins:",
        continuous_update=False,
    )
    floor_delta = widgets.IntSlider(
        value=int(defaults.floor_max_delta_hz),
        min=200,
        max=10000,
        step=100,
        description="Floor span Hz:",
        continuous_update=False,
    )
    reset_button = widgets.Button(description="Reset", button_style="warning", icon="undo")

    output = widgets.Output()

    def update(*_):
        det_id = detection_dropdown.value
        row = detection_lookup[det_id]
        config = ViewerConfig(
            slice_ms=slice_slider.value,
            use_noise_reduction=noise_checkbox.value,
            trill_freq_threshold=trill_freq_slider.value,
            trill_power_relative=energy_slider.value,
            dedupe_trills=dedupe_checkbox.value,
            dedupe_window_ms=dedupe_window_slider.value,
            floor_percentile=floor_percentile.value,
            floor_margin=floor_margin.value,
            min_floor_bins=min_floor_bins.value,
            floor_smooth_bins=smooth_bins.value,
            floor_max_delta_hz=floor_delta.value,
        )
        with output:
            clear_output(wait=True)
            try:
                traj_fig = render_detection(row, config, base_cache)
                display(traj_fig)
                plt.close(traj_fig)
            except Exception as exc:
                print(f"Error rendering detection: {exc}")

    controls = widgets.VBox(
        [
            detection_dropdown,
            widgets.HBox([slice_slider, noise_checkbox, reset_button]),
            widgets.HBox([trill_freq_slider, energy_slider]),
            widgets.HBox([dedupe_checkbox, dedupe_window_slider]),
            widgets.HBox([floor_percentile, floor_margin]),
            widgets.HBox([min_floor_bins, smooth_bins, floor_delta]),
        ]
    )

    def handle_reset(_):
        slice_slider.value = defaults.slice_ms
        noise_checkbox.value = defaults.use_noise_reduction
        trill_freq_slider.value = int(defaults.trill_freq_threshold)
        energy_slider.value = defaults.trill_power_relative
        dedupe_checkbox.value = defaults.dedupe_trills
        dedupe_window_slider.value = defaults.dedupe_window_ms
        floor_percentile.value = int(defaults.floor_percentile)
        floor_margin.value = defaults.floor_margin
        min_floor_bins.value = defaults.min_floor_bins
        smooth_bins.value = defaults.floor_smooth_bins
        floor_delta.value = int(defaults.floor_max_delta_hz)

    reset_button.on_click(handle_reset)

    for widget in [
        detection_dropdown,
        slice_slider,
        noise_checkbox,
        trill_freq_slider,
        energy_slider,
        dedupe_checkbox,
        dedupe_window_slider,
        floor_percentile,
        floor_margin,
        min_floor_bins,
        smooth_bins,
        floor_delta,
    ]:
        widget.observe(update, names="value")

    update()
    return widgets.VBox([controls, output])
