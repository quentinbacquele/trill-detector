# Interactive Detection Viewer Parameters

The `interactive_detection_viewer.py` helper exposes several knobs for exploring detections inside a notebook. Each slider/toggle directly controls part of the processing pipeline. Use this reference to understand what changing a given control will do.

| Control | Description | Tips |
| --- | --- | --- |
| **Detection** | Selects which detection row (from the CSV) is rendered. Each detection keeps its own cached audio to speed up repeated tweaks. | Switch detections at any time—the other settings persist. |
| **Slice (ms)** | Length of each analysis slice (window) in milliseconds. Shorter slices increase temporal precision; longer slices increase frequency resolution. | Typical values: 10–40 ms. |
| **Noise reduction** | Toggles spectral-gating denoising that uses the lowest-RMS slice as a noise prototype. Off = raw audio. | Disable when you want to see the untouched spectrogram. |
| **Trill Hz** | Minimum dominant frequency (Hz) required to label a slice as a “Trill”. | Use higher values to ignore low-frequency artifacts. |
| **Rel energy** | Relative energy threshold (0–1) expressed as a fraction of the strongest dominant-power slice within the detection. Slices below this fraction are treated as “Silence” even if they meet the frequency threshold. | Example: 0.05 keeps slices with ≥5% of the strongest energy. |
| **Floor pct** | Percentile used to estimate the local noise floor inside each slice’s spectrum (10 % = 10th percentile). | Increase to raise the floor estimate in noisier recordings. |
| **Floor margin** | Multiplier applied to the estimated floor before searching for “floor crossings”. Larger values widen the gated area around the dominant peak. | Try 1.0–1.5 depending on how aggressively you want to trim peaks. |
| **Min floor bins** | Number of consecutive frequency bins that must stay below the floor threshold when locating the left/right red boundaries. Higher values reduce sensitivity to brief dips. | Increase if the red lines jump too close to the peak; decrease for tightly bounded peaks. |
| **Smooth bins** | Width (in frequency bins) of the moving average applied to the power spectrum before searching for floor crossings. Larger values smooth more aggressively. | Use odd values (e.g., 5, 7, 9) to maintain symmetry. |
| **Floor span Hz** | Maximum frequency span (Hz) allowed when searching for floor crossings on each side of the dominant peak. The search stops once it exceeds this distance. | Set higher (up to 10 kHz) for broadband signals; lower for narrow-band trills. |
| **Reset** | Restores every control (except detection selection) to its default value. | Use after experimenting to get back to the baseline view. |

### Visualization Notes

- The background spectrogram always shows the selected detection window (including padding).
- The green curve traces the dominant frequency over time (only for slices labeled “Trill”).
- Red dashed curves mark the low/high floor crossings, and the shaded red band fills the span between them.
- The “Trill rate” annotation reports how many slices were labeled as trills per second of analyzed audio.
