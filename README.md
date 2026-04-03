# Trill Detector

A YOLO-based system for detecting bird trill vocalizations in audio recordings. The pipeline converts audio into spectrograms and uses deep learning to identify trill calls.

## Getting Started

### 1. Download the Project

Open Terminal and run:

```bash
git clone https://github.com/quentinbacquele/trill-detector.git
cd trill-detector
```

### 2. Install Python (if needed)

Check if Python is installed:

```bash
python3 --version
```

If not installed, download from [python.org](https://www.python.org/downloads/) (version 3.10 or higher recommended).

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or if you use `uv`:

```bash
uv sync
```

### 4. Update the Project (when new versions are released)

To get the latest updates:

```bash
cd trill-detector
git pull
```

---

## Trained Models

Pre-trained models are included in `models/`:

| Model | Size | Description |
|-------|------|-------------|
| `yolo11n_best.pt` | 5 MB | Nano model (faster, lighter) |
| `yolo11l_best.pt` | 49 MB | Large model (more accurate) |

---

## Pipeline Overview

The workflow consists of 4 steps:

```
Raw Audio + Annotations
        ↓
1. process_trill_metadata.py  →  processed_annotations.csv
        ↓
2. augment_yolo_dataset.py    →  yolo-dataset/
        ↓
3. train_yolo_model.py        →  trained model weights
        ↓
4. infer_trills.py            →  trill_detections.csv
```

---

## Step 1: Process Raw Metadata

Converts master annotation logs into a structured CSV with per-file annotations.

The script rebuilds the annotation timeline from each metafile. Every WAV line contributes its real audio duration, and every spacer line contributes the configured spacer duration. Spacer filenames are only used to mark where a spacer exists; the effective spacer length comes from `--default-spacer-seconds` or a matching `--spacer-duration-override`.

```bash
python process_trill_metadata.py \
  --root "path/to/recordings" \
  --output-csv processed_annotations.csv \
  --spectrogram-dir spectrograms \
  --filter-twitter
```

Key options:

| Flag | Description |
|------|-------------|
| `--root` | Folder containing metafiles and WAV recordings |
| `--default-spacer-seconds` | Default spacer duration used for every spacer entry in a metafile (default `55.125`) |
| `--spacer-duration-override NAME=SECONDS` | Override spacer duration for a specific metafile or deployment folder |
| `--skip-spectrograms` | Skip image generation (faster) |
| `--extract-clips` | Save audio clips for each annotation |
| `--filter-twitter` | Filter to specific call types from signatures CSV |
| `--diagnose` | Print the reconstructed audio/spacer timeline used for annotation mapping |

Run `python process_trill_metadata.py --help` for all options.

---

## Step 2: Build YOLO Dataset

Creates train/val/test splits with augmented spectrograms.

```bash
python augment_yolo_dataset.py \
  --annotations-csv processed_annotations.csv \
  --audio-root "path/to/recordings" \
  --output-dir yolo-dataset \
  --overwrite
```

Key options:

| Flag | Description |
|------|-------------|
| `--slice-seconds` | Duration of each spectrogram slice |
| `--train-ratio` | Fraction for training (default 0.8) |
| `--horizontal-flip` | Enable horizontal flip augmentation |
| `--dropout-variants` | Number of dropout augmentations |

Run `python augment_yolo_dataset.py --help` for all options.

---

## Step 3: Train the Model

Train a YOLO model on your dataset.

```bash
python train_yolo_model.py \
  --model yolo11n.pt \
  --dataset yolo-dataset/dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --device mps   # Use 'cpu' or '0' for GPU
```

Key options:

| Flag | Description |
|------|-------------|
| `--model` | Base model (yolo11n.pt, yolo11m.pt, yolo11l.pt) |
| `--epochs` | Number of training epochs |
| `--device` | cpu, mps (Mac), 0 (CUDA GPU) |
| `--resume` | Continue from checkpoint |

Run `python train_yolo_model.py --help` for all options.

---

## Step 4: Run Inference

Detect trills in new audio files using a trained model.

```bash
python infer_trills.py \
  "path/to/audio" \
  --recursive \
  --model models/yolo11l_best.pt \
  --output-csv detections.csv \
  --confidence 0.35
```

Key options:

| Flag | Description |
|------|-------------|
| `--model` | Path to trained weights |
| `--recursive` | Search subdirectories for audio |
| `--confidence` | Minimum detection confidence (0-1) |
| `--slice-seconds` | Window size (match training) |
| `--hop-seconds` | Stride between windows |

Output CSV contains: audio path, time boundaries, frequency bounds, and confidence scores.

Run `python infer_trills.py --help` for all options.

---

## Additional Tools

### Batch Inference on Nest Recordings

```bash
python run_nest_inference.py \
  --input-dir "NestRecordings" \
  --output-dir "detections" \
  --model models/yolo11l_best.pt
```

### Interactive Viewer (Jupyter)

Explore detections interactively:

```bash
jupyter notebook interactive_trill_tracking.ipynb
```

### Power Spectrum Analysis

```bash
python plot_power_spectra.py --detections detections.csv
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Clone repo | `git clone https://github.com/quentinbacquele/trill-detector.git` |
| Update repo | `git pull` |
| Install deps | `pip install -r requirements.txt` |
| Run inference | `python infer_trills.py audio/ --model models/yolo11l_best.pt` |
