# Trill Sparrow Dataset Workflow

This project contains two main scripts for turning raw annotation logs into an augmented YOLO detection dataset:

1. `process_trill_metadata.py` remaps master annotation logs onto individual WAV files, verifies time offsets, and writes a consolidated `processed_annotations.csv` alongside optional spectrogram images and audio clips.
2. `augment_yolo_dataset.py` consumes the processed CSV and original WAV recordings to create train/val/test splits of spectrogram images with YOLO labels and augmented variants.

Follow the steps below to regenerate the dataset from scratch.

## 1. Process Raw Metadata

Run `process_trill_metadata.py` from the project root. The most common workflow is to point the script at the folder that holds the metafiles and WAV recordings; everything else can use defaults.

```bash
python process_trill_metadata.py \
  --root "2024 06 09 SMM144" \
  --output-csv processed_annotations.csv \
  --spectrogram-dir spectrograms
```

Key options:

- `--root` controls where `*metafile.txt` and referenced WAV files are searched. Omit it if you run the command from inside that folder.
- `--meta` lets you target specific metafiles instead of scanning the entire root.
- `--audio-dir` overrides the location of the WAV files if they are stored elsewhere.
- `--default-spacer-seconds` sets the baseline duration (seconds) to assume for spacer entries when the metafile lists them without audio; it defaults to 55 s.
- `--spacer-duration-override NAME=SECONDS` customises the spacer duration for individual metafiles. Repeat the flag to override multiple metafiles (the match is case-insensitive against the metafile stem/name). Example:

  ```bash
  python process_trill_metadata.py \
    --root "2024 06 09 SMM144" \
    --default-spacer-seconds 55 \
    --spacer-duration-override SMM144_20240612_040000=30 \
    --spacer-duration-override other_metafile=45
  ```
- `--skip-spectrograms` disables image generation (useful for a quick CSV-only run).
- `--extract-clips` and `--clip-dir` enable writing per-annotation WAV snippets.
- `--filter-twitter` (with `--twitter-signatures`) restricts the output to annotations whose `comment` matches an `AnnotationName` entry from the signatures CSV—handy when you want an “only trills” dataset defined in that sheet.

Example for filtering to trills listed in `Twitter Vocal Signatures.csv`:

```bash
python process_trill_metadata.py \
  --root "2024 06 09 SMM144" \
  --filter-twitter \
  --twitter-signatures "Twitter Vocal Signatures.csv"
```

The script writes:

- `processed_annotations.csv` (input for the augmentation step). The CSV now preserves the original `FolderName`, `MetafileName`, `AnnotationFileName`, `AnnotationName`, and a new absolute `AudioPath` so every annotation can be traced to the exact WAV file wherever you run the next steps.
- Spectrogram PNGs under `spectrograms/` (unless skipped)
- Optional audio clips under `clips/`

Refer to `python process_trill_metadata.py --help` for the full argument list.

## 2. Build the Augmented YOLO Dataset

Once the CSV exists and all source WAV files are available, run:

```bash
python augment_yolo_dataset.py \
  --annotations-csv processed_annotations.csv \
  --audio-root "2024 06 09 SMM144" \
  --output-dir yolo-dataset \
  --overwrite
```

Important flags:

- `--audio-root` provides the base directory for resolving the `AudioPath` entries from the CSV. Leave it at the default (`.`) if the relative paths already point into your current project tree; otherwise point it at the parent directory holding the recordings.
- `--overwrite` clears the output directory if you are regenerating the dataset.
- `--slice-seconds`, `--train-ratio`, `--val-ratio`, and `--test-ratio` control slice duration and dataset splits.
- Augmentation knobs include `--shift-variants`, `--horizontal-flip`, `--dropout-variants`, `--dropout-rects`, `--dropout-size-range`, `--xy-mask-variants`, and `--xy-mask-max-frac`. With defaults (no flips) each train annotation can produce up to nine spectrogram variants.

`augment_yolo_dataset.py` first attempts to load the WAV from the absolute `AudioPath` column. If a legacy row contains a relative path, it is resolved against `--audio-root`; when unavailable it falls back to combining `FolderName` and `soundfile`, and finally to a recursive search as a safety net.

Outputs are written to `yolo-dataset/` with sub-directories matching YOLOv5 conventions (images/labels for train/val/test, plus annotated image previews and `dataset.yaml`).

Run `python augment_yolo_dataset.py --help` to inspect all available parameters.

## 3. Train the YOLO Model

Install Ultralytics once (ideally in a virtual environment):

```bash
pip install ultralytics
```

Then kick off training with the helper script. By default it targets the freshly created `yolo-dataset/dataset.yaml`, loads the light YOLO11n checkpoint, and trains for 100 epochs at 640×640 resolution:

```bash
python train_yolo_model.py \
  --model yolo11n.pt \
  --dataset yolo-dataset/dataset.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0          # or 'cpu', 'mps', '0,1', -1, etc.
```

Useful flags:

- `--resume` continues from the checkpoint specified in `--model`.
- `--skip-val` disables per-epoch validation (handy for quick smoke tests, but keep validation on for meaningful metrics).
- `--project/--name/--exist-ok` mirror the Ultralytics CLI arguments for run organisation.
- `--cache ram|disk` caches spectrograms to accelerate subsequent epochs.

Run `python train_yolo_model.py --help` to see the full list of options, all of which forward to the underlying Ultralytics `YOLO.train` call.

## 4. Run Inference on New Audio

After training completes, point the best checkpoint at any WAV file or folder (recursively if desired) to generate detections. The script recreates the spectrogram slices used during training, runs YOLO, and converts bounding boxes back into physical time/frequency coordinates.

```bash
python infer_trills.py \
  "2024 06 09 SMM144/recordings" \
  --recursive \
  --model runs/trills/exp/weights/best.pt \
  --output-csv detections.csv \
  --slice-seconds 3.0 \
  --hop-seconds 0.5 \
  --confidence 0.35 \
  --max-detections 2
```

Key flags:

- Provide one or more WAV paths or directories as positional arguments; add `--recursive` to descend into sub-folders.
- `--slice-seconds` and `--hop-seconds` control the sliding window; keep the slice length identical to training for best results.
- `--confidence` filters low-scoring detections, while `--max-detections` caps how many boxes are retained per spectrogram (default 2).
- `--imgsz`, `--figure-width/--figure-height`, and `--dpi` should only change if you retrained with different rendering settings.

The resulting CSV contains the audio path, window index, absolute time boundaries, frequency bounds (Hz), duration, and YOLO confidence for each detection.

## 5. Regeneration Checklist

1. Ensure all metafiles, annotation logs, and WAV recordings reside under the same root.
2. Execute `process_trill_metadata.py` to refresh `processed_annotations.csv`.
3. Execute `augment_yolo_dataset.py` (optionally with `--overwrite`) to rebuild the YOLO dataset.
4. Train via `train_yolo_model.py`.
5. Run `infer_trills.py` against fresh recordings.
6. Inspect `yolo-dataset/dataset_stats.json`, the generated spectrograms, Ultralytics run artifacts, and inference CSVs to confirm everything looks reasonable.

That’s it—your augmented detection dataset is ready for training.
