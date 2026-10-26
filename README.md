# AI-generated music detection with SVD and subspace stability

Please filly me out >~<

I've left some instructions here, but some more info in README would be nice

## Setup

Requirements: Python 3.10+, FFmpeg (`ffmpeg` and `ffprobe` must be on `PATH`),
    and enough disk space for the 12.9 GB FakeMusicCaps archive plus extracted files and processed WAVs.

```bash
cd music-ai-svd-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
ffmpeg -version
ffprobe -version
pytest
```

## Download data

First download the captions and 10-second human clips. Start with a tiny smoke test before all 5,521 rows:

```bash
python scripts/fetch_musiccaps.py --limit 10 --workers 1
# Then, when the sample works:
python scripts/fetch_musiccaps.py --workers 2
# Retry failed/unavailable links later:
python scripts/fetch_musiccaps.py --retry-failed --workers 1
```

The script downloads the public metadata CSV directly
(no Hugging Face `datasets` loader is needed), stores metadata at `data/raw/musiccaps_metadata.csv`,
    raw 10-second audio at `data/raw/musiccaps_audio/`, and append-as-it-goes status rows at `data/raw/musiccaps_download_status.csv`.

Download and extract FakeMusicCaps (expect a multi-GB download and extraction):

```bash
python scripts/download_fakemusiccaps.py --extract
```

The default extracted root is `data/raw/fakemusiccaps/extracted/`.

## Run the pipeline

```bash
python scripts/build_manifest.py
python scripts/preprocess_audio.py
python scripts/extract_features.py
python scripts/train_evaluate.py
```

Outputs:

- `data/processed/manifest.csv`: balanced paired metadata with caption-group split.
- `data/processed/manifest_harmonized.csv`: canonical audio paths plus raw provenance and bandwidth baseline.
- `data/processed/features.csv`: exact per-clip SVD features, principal-angle stability, and bandwidth.
- `artifacts/metrics.csv`: test-set accuracy, balanced accuracy, ROC-AUC, precision/recall/F1, confusion-matrix counts, and validation-selected threshold for three models.
- `artifacts/shared_spectral_basis.npy`: randomized shared left-singular basis learned only on training clips.
- `artifacts/features_with_shared_basis.csv`: local and shared-basis features.
