# AI-generated music detection with SVD and subspace comparison

The pipeline compares caption-matched human and AI clips using log-magnitude STFTs, singular values and singular-vector loading profiles. It also forms one joint SVD per pair and learns separate low-rank human/AI spaces from stacked training matrices. KNN is evaluated in the two class spaces; balanced-accuracy uncertainty is estimated by bootstrapping complete matched pairs.

## Setup

Requirements: Python 3.10+, FFmpeg (`ffmpeg` and `ffprobe` on `PATH`), and disk space for the source audio, processed WAVs and temporary class-stacked matrices.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
```

## Run

After downloading and preprocessing the audio as described in the existing download scripts:

```bash
python scripts/build_manifest.py
python scripts/preprocess_audio.py
python scripts/extract_features.py
python scripts/pairwise_svd.py
python scripts/train_evaluate.py --pairwise-features data/processed/pairwise_svd_features.csv
```

`extract_features.py` computes exact per-clip SVD features. Vector descriptors use squared singular-vector loadings (sign invariant): frequency distributions from the left vectors and time distributions from the right vectors, including leading-mode and subspace-leverage summaries.

`pairwise_svd.py` groups each caption-matched human/AI pair and computes exactly one SVD of `[L_human L_AI]`. It writes side-specific projections and the share of each joint right singular vector contributed by that clip.

`train_evaluate.py` builds two training-only stacked matrices, `A_human=[L_1 ... L_n]` and `A_AI=[L_1 ... L_n]`, materializes each as a temporary disk-backed matrix, and computes a truncated randomized SVD. The matrix file is removed after its class basis is fitted unless `--keep-stacked-matrices` is passed. The two left-singular-vector bases are also compared directly: principal angles measure subspace overlap, while a maximum-weight assignment on absolute dot products matches individual modes without sign ambiguity. Principal angles are the robust primary measure when singular values are close, since individual modes can rotate within a nearly-degenerate subspace. Clips are projected into both spaces, and distance-weighted KNN uses the resulting projection energies/residuals. KNN neighbourhood size is selected on the validation split.

Balanced-accuracy 95% confidence intervals use 2,000 percentile-bootstrap replicates, resampling caption groups as units (all matched pairs sharing a caption stay together). This preserves both pairwise and repeated-caption dependence.

## Outputs

- `data/processed/features.csv`: singular values, singular-vector descriptors and windowed principal-angle features.
- `data/processed/pairwise_svd_features.csv`: one joint-SVD feature row per side of every pair.
- `artifacts/human_space_basis.npy`, `artifacts/ai_space_basis.npy`: randomized class-space bases learned from training clips only.
- `artifacts/features_with_class_spaces.csv`: class-space projection features for all splits.
- `artifacts/metrics.csv`: baseline/local models and KNN class-space results, including paired-bootstrap 95% CIs for balanced accuracy.
- `artifacts/class_space_comparison.json`, `class_space_principal_angles.csv`, `class_space_vector_alignment.csv`: direct human-vs-AI stacked-basis comparison.
- `artifacts/experiment.json`: settings and method descriptions.

The current bundle does not include the processed WAV corpus, so run the commands above locally to generate new experiment metrics; no results are fabricated in the interim report.
