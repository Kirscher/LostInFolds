# LostInFolds

[![Tests](https://github.com/Kirscher/LostInFolds/actions/workflows/tests.yml/badge.svg)](https://github.com/Kirscher/LostInFolds/actions/workflows/tests.yml)

Companion repository for the MICCAI 2026 paper **"Lost in the Folds: When Cross-Validation Is Not a Deep Ensemble for Uncertainty Estimation"**.

The code supports public multi-rater medical image segmentation experiments comparing cross-validation (CV) ensembles with deep ensembles (DE). It includes dataset preparation helpers and ensemble uncertainty metrics used for calibration, ambiguity modeling, failure detection, and segmentation quality analysis.

## Installation

Use Python 3.12 or a compatible recent Python 3 release.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the test suite with:

```bash
pytest -q
```

## Documentation

- **[Datasets](src/data/README.md)** — Multi-rater datasets (GleasonXAI, RIGA, CURVAS, etc.), folder layout, and preparation instructions.
- **[Ensemble metrics](src/ensemble_metrics/README.md)** — Task definitions, metrics (ACE, BA-ECE, SPACE, NCC, GED, AURC), and usage of the metrics module.

## Quick links

| Topic | Location |
|-------|----------|
| Datasets layout (imagesTr, labelsTr, …) | [src/data/README.md](src/data/README.md) |
| Datasets preparation | [src/data/README.md](src/data/README.md) |
| Metrics module (CLI, API, structure) | [src/ensemble_metrics/README.md](src/ensemble_metrics/README.md) |
| Metrics & uncertainty tasks | [src/ensemble_metrics/README.md](src/ensemble_metrics/README.md) |

## Data and Model Artifacts

Datasets must be downloaded from their original providers and prepared locally. This repository does not include medical image data, model checkpoints, predictions, logs, or generated nnU-Net folders. Common local artifact directories such as `data/`, `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`, `results/`, and `checkpoints/` are ignored by Git.

## Citation

If you use this work, please cite:

```
TODO(author): Add BibTeX or proceedings reference when published.
```
