# Ensemble Metrics Module

A modular package for computing ensemble-based uncertainty metrics from nnUNet multi-fold predictions.

## Structure

```text
src/ensemble_metrics/
├── __init__.py           # Module exports
├── metric_functions.py   # Core metric computation functions
├── metrics.py            # Metric classes (MutualInformationMetric, etc.)
├── utils.py              # Utility functions (data loading, discovery)
├── compute.py            # Main computation orchestration
└── cli.py                # CLI entry point
```

## Usage

### Command Line

```bash
python -m src.ensemble_metrics.cli --pred-root /path/to/predictions --output-dir /path/to/output
```

### Python API

```python
from src.ensemble_metrics import (
    MutualInformationMetric,
    compute_mutual_information_wrapper,
    discover_folds,
    load_prediction,
)

# Use metric classes
metric = MutualInformationMetric(output_dir="/path/to/output")
result = metric.compute_case(
    case_id="case_001",
    preds_per_fold={0: pred_fold0, 1: pred_fold1, 2: pred_fold2},
    affine=affine_matrix
)

# Or use functions directly
mi_map = compute_mutual_information_wrapper(ensemble_probs)
```

## Available Metrics

- **mutual_information**: Computes mutual information maps (epistemic uncertainty)
- **mean_entropy**: Computes mean entropy maps across folds
- **pairwise_dice**: Computes pairwise Dice scores between folds
- **consensus_segmentation**: Exports consensus segmentation and compares with ground truth

## Future Integration

This module is designed to be extended with metrics from the `values` module. Future versions will support importing additional metrics from `values/evaluation/metrics/`.
