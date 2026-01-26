#!/usr/bin/env python3
"""Ensemble metrics module for nnUNet multi-fold predictions."""

# Core metric functions
from .metric_functions import (
    load_array,
    compute_entropy_map,
    compute_mutual_information_wrapper,
    compute_ensemble_entropy,
    compute_dice,
)

# Metric classes
from .metrics import (
    BaseMetric,
    MutualInformationMetric,
    MeanEntropyMetric,
    PairwiseDiceMetric,
    ConsensusSegmentationMetric,
    METRICS,
)

# Utility functions
from .utils import (
    discover_folds,
    discover_cases,
    load_prediction,
    load_ground_truth,
    standardize_prediction,
    one_hot_encode,
)

__all__ = [
    # Metric functions
    "load_array",
    "compute_entropy_map",
    "compute_mutual_information_wrapper",
    "compute_ensemble_entropy",
    "compute_dice",
    # Metric classes
    "BaseMetric",
    "MutualInformationMetric",
    "MeanEntropyMetric",
    "PairwiseDiceMetric",
    "ConsensusSegmentationMetric",
    "METRICS",
    # Utility functions
    "discover_folds",
    "discover_cases",
    "load_prediction",
    "load_ground_truth",
    "standardize_prediction",
    "one_hot_encode",
]

__version__ = "0.1.0"
