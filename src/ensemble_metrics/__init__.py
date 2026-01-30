#!/usr/bin/env python3
"""Ensemble metrics module for nnUNet multi-fold predictions."""

# Core metric functions
from .metric_functions import (compute_dice, compute_ensemble_entropy,
                               compute_entropy_map,
                               compute_mutual_information_wrapper, load_array)
# Metric classes
from .metrics import (ACEMeric, AURCMetric, BAECEMetric, BaseMetric,
                      ConsensusSegmentationMetric, ExpectedEntropyMetric,
                      GEDMetric, METRICS, MutualInformationMetric,
                      NCCMetric, PairwiseDiceMetric,
                      PredictiveEntropyMetric)
# Utility functions
from .utils import (discover_cases, discover_folds, load_ground_truth,
                    load_prediction, one_hot_encode, standardize_prediction)

__all__ = [
    # Metric functions
    "load_array",
    "compute_entropy_map",
    "compute_mutual_information_wrapper",
    "compute_ensemble_entropy",
    "compute_dice",
    # Metric classes
    "BaseMetric",
    "PredictiveEntropyMetric",
    "MutualInformationMetric",
    "ExpectedEntropyMetric",
    "PairwiseDiceMetric",
    "ConsensusSegmentationMetric",
    "NCCMetric",
    "ACEMeric",
    "GEDMetric",
    "AURCMetric",
    "BAECEMetric",
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
