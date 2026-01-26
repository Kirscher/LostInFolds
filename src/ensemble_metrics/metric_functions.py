#!/usr/bin/env python3
"""Core metric computation functions for ensemble predictions."""

import numpy as np
import nibabel as nib


def load_array(filepath: str, is_ensemble: bool = False) -> np.ndarray:
    """Load array from .npz file (nnUNet format)."""
    data = np.load(filepath)
    if is_ensemble:
        key = 'softmax'
    else:
        key = 'probabilities'
    
    if key not in data:
        # Try alternative keys
        for alt_key in ['probabilities', 'softmax', 'data']:
            if alt_key in data:
                return data[alt_key]
        raise KeyError(f"Expected key '{key}' not found in {filepath}. Available keys: {list(data.keys())}")
    
    return data[key]


def compute_entropy_map(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute entropy map from probability array."""
    probs = np.clip(probs, eps, 1.0)
    probs = probs / (np.sum(probs, axis=0, keepdims=True) + eps)
    log_probs = np.log(probs + eps)
    entropy = -np.sum(probs * log_probs, axis=0)
    return entropy


def compute_mutual_information_wrapper(ensemble_probs: np.ndarray) -> np.ndarray:
    """Compute mutual information map: Predictive Entropy - Expected Entropy."""
    num_folds = ensemble_probs.shape[0]
    mean_probs = np.mean(ensemble_probs, axis=0)
    pred_entropy = compute_entropy_map(mean_probs)
    expected_entropy = np.mean([compute_entropy_map(ensemble_probs[i]) for i in range(num_folds)], axis=0)
    mi = pred_entropy - expected_entropy
    return np.maximum(mi, 0.0)


def compute_ensemble_entropy(ensemble_probs: np.ndarray) -> np.ndarray:
    """Compute mean entropy across folds."""
    num_folds = ensemble_probs.shape[0]
    entropies = [compute_entropy_map(ensemble_probs[i]) for i in range(num_folds)]
    return np.mean(entropies, axis=0)


def compute_dice(
    gt: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
    include_background: bool = False
) -> dict:
    """Compute Dice scores for each class."""
    dice_scores = {}
    class_range = range(num_classes) if include_background else range(1, num_classes)
    
    for c in class_range:
        gt_c = (gt == c)
        pred_c = (pred == c)
        intersection = np.sum(gt_c & pred_c)
        union = np.sum(gt_c) + np.sum(pred_c)
        dice = 1.0 if union == 0 else 2.0 * intersection / union
        dice_scores[f"class_{c}"] = float(dice)
    
    return dice_scores
