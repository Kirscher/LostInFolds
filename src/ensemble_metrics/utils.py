#!/usr/bin/env python3
"""Utility functions for ensemble metrics computation."""

import glob
import os
import re
from functools import lru_cache, reduce
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from .metric_functions import load_array

_affine_cache: Dict[str, np.ndarray] = {}


def discover_folds(pred_root: str, num_folds: Optional[int] = None) -> List[Tuple[int, str]]:
    """Discover fold directories in prediction root."""
    fold_dirs = []
    pattern = re.compile(r'fold[_-]?(\d+)', re.IGNORECASE)
    
    for item in os.listdir(pred_root):
        item_path = os.path.join(pred_root, item)
        if os.path.isdir(item_path):
            match = pattern.match(item)
            if match:
                fold_idx = int(match.group(1))
                fold_dirs.append((fold_idx, item_path))
    
    fold_dirs.sort(key=lambda x: x[0])
    
    if num_folds is not None:
        fold_dirs = fold_dirs[:num_folds]
    
    return fold_dirs


def discover_cases(fold_paths: List[str]) -> List[str]:
    """Discover case IDs by intersecting files across all folds."""
    case_sets = []
    suffixes = {'.npz', '.nii.gz'}
    
    for fold_path in fold_paths:
        case_ids = set()
        try:
            for filename in os.listdir(fold_path):
                if any(filename.endswith(suffix) for suffix in suffixes):
                    case_id = filename[:-7] if filename.endswith('.nii.gz') else filename[:-4]
                    case_ids.add(case_id)
        except OSError:
            continue
        case_sets.append(case_ids)
    
    if not case_sets:
        return []
    
    return sorted(list(reduce(set.intersection, case_sets)))


def _get_affine_from_fold(fold_path: str) -> Optional[np.ndarray]:
    """Get affine matrix from any .nii.gz file in fold."""
    if fold_path in _affine_cache:
        return _affine_cache[fold_path]
    
    nii_files = glob.glob(os.path.join(fold_path, "*.nii.gz"))
    if nii_files:
        img = nib.load(nii_files[0])
        affine = img.affine
        _affine_cache[fold_path] = affine
        return affine
    return None


def load_prediction(fold_path: str, case_id: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load prediction for a case from a fold directory."""
    npz_path = os.path.join(fold_path, f"{case_id}.npz")
    if os.path.exists(npz_path):
        data = load_array(npz_path, is_ensemble=False)
        data = data.swapaxes(1, -1)
        nii_ref = os.path.join(fold_path, f"{case_id}.nii.gz")
        if os.path.exists(nii_ref):
            affine = nib.load(nii_ref).affine
        else:
            affine = _get_affine_from_fold(fold_path)
        return data, affine
    
    nii_path = os.path.join(fold_path, f"{case_id}.nii.gz")
    if os.path.exists(nii_path):
        img = nib.load(nii_path)
        return img.get_fdata(), img.affine
    
    raise FileNotFoundError(f"Prediction file not found for case {case_id} in {fold_path}")


def calculate_majority_consensus(raters: List[np.ndarray]) -> np.ndarray:
    """Calculate majority consensus segmentation from multiple raters."""
    raters_flattened = [rater.ravel() for rater in raters]
    raters_stacked = np.stack(raters_flattened, axis=0)
    majority_flat = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=raters_stacked
    )
    majority_consensus = majority_flat.reshape(raters[0].shape)
    return majority_consensus

def load_ground_truth(gt_dir: str, case_id: str, num_raters: int, consensus_type: str=None) -> Optional[Tuple[Dict, np.ndarray]]:
    """Load ground truth for a case."""
    if num_raters == 1:
        gt_path = os.path.join(gt_dir, f"{case_id}.nii.gz")
        if os.path.exists(gt_path):
            img = nib.load(gt_path)
            gt = {
                "raters": img.get_fdata().astype(np.int32).expand_dims(axis=0),
                "consensus": img.get_fdata().astype(np.int32),
            }
            return gt, img.affine
        else:
            return None

    gt_files = [f"{case_id}_{i:02d}.nii.gz" for i in range(1, num_raters + 1)]
    # if consensus_type is not None:
    #     gt_files.append(f"{case_id}_{consensus_type}.nii.gz")

    label_stacked = []
    affine_stacked = []
    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        if os.path.exists(gt_path):
            label = nib.load(gt_path)
            label_stacked.append(label.get_fdata().astype(np.int32))
            affine_stacked.append(label.affine)
        else:
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    if consensus_type is not None:
        consensus_file = f"{case_id}_{consensus_type}.nii.gz"
        consensus_path = os.path.join(gt_dir, consensus_file)
        if os.path.exists(consensus_path):
            consensus_label = nib.load(consensus_path).get_fdata().astype(np.int32)
        else:
            raise FileNotFoundError(f"Consensus ground truth file not found: {consensus_path}")
    else:
        consensus_label = calculate_majority_consensus(label_stacked)
    
    gt = {
        "raters": np.stack(label_stacked, axis=0),
        "consensus": consensus_label,
    }

    return gt, affine_stacked[0]


def standardize_prediction(pred: np.ndarray) -> np.ndarray:
    """Standardize prediction to (num_classes, ...) format."""
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    
    if pred.ndim >= 2 and pred.shape[0] in [2, 3, 4, 5, 6, 7, 8]:
        if pred.ndim >= 3:
            flat_pred = pred.reshape(pred.shape[0], -1)
            n_check = min(1000, flat_pred.shape[1])
            sample_sums = np.sum(flat_pred[:, :n_check], axis=0)
            if np.mean(np.abs(sample_sums - 1.0)) < 0.1:
                return pred
            else:
                labels = np.argmax(pred, axis=0) if pred.shape[0] > 1 else pred[0]
                return one_hot_encode(labels, int(np.max(labels)) + 1)
        return pred
    
    if pred.ndim in [2, 3]:
        return one_hot_encode(pred, int(np.max(pred)) + 1)
    
    return pred


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert label array to one-hot encoded probabilities."""
    shape = labels.shape
    flat_labels = labels.flatten()
    one_hot = np.zeros((num_classes, flat_labels.size), dtype=np.float32)
    one_hot[flat_labels, np.arange(flat_labels.size)] = 1.0
    return one_hot.reshape((num_classes, *shape))
