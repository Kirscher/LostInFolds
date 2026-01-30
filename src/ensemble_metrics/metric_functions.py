#!/usr/bin/env python3
"""Core metric computation functions for ensemble predictions."""

from typing import Dict

import nibabel as nib
import numpy as np
from sklearn import preprocessing as sk_preprocess
from sklearn import utils as sk_utils


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


def compute_ncc(gt_unc_map: np.array, pred_unc_map: np.array) -> float:
    """
    Compute the normalized cross correlation between a ground truth uncertainty and a predicted uncertainty map,
    to determine how similar the maps are.
    :param gt_unc_map: the ground truth uncertainty map based on the rater variability
    :param pred_unc_map: the predicted uncertainty map
    :return: float: the normalized cross correlation between gt and predicted uncertainty map
    """
    mu_gt = np.mean(gt_unc_map)
    mu_pred = np.mean(pred_unc_map)
    sigma_gt = np.std(gt_unc_map, ddof=1)
    sigma_pred = np.std(pred_unc_map, ddof=1)
    gt_norm = gt_unc_map - mu_gt
    pred_norm = pred_unc_map - mu_pred
    prod = np.sum(np.multiply(gt_norm, pred_norm))
    ncc = (1 / (np.size(gt_unc_map) * sigma_gt * sigma_pred)) * prod
    return ncc


def get_repeated_interleave(labels: np.ndarray, num_repeats: int) -> np.ndarray:
    """Repeat the labels along the first axis in a "interleave" (from torch.repeat_interleaved) manner.
    That is, given lables l1, l2, l3, they are repeated as l1, l1, ..., l2, l2, ..., l3, l3, ..."""
    return np.repeat(labels, num_repeats, axis=0)


def get_repeated_stacked(labels: np.ndarray, num_repeats: int) -> np.ndarray:
    """Repeat the labels along the first axis in a "stacked" manner.
    That is, given lables l1, l2, l3, they are repeated as l1, l2, l3, l1, l2, l3, ..."""
    return np.tile(labels, (num_repeats, *((labels.ndim - 1) * [1])))


def get_dist_dict_from_dice(dice_dict: Dict[str, float]) -> Dict[str, float]:
    """Convert Dice scores to distance metrics."""
    dist_dict = {}
    for key, dice in dice_dict.items():
        dist = 1.0 - dice
        dist_dict[key] = dist
    return dist_dict


def compute_ged(gt_raters: np.array, ensemble_pred: np.array, num_classes: int, include_background:bool =False) -> Dict[str, float]:
    """Compute Generalized Energy Distance (GED) metric."""
    """
    Input:
        gt_raters: np.ndarray of shape (num_raters, H, W, (D))
        ensemble_pred: np.ndarray of shape (num_folds, H, W, (D)) representing predicted segmentations from each fold
    Output:
        ged: Dict with GED per class and overall GED
    """
    gt_repeat_pred_interleave = get_repeated_interleave(labels=gt_raters, num_repeats=ensemble_pred.shape[0])
    pred_repeat_gt_stacked = get_repeated_stacked(labels=ensemble_pred, num_repeats=gt_raters.shape[0])
    dice_gt_pred = compute_dice(gt_repeat_pred_interleave, pred_repeat_gt_stacked, num_classes=num_classes, include_background=include_background)
    dice_gt_pred["overall_dice"] = float(np.mean(list(dice_gt_pred.values())))
    dist_gt_pred = get_dist_dict_from_dice(dice_gt_pred)

    gt_repeat_gt_interleave = get_repeated_interleave(labels=gt_raters, num_repeats=gt_raters.shape[0])
    gt_repeat_gt_stacked = get_repeated_stacked(labels=gt_raters, num_repeats=gt_raters.shape[0])
    dice_gt_gt = compute_dice(gt_repeat_gt_interleave, gt_repeat_gt_stacked, num_classes=num_classes, include_background=include_background)
    dice_gt_gt["overall_dice"] = float(np.mean(list(dice_gt_gt.values())))
    dist_gt_gt = get_dist_dict_from_dice(dice_gt_gt)

    pred_repeat_pred_interleave = get_repeated_interleave(labels=ensemble_pred, num_repeats=ensemble_pred.shape[0])
    pred_repeat_pred_stacked = get_repeated_stacked(labels=ensemble_pred, num_repeats=ensemble_pred.shape[0])
    dice_pred_pred = compute_dice(pred_repeat_pred_interleave, pred_repeat_pred_stacked, num_classes=num_classes, include_background=include_background)
    dice_pred_pred["overall_dice"] = float(np.mean(list(dice_pred_pred.values())))
    dist_pred_pred = get_dist_dict_from_dice(dice_pred_pred)

    ged_dict = {}
    class_range = range(num_classes) if include_background else range(1, num_classes)
    for c in class_range:
        ged = 2 * dist_gt_pred[f"class_{c}"] - dist_gt_gt[f"class_{c}"] - dist_pred_pred[f"class_{c}"]
        ged_dict[f"class_{c}"] = ged
    ged_dict["overall_ged"] = 2 * dist_gt_pred["overall_dice"] - dist_gt_gt["overall_dice"] - dist_pred_pred["overall_dice"]

    return ged_dict


def get_correct_binary_multirater(gt_raters: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Get binary correctness array for multiple raters."""
    """
    Input:
        gt_raters: np.ndarray of shape (num_raters, H, W, (D))
        pred: np.ndarray of shape (H, W, (D)) representing the average predicted segmentation
    Output:
        correct: flattened np array of shape (num_raters*H*W*(D)) with binary values indicating correctness
    """
    num_raters = gt_raters.shape[0]
    correct = np.zeros_like(gt_raters, dtype=int)
    for i in range(num_raters):
        correct[i] = (gt_raters[i] == pred).astype(int)
    correct = correct.ravel()
    return correct


def get_max_prob_for_pred_classes(probs_per_fold: Dict[int, np.ndarray], consensus_pred: np.ndarray) -> np.ndarray:
    """Get maximum predicted probability for the consensus predicted classes."""
    probs_pred_class = []
    for fold_probs in probs_per_fold.values():
        consensus_pred = consensus_pred.astype(int)
        probs = np.take_along_axis(
            fold_probs,
            consensus_pred[np.newaxis, ...],
            axis=0
        ).squeeze(axis=0)
        probs_pred_class.append(probs)
    probs_pred_class = np.stack(probs_pred_class, axis=0)
    max_probs_pred_class = np.max(probs_pred_class, axis=0)
    return max_probs_pred_class


def calib_stats(correct, calib_confids, n_bins=20):
    """Calculate calibration statistics."""
    y_true = sk_utils.column_or_1d(correct)
    y_prob = sk_utils.column_or_1d(calib_confids)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "y_prob has values outside [0, 1] and normalize is " "set to False."
        )

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. " f"Provided labels {labels}."
        )
    y_true = sk_preprocess.label_binarize(y_true, classes=labels)[:, 0]

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    num_nonzero = len(nonzero[nonzero == True])
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    prob_total = bin_total[nonzero] / bin_total.sum()

    bin_discrepancies = np.abs(prob_true - prob_pred)
    return bin_discrepancies, prob_total, num_nonzero


def compute_ace(correct, calib_confids, n_bins=20):
    bin_discrepancies, _, num_nonzero = calib_stats(correct, calib_confids, n_bins=n_bins)
    return (1 / num_nonzero) * np.sum(bin_discrepancies)