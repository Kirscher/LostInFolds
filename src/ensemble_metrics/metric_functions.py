#!/usr/bin/env python3
"""Core metric computation functions for ensemble predictions."""

import ctypes
import gc
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
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


def compute_ncc(gt_unc_map: np.ndarray, pred_unc_map: np.ndarray) -> float:
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
    
    # Handle division by zero: if either array is constant, return 0 or NaN
    if sigma_gt == 0 or sigma_pred == 0:
        return 0.0
    
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


def _mean_pairwise_dice(set_a: np.ndarray, set_b: np.ndarray, num_classes: int, include_background: bool = False) -> Dict[str, float]:
    """Compute mean Dice across all (i, j) pairs from set_a and set_b iteratively (memory-efficient)."""
    class_range = list(range(num_classes) if include_background else range(1, num_classes))
    n_a, n_b = set_a.shape[0], set_b.shape[0]
    accum = {f"class_{c}": 0.0 for c in class_range}
    n_pairs = n_a * n_b
    for i in range(n_a):
        for j in range(n_b):
            pair_dice = compute_dice(set_a[i], set_b[j], num_classes=num_classes, include_background=include_background)
            for c in class_range:
                accum[f"class_{c}"] += pair_dice[f"class_{c}"]
    mean_dice = {k: v / n_pairs for k, v in accum.items()}
    mean_dice["overall_dice"] = float(np.mean(list(mean_dice.values())))
    return mean_dice


def compute_ged(gt_raters: np.ndarray, ensemble_pred: np.ndarray, num_classes: int, include_background: bool = False) -> Dict[str, float]:
    """Compute Generalized Energy Distance (GED) metric.

    Memory-efficient version: iterates over pairs instead of expanding
    all combinations into giant tiled arrays.

    Input:
        gt_raters: np.ndarray of shape (num_raters, H, W, (D))
        ensemble_pred: np.ndarray of shape (num_folds, H, W, (D))
    Output:
        ged: Dict with GED per class and overall GED
    """
    dice_gt_pred = _mean_pairwise_dice(gt_raters, ensemble_pred, num_classes, include_background)
    dist_gt_pred = get_dist_dict_from_dice(dice_gt_pred)

    dice_gt_gt = _mean_pairwise_dice(gt_raters, gt_raters, num_classes, include_background)
    dist_gt_gt = get_dist_dict_from_dice(dice_gt_gt)

    dice_pred_pred = _mean_pairwise_dice(ensemble_pred, ensemble_pred, num_classes, include_background)
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
    consensus_pred_int = consensus_pred.astype(int)
    for fold_probs in probs_per_fold.values():
        probs = np.take_along_axis(
            fold_probs,
            consensus_pred_int[np.newaxis, ...],
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


# ------------------------------------------------------------
# Boundary & distance utilities
# ------------------------------------------------------------

def find_boundary_mask(labels: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask of boundary pixels.

    A pixel is boundary if any neighbor has different label.
    Works for 2D or ND arrays.
    """
    labels = np.asarray(labels)
    boundary = np.zeros_like(labels, dtype=bool)

    for axis in range(labels.ndim):
        diff = np.diff(labels, axis=axis)
        pad1 = [(0, 0)] * labels.ndim
        pad2 = [(0, 0)] * labels.ndim

        pad1[axis] = (1, 0)
        pad2[axis] = (0, 1)

        boundary |= np.pad(diff != 0, pad1)
        boundary |= np.pad(diff != 0, pad2)

    return boundary


def unsigned_distance_to_boundary(labels: np.ndarray) -> np.ndarray:
    """
    Distance from each pixel to nearest ground-truth boundary.
    """
    boundary = find_boundary_mask(labels)
    # distance_transform_edt computes distance to nearest zero
    dist = distance_transform_edt(~boundary)
    return dist


def ring_band_masks(
    dist: np.ndarray,
    edges: List[float],
) -> List[np.ndarray]:
    """
    Creates masks for distance rings:
    [edges[i], edges[i+1])
    """
    bands = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        bands.append((dist >= lo) & (dist < hi))
    return bands


# ------------------------------------------------------------
# BA-ECE
# ------------------------------------------------------------

def compute_ba_ece(
    confidence: np.ndarray,
    labels: np.ndarray,
    pred_labels: np.ndarray,
    edges: Union[List[float], Tuple[float, ...]] = (0, 3, 7, 15, np.inf),
) -> Dict[str, object]:
    """
    Boundary-Aware Expected Calibration Error (BA-ECE)

    Parameters
    ----------
    confidence :
        Confidence of the prediction.
    labels : (n_raters, ...)
        Ground-truth labels. Contains multiple raters along first axis.
    pred_labels : optional (...)
        Predicted class labels.
    edges : distance band edges.

    Returns
    -------
    dict with:
        ba_ece : float
        bands : list of per-band dicts
        counts : list of counts per band
    """

    # Distance to GT boundary
    dist = []
    bands = []
    for rater_idx in range(labels.shape[0]):
        dist_rater = unsigned_distance_to_boundary(labels[rater_idx])
        bands_rater = ring_band_masks(dist_rater, list(edges))
        dist.append(dist_rater)
        bands.append(bands_rater)
    dist = np.stack(dist, axis=0)
    bands = np.stack(bands, axis=0)

    confidence = confidence.reshape(-1)
    y = labels.reshape(-1).astype(int)
    pred = pred_labels.reshape(-1).astype(int)
    dist_flat = dist.reshape(-1)

    # Uncertainty = 1 - confidence
    # confidence = P.max(axis=1)
    uncertainty = 1.0 - confidence

    error = (pred != y).astype(np.float32)

    total_weight = 0.0
    weighted_sum = 0.0

    band_results = []
    counts = []

    for i, _ in enumerate(bands[1]):
        mask = bands[:, i, ...]
        sel = mask.reshape(-1)
        cnt = int(sel.sum())
        counts.append(cnt)

        if cnt == 0:
            band_results.append(
                dict(
                    band=i,
                    count=0,
                    mean_uncertainty=np.nan,
                    mean_error=np.nan,
                    calibration_error=np.nan,
                    mean_distance=np.nan,
                    weight=0.0,
                )
            )
            continue

        mean_unc = float(uncertainty[sel].mean())
        mean_err = float(error[sel].mean())
        cal_err = abs(mean_unc - mean_err)
        mean_dist = float(dist_flat[sel].mean())

        weight = 1.0 / (mean_dist + 1.0)

        total_weight += weight
        weighted_sum += weight * cal_err

        band_results.append(
            dict(
                band=i,
                count=cnt,
                mean_uncertainty=mean_unc,
                mean_error=mean_err,
                calibration_error=cal_err,
                mean_distance=mean_dist,
                weight=weight,
            )
        )

    ba = float(weighted_sum / max(total_weight, 1e-12))

    return {
        "ba_ece": ba,
        "bands": band_results,
        "counts": counts,
    }


def compute_ba_ece_streaming(
    confidence: np.ndarray,
    labels: np.ndarray,
    pred_labels: np.ndarray,
    edges: Union[List[float], Tuple[float, ...]] = (0, 3, 7, 15, np.inf),
) -> Dict[str, object]:
    """Memory-efficient BA-ECE that streams one rater at a time.

    Instead of stacking (n_raters, ...) distance and band arrays
    this processes each rater
    independently and accumulates band statistics incrementally.

    Parameters
    ----------
    confidence : (H, W, D) — single confidence map (not tiled over raters)
    labels : (n_raters, H, W, D) — per-rater GT labels
    pred_labels : (H, W, D) — predicted class labels
    edges : distance band edges
    """
    n_raters = labels.shape[0]
    n_bands = len(edges) - 1
    edge_list = list(edges)

    # Per-band accumulators (streaming across raters)
    band_unc_sum = np.zeros(n_bands, dtype=np.float64)
    band_err_sum = np.zeros(n_bands, dtype=np.float64)
    band_dist_sum = np.zeros(n_bands, dtype=np.float64)
    band_count = np.zeros(n_bands, dtype=np.int64)

    conf_flat = confidence.ravel().astype(np.float32)  # (V,)
    pred_flat = pred_labels.ravel()  # (V,)

    for r_idx in range(n_raters):
        rater = labels[r_idx]
        dist_r = unsigned_distance_to_boundary(rater)
        rater_flat = rater.ravel()
        dist_flat = dist_r.ravel()

        error_r = (pred_flat != rater_flat).astype(np.float32)
        unc_r = 1.0 - conf_flat

        for b in range(n_bands):
            lo, hi = edge_list[b], edge_list[b + 1]
            sel = (dist_flat >= lo) & (dist_flat < hi)
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            band_count[b] += cnt
            band_unc_sum[b] += float(unc_r[sel].sum())
            band_err_sum[b] += float(error_r[sel].sum())
            band_dist_sum[b] += float(dist_flat[sel].sum())
            del sel

        del dist_r, rater_flat, dist_flat, error_r, unc_r
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    # Aggregate bands
    total_weight = 0.0
    weighted_sum = 0.0
    band_results = []
    counts_list = []

    for b in range(n_bands):
        cnt = int(band_count[b])
        counts_list.append(cnt)
        if cnt == 0:
            band_results.append(dict(
                band=b, count=0,
                mean_uncertainty=np.nan, mean_error=np.nan,
                calibration_error=np.nan, mean_distance=np.nan,
                weight=0.0,
            ))
            continue

        mean_unc = band_unc_sum[b] / cnt
        mean_err = band_err_sum[b] / cnt
        cal_err = abs(mean_unc - mean_err)
        mean_dist = band_dist_sum[b] / cnt

        weight = 1.0 / (mean_dist + 1.0)
        total_weight += weight
        weighted_sum += weight * cal_err

        band_results.append(dict(
            band=b, count=cnt,
            mean_uncertainty=float(mean_unc),
            mean_error=float(mean_err),
            calibration_error=float(cal_err),
            mean_distance=float(mean_dist),
            weight=float(weight),
        ))

    ba = float(weighted_sum / max(total_weight, 1e-12))
    return {"ba_ece": ba, "bands": band_results, "counts": counts_list}


def rc_curve_stats(
    risks: np.ndarray, confids: np.ndarray
) -> tuple[list[float], list[float], list[float]]:
    coverages = []
    selective_risks = []
    assert (
        len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids)
    )

    n_samples = len(risks)
    if n_samples == 0:
        return [], [], []
    idx_sorted = np.argsort(confids)

    coverage = n_samples
    error_sum = sum(risks[idx_sorted])

    coverages.append(coverage / n_samples)
    selective_risks.append(error_sum / n_samples)

    weights = []

    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        coverage = coverage - 1
        error_sum = error_sum - risks[idx_sorted[i]]
        tmp_weight += 1
        if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
            coverages.append(coverage / n_samples)
            selective_risks.append(error_sum / (n_samples - 1 - i))
            weights.append(tmp_weight / n_samples)
            tmp_weight = 0

    # add a well-defined final point to the RC-curve.
    if tmp_weight > 0:
        coverages.append(0)
        selective_risks.append(selective_risks[-1])
        weights.append(tmp_weight / n_samples)

    return coverages, selective_risks, weights


def compute_aurc(risks: np.ndarray, confids: np.ndarray):
    _, risks, weights = rc_curve_stats(risks, confids)
    return sum(
        [(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]
    )