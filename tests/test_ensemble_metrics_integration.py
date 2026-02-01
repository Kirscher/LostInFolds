#!/usr/bin/env python3
"""Comprehensive integration tests for LostInFolds ensemble metrics."""

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from src.ensemble_metrics.metrics import (
    ACEMeric,
    AURCMetric,
    BAECEMetric,
    ConsensusSegmentationMetric,
    ExpectedEntropyMetric,
    GEDMetric,
    MutualInformationMetric,
    NCCMetric,
    PairwiseDiceMetric,
    PredictiveEntropyMetric,
)


def create_dummy_affine() -> np.ndarray:
    """Create identity affine matrix for NIfTI output."""
    return np.eye(4)


def create_dummy_multirater_labels(
    num_raters: int = 3,
    num_classes: int = 4,
    spatial_shape: tuple = (32, 32, 32),
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic multirater ground truth labels.
    
    Args:
        num_raters: Number of raters
        num_classes: Number of classes (including background)
        spatial_shape: Spatial dimensions (H, W, D)
        seed: Random seed for reproducibility
        
    Returns:
        Dict with "raters" key containing array of shape (num_raters, H, W, D)
        and "consensus" key containing majority vote consensus
    """
    np.random.seed(seed)
    H, W, D = spatial_shape
    
    # Create base structure with some spatial coherence
    base_labels = np.zeros((H, W, D), dtype=np.int32)
    
    # Add some structured regions
    # Region 1: Class 1 in center
    center_h, center_w, center_d = H // 2, W // 2, D // 2
    radius = 8
    y, x, z = np.ogrid[:H, :W, :D]
    mask1 = (x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2 <= radius**2
    base_labels[mask1] = 1
    
    # Region 2: Class 2 in corner
    corner_mask = (x < H // 4) & (y < W // 4) & (z < D // 4)
    base_labels[corner_mask] = 2
    
    # Region 3: Class 3 in another corner
    corner_mask2 = (x > 3 * H // 4) & (y > 3 * W // 4) & (z > 3 * D // 4)
    base_labels[corner_mask2] = 3
    
    # Generate rater labels with some agreement and disagreement
    raters = []
    for r in range(num_raters):
        rater_labels = base_labels.copy()
        
        # Add some rater-specific variations
        # Rater 0: More conservative (smaller regions)
        if r == 0:
            # Shrink class 1 region slightly
            shrink_mask = (x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2 <= (radius - 2)**2
            rater_labels[~shrink_mask & (rater_labels == 1)] = 0
        # Rater 1: More aggressive (larger regions)
        elif r == 1:
            # Expand class 1 region slightly
            expand_mask = (x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2 <= (radius + 2)**2
            rater_labels[expand_mask & (rater_labels == 0)] = 1
        # Rater 2: Some mislabeling
        else:
            # Randomly change some pixels
            noise_mask = np.random.random((H, W, D)) < 0.05
            rater_labels[noise_mask] = np.random.randint(0, num_classes, size=np.sum(noise_mask))
        
        raters.append(rater_labels)
    
    raters_array = np.stack(raters, axis=0)
    
    # Compute majority vote consensus
    from src.ensemble_metrics.utils import calculate_majority_consensus
    consensus = calculate_majority_consensus([raters_array[i] for i in range(num_raters)])
    
    return {
        "raters": raters_array.astype(np.int32),
        "consensus": consensus.astype(np.int32),
    }


def create_dummy_ensemble_probs(
    num_folds: int = 5,
    num_classes: int = 4,
    spatial_shape: tuple = (32, 32, 32),
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """
    Generate synthetic ensemble probability predictions.
    
    Args:
        num_folds: Number of folds in ensemble
        num_classes: Number of classes
        spatial_shape: Spatial dimensions (H, W, D)
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping fold index to probability array of shape (num_classes, H, W, D)
    """
    np.random.seed(seed)
    H, W, D = spatial_shape
    
    preds_per_fold = {}
    
    for fold_idx in range(num_folds):
        # Create base probabilities with some structure
        probs = np.zeros((num_classes, H, W, D), dtype=np.float32)
        
        # Add some spatial structure matching the GT regions
        center_h, center_w, center_d = H // 2, W // 2, D // 2
        y, x, z = np.ogrid[:H, :W, :D]
        
        # Class 1: High probability in center
        dist_center = np.sqrt((x - center_w)**2 + (y - center_h)**2 + (z - center_d)**2)
        probs[1] = np.exp(-dist_center / 10.0)
        
        # Class 2: High probability in corner
        dist_corner = np.sqrt((x - H // 4)**2 + (y - W // 4)**2 + (z - D // 4)**2)
        probs[2] = np.exp(-dist_corner / 8.0)
        
        # Class 3: High probability in another corner
        dist_corner2 = np.sqrt((x - 3 * H // 4)**2 + (y - 3 * W // 4)**2 + (z - 3 * D // 4)**2)
        probs[3] = np.exp(-dist_corner2 / 8.0)
        
        # Background: inverse of foreground
        probs[0] = 1.0 - (probs[1] + probs[2] + probs[3])
        probs[0] = np.clip(probs[0], 0.1, 1.0)
        
        # Add fold-specific variation
        variation = np.random.normal(0, 0.1, size=(num_classes, H, W, D))
        probs += variation
        probs = np.clip(probs, 0.0, 1.0)
        
        # Normalize to ensure probabilities sum to 1
        probs = probs / (np.sum(probs, axis=0, keepdims=True) + 1e-10)
        
        preds_per_fold[fold_idx] = probs.astype(np.float32)
    
    return preds_per_fold


@pytest.fixture
def dummy_data():
    """Fixture providing dummy multirater labels and ensemble probabilities."""
    num_raters = 3
    num_classes = 4
    spatial_shape = (32, 32, 32)
    
    multirater_gt = create_dummy_multirater_labels(
        num_raters=num_raters,
        num_classes=num_classes,
        spatial_shape=spatial_shape,
    )
    
    ensemble_probs = create_dummy_ensemble_probs(
        num_folds=5,
        num_classes=num_classes,
        spatial_shape=spatial_shape,
    )
    
    return {
        "multirater_gt": multirater_gt,
        "ensemble_probs": ensemble_probs,
        "num_classes": num_classes,
        "spatial_shape": spatial_shape,
    }


def test_standard_metrics_without_gt(dummy_data, tmp_path):
    """Test metrics that don't require ground truth."""
    preds_per_fold = dummy_data["ensemble_probs"]
    affine = create_dummy_affine()
    case_id = "test_case_001"
    case_output_dir = tmp_path / case_id
    case_output_dir.mkdir()
    
    # Test PredictiveEntropyMetric
    metric = PredictiveEntropyMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=None,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    assert "predictive_entropy_mean" in result
    assert "predictive_entropy_std" in result
    assert "predictive_entropy_max" in result
    assert result["predictive_entropy_mean"] >= 0
    assert result["predictive_entropy_max"] >= result["predictive_entropy_mean"]
    assert (case_output_dir / "predictive_entropy_map.nii.gz").exists()
    
    # Test MutualInformationMetric
    metric = MutualInformationMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=None,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    assert "mi_mean" in result
    assert "mi_std" in result
    assert "mi_max" in result
    assert result["mi_mean"] >= 0
    assert result["mi_max"] >= result["mi_mean"]
    assert (case_output_dir / "mi_map.nii.gz").exists()
    
    # Test ExpectedEntropyMetric
    metric = ExpectedEntropyMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=None,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    assert "expected_entropy_mean" in result
    assert "expected_entropy_std" in result
    assert "expected_entropy_max" in result
    assert result["expected_entropy_mean"] >= 0
    assert result["expected_entropy_max"] >= result["expected_entropy_mean"]
    assert (case_output_dir / "expected_entropy_map.nii.gz").exists()
    
    
    # Test PairwiseDiceMetric
    metric = PairwiseDiceMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=None,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    metric.export_summaries()
    assert (tmp_path / "pairwise_dice_per_case.csv").exists()
    assert (tmp_path / "pairwise_dice_summary.csv").exists()


def test_standard_metrics_with_single_gt(dummy_data, tmp_path):
    """Test ConsensusSegmentationMetric with single ground truth."""
    preds_per_fold = dummy_data["ensemble_probs"]
    multirater_gt = dummy_data["multirater_gt"]
    affine = create_dummy_affine()
    case_id = "test_case_002"
    case_output_dir = tmp_path / case_id
    case_output_dir.mkdir()
    
    # Use consensus as single GT - pass as dict for compatibility
    single_gt_dict = {"consensus": multirater_gt["consensus"]}
    
    metric = ConsensusSegmentationMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=single_gt_dict,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    assert (case_output_dir / "consensus_seg.nii.gz").exists()
    
    metric.export_summaries()
    assert (tmp_path / "dice_vs_gt_per_case.csv").exists()
    assert (tmp_path / "dice_vs_gt_summary.csv").exists()


def test_multirater_metrics(dummy_data, tmp_path):
    """Test all multirater-specific metrics."""
    preds_per_fold = dummy_data["ensemble_probs"]
    multirater_gt = dummy_data["multirater_gt"]
    affine = create_dummy_affine()
    case_id = "test_case_003"
    case_output_dir = tmp_path / case_id
    case_output_dir.mkdir()
    
    # First, compute expected entropy (required by NCC)
    expected_entropy_metric = ExpectedEntropyMetric(output_dir=str(tmp_path))
    expected_entropy_metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=None,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    
    # Compute consensus segmentation (required by ACE, BA-ECE)
    consensus_metric = ConsensusSegmentationMetric(output_dir=str(tmp_path))
    consensus_metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=multirater_gt,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    
    # Test GEDMetric
    metric = GEDMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=multirater_gt,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    metric.export_summaries()
    assert (tmp_path / "ged.csv").exists()
    # Verify GED values are reasonable
    import pandas as pd
    ged_df = pd.read_csv(tmp_path / "ged.csv")
    assert "overall_ged" in ged_df.columns
    assert len(ged_df) > 0
    
    # Test NCCMetric
    metric = NCCMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=multirater_gt,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    metric.export_summaries()
    assert (tmp_path / "ncc.csv").exists()
    # Verify NCC values are reasonable (should be in [-1, 1] range)
    import pandas as pd
    ncc_df = pd.read_csv(tmp_path / "ncc.csv")
    assert "ncc" in ncc_df.columns
    assert len(ncc_df) > 0
    # NCC should be between -1 and 1
    ncc_values = ncc_df["ncc"].dropna()
    if len(ncc_values) > 0:
        assert all(ncc_values >= -1.0) and all(ncc_values <= 1.0)
    
    # Test ACEMeric
    metric = ACEMeric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=multirater_gt,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    metric.export_summaries()
    assert (tmp_path / "ace.csv").exists()
    # Verify ACE values are reasonable (should be >= 0)
    import pandas as pd
    ace_df = pd.read_csv(tmp_path / "ace.csv")
    assert "ace" in ace_df.columns
    assert len(ace_df) > 0
    ace_values = ace_df["ace"].dropna()
    if len(ace_values) > 0:
        assert all(ace_values >= 0.0)
    
    # Test BAECEMetric
    metric = BAECEMetric(output_dir=str(tmp_path))
    result = metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=multirater_gt,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    assert "case_id" in result
    metric.export_summaries()
    assert (tmp_path / "ba_ece.csv").exists()
    # Verify CSV contains ba_ece values
    import pandas as pd
    baece_df = pd.read_csv(tmp_path / "ba_ece.csv")
    assert "ba_ece" in baece_df.columns
    assert len(baece_df) > 0


def test_all_metrics_integration(dummy_data, tmp_path):
    """Integration test using all metrics together."""
    preds_per_fold = dummy_data["ensemble_probs"]
    multirater_gt = dummy_data["multirater_gt"]
    affine = create_dummy_affine()
    case_id = "test_case_integration"
    case_output_dir = tmp_path / case_id
    case_output_dir.mkdir()
    
    # Create output directory for this case
    case_output_dir.mkdir(exist_ok=True)
    
    # Step 1: Compute metrics that don't require dependencies
    metrics_to_run = [
        ("predictive_entropy", PredictiveEntropyMetric),
        ("mutual_information", MutualInformationMetric),
        ("expected_entropy", ExpectedEntropyMetric),
        ("pairwise_dice", PairwiseDiceMetric),
    ]
    
    for name, MetricClass in metrics_to_run:
        metric = MetricClass(output_dir=str(tmp_path))
        result = metric.compute_case(
            case_id=case_id,
            preds_per_fold=preds_per_fold,
            gt=None,
            affine=affine,
            case_output_dir=str(case_output_dir),
        )
        assert "case_id" in result
        metric.export_summaries()
    
    # Step 2: Compute consensus segmentation (needed for some metrics)
    consensus_metric = ConsensusSegmentationMetric(output_dir=str(tmp_path))
    consensus_metric.compute_case(
        case_id=case_id,
        preds_per_fold=preds_per_fold,
        gt=multirater_gt,
        affine=affine,
        case_output_dir=str(case_output_dir),
    )
    consensus_metric.export_summaries()
    
    # Step 3: Create required CSV files for AURC
    # AURC needs dice_vs_gt.csv and pairwise_dice.csv
    import pandas as pd
    
    # Create dummy dice_vs_gt.csv
    dice_data = {
        "case_id": [case_id],
        "consensus_overall_dice": [0.85],
        "consensus_class_1": [0.80],
        "consensus_class_2": [0.90],
        "consensus_class_3": [0.85],
    }
    dice_df = pd.DataFrame(dice_data)
    dice_df.to_csv(case_output_dir / "dice_vs_gt.csv", index=False)
    
    # Create dummy pairwise_dice.csv
    pairwise_data = {
        "case_id": [case_id],
        "fold_i": [0],
        "fold_j": [1],
        "overall_dice": [0.82],
        "class_1": [0.78],
        "class_2": [0.88],
        "class_3": [0.80],
    }
    pairwise_df = pd.DataFrame(pairwise_data)
    pairwise_df.to_csv(case_output_dir / "pairwise_dice.csv", index=False)
    
    # Step 4: Compute multirater metrics
    multirater_metrics = [
        ("ged", GEDMetric),
        ("ncc", NCCMetric),
        ("ace", ACEMeric),
        ("ba_ece", BAECEMetric),
        ("aurc", AURCMetric),
    ]
    
    for name, MetricClass in multirater_metrics:
        metric = MetricClass(output_dir=str(tmp_path))
        result = metric.compute_case(
            case_id=case_id,
            preds_per_fold=preds_per_fold,
            gt=multirater_gt,
            affine=affine,
            case_output_dir=str(case_output_dir),
        )
        assert "case_id" in result
        metric.export_summaries()
    
    # Verify all output files were created
    expected_files = [
        "predictive_entropy_map.nii.gz",
        "mi_map.nii.gz",
        "expected_entropy_map.nii.gz",
        "consensus_seg.nii.gz",
    ]
    
    for filename in expected_files:
        assert (case_output_dir / filename).exists(), f"Expected file {filename} not found"
    
    # Verify CSV summaries
    expected_csvs = [
        "pairwise_dice_per_case.csv",
        "pairwise_dice_summary.csv",
        "dice_vs_gt_per_case.csv",
        "dice_vs_gt_summary.csv",
        "ged.csv",
        "ncc.csv",
        "ace.csv",
        "ba_ece.csv",
        "aurc.csv",
    ]
    
    for csv_file in expected_csvs:
        assert (tmp_path / csv_file).exists(), f"Expected CSV {csv_file} not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
