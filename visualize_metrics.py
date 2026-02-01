#!/usr/bin/env python3
"""Visualize dummy data and computed metrics for LostInFolds ensemble metrics."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_ensemble_metrics_integration import (
    create_dummy_multirater_labels,
    create_dummy_ensemble_probs,
)
from src.ensemble_metrics.metric_functions import (
    compute_entropy_map,
    compute_mutual_information_wrapper,
)
from src.ensemble_metrics.metrics import (
    PredictiveEntropyMetric,
    MutualInformationMetric,
    ExpectedEntropyMetric,
    ConsensusSegmentationMetric,
    GEDMetric,
)


def visualize_multirater_labels(multirater_gt, output_dir):
    """Visualize multirater ground truth labels."""
    raters = multirater_gt["raters"]
    consensus = multirater_gt["consensus"]
    num_raters = raters.shape[0]
    
    # Take middle slice
    slice_idx = raters.shape[1] // 2
    
    fig, axes = plt.subplots(2, num_raters + 1, figsize=(4 * (num_raters + 1), 8))
    fig.suptitle("Multirater Ground Truth Labels (Middle Slice)", fontsize=16, fontweight='bold')
    
    # Show each rater
    for r in range(num_raters):
        ax = axes[0, r]
        im = ax.imshow(raters[r, slice_idx, :, :], cmap='tab10', vmin=0, vmax=3)
        ax.set_title(f"Rater {r+1}", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Show consensus
    ax = axes[0, num_raters]
    im = ax.imshow(consensus[slice_idx, :, :], cmap='tab10', vmin=0, vmax=3)
    ax.set_title("Consensus (Majority Vote)", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Show rater agreement/disagreement
    for r in range(num_raters):
        ax = axes[1, r]
        agreement = (raters[r, slice_idx, :, :] == consensus[slice_idx, :, :]).astype(float)
        im = ax.imshow(agreement, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f"Rater {r+1} vs Consensus", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Show inter-rater variability
    ax = axes[1, num_raters]
    variability = np.std(raters[:, slice_idx, :, :], axis=0)
    im = ax.imshow(variability, cmap='hot', vmin=0, vmax=variability.max())
    ax.set_title("Inter-Rater Variability\n(Std Dev)", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multirater_labels.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'multirater_labels.png'}")
    plt.close()


def visualize_ensemble_probs(ensemble_probs, output_dir):
    """Visualize ensemble probability predictions."""
    num_folds = len(ensemble_probs)
    fold_indices = sorted(ensemble_probs.keys())
    
    # Take middle slice
    slice_idx = ensemble_probs[fold_indices[0]].shape[2] // 2
    
    fig, axes = plt.subplots(num_folds + 1, 4, figsize=(16, 4 * (num_folds + 1)))
    fig.suptitle("Ensemble Probability Predictions (Middle Slice)", fontsize=16, fontweight='bold')
    
    # Show each fold
    for i, fold_idx in enumerate(fold_indices):
        probs = ensemble_probs[fold_idx]
        for c in range(4):
            ax = axes[i, c]
            im = ax.imshow(probs[c, slice_idx, :, :], cmap='viridis', vmin=0, vmax=1)
            ax.set_title(f"Fold {fold_idx}, Class {c}", fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Show mean probabilities
    mean_probs = np.mean([ensemble_probs[f] for f in fold_indices], axis=0)
    for c in range(4):
        ax = axes[num_folds, c]
        im = ax.imshow(mean_probs[c, slice_idx, :, :], cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Mean Probs, Class {c}", fontsize=10, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ensemble_probabilities.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'ensemble_probabilities.png'}")
    plt.close()


def visualize_uncertainty_metrics(preds_per_fold, output_dir):
    """Visualize uncertainty metrics."""
    fold_indices = sorted(preds_per_fold.keys())
    ensemble_probs = np.stack([preds_per_fold[f] for f in fold_indices], axis=0)
    
    # Take middle slice
    slice_idx = ensemble_probs.shape[2] // 2
    
    # Compute metrics
    mean_probs = np.mean(ensemble_probs, axis=0)
    predictive_entropy = compute_entropy_map(mean_probs)
    mi_map = compute_mutual_information_wrapper(ensemble_probs)
    expected_entropy = np.mean([compute_entropy_map(ensemble_probs[i]) for i in range(len(fold_indices))], axis=0)
    
    # Consensus segmentation
    consensus_seg = np.argmax(mean_probs, axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Uncertainty Metrics (Middle Slice)", fontsize=16, fontweight='bold')
    
    # Consensus segmentation
    ax = axes[0, 0]
    im = ax.imshow(consensus_seg[slice_idx, :, :], cmap='tab10', vmin=0, vmax=3)
    ax.set_title("Consensus Segmentation", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Predictive entropy
    ax = axes[0, 1]
    im = ax.imshow(predictive_entropy[slice_idx, :, :], cmap='hot', vmin=0)
    ax.set_title(f"Predictive Entropy\n(Mean: {predictive_entropy[slice_idx, :, :].mean():.3f})", 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Mutual information
    ax = axes[0, 2]
    im = ax.imshow(mi_map[slice_idx, :, :], cmap='plasma', vmin=0)
    ax.set_title(f"Mutual Information\n(Mean: {mi_map[slice_idx, :, :].mean():.3f})", 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Expected entropy
    ax = axes[1, 0]
    im = ax.imshow(expected_entropy[slice_idx, :, :], cmap='inferno', vmin=0)
    ax.set_title(f"Expected Entropy\n(Mean: {expected_entropy[slice_idx, :, :].mean():.3f})", 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Epistemic uncertainty (MI)
    ax = axes[1, 1]
    epistemic = mi_map[slice_idx, :, :]
    im = ax.imshow(epistemic, cmap='plasma', vmin=0)
    ax.set_title(f"Epistemic Uncertainty (MI)\n(Mean: {epistemic.mean():.3f})", 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Aleatoric uncertainty (Expected entropy)
    ax = axes[1, 2]
    aleatoric = expected_entropy[slice_idx, :, :]
    im = ax.imshow(aleatoric, cmap='inferno', vmin=0)
    ax.set_title(f"Aleatoric Uncertainty\n(Mean: {aleatoric.mean():.3f})", 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_metrics.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'uncertainty_metrics.png'}")
    plt.close()


def visualize_metric_comparison(multirater_gt, preds_per_fold, output_dir):
    """Compare predictions with ground truth."""
    fold_indices = sorted(preds_per_fold.keys())
    ensemble_probs = np.stack([preds_per_fold[f] for f in fold_indices], axis=0)
    mean_probs = np.mean(ensemble_probs, axis=0)
    consensus_pred = np.argmax(mean_probs, axis=0)
    consensus_gt = multirater_gt["consensus"]
    
    slice_idx = consensus_pred.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Prediction vs Ground Truth Comparison (Middle Slice)", fontsize=16, fontweight='bold')
    
    # Ground truth consensus
    ax = axes[0, 0]
    im = ax.imshow(consensus_gt[slice_idx, :, :], cmap='tab10', vmin=0, vmax=3)
    ax.set_title("Ground Truth Consensus", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Predicted consensus
    ax = axes[0, 1]
    im = ax.imshow(consensus_pred[slice_idx, :, :], cmap='tab10', vmin=0, vmax=3)
    ax.set_title("Predicted Consensus", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Agreement
    ax = axes[0, 2]
    agreement = (consensus_pred[slice_idx, :, :] == consensus_gt[slice_idx, :, :]).astype(float)
    im = ax.imshow(agreement, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title(f"Agreement\n(Accuracy: {agreement.mean():.3f})", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Per-class comparison
    for c in range(1, 4):
        ax = axes[1, c-1]
        gt_mask = (consensus_gt[slice_idx, :, :] == c)
        pred_mask = (consensus_pred[slice_idx, :, :] == c)
        comparison = np.zeros_like(consensus_gt[slice_idx, :, :], dtype=float)
        comparison[gt_mask & pred_mask] = 1.0  # True positive
        comparison[gt_mask & ~pred_mask] = 0.5  # False negative
        comparison[~gt_mask & pred_mask] = 0.75  # False positive
        
        im = ax.imshow(comparison, cmap='RdYlGn', vmin=0, vmax=1)
        tp = np.sum(gt_mask & pred_mask)
        fn = np.sum(gt_mask & ~pred_mask)
        fp = np.sum(~gt_mask & pred_mask)
        dice = 2 * tp / (2 * tp + fn + fp) if (tp + fn + fp) > 0 else 0
        ax.set_title(f"Class {c}\n(Dice: {dice:.3f})", fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'prediction_comparison.png'}")
    plt.close()


def compute_and_print_metrics(multirater_gt, ensemble_probs, output_dir):
    """Compute and print all metric values."""
    import tempfile
    from src.ensemble_metrics.metric_functions import (
        compute_dice,
        compute_ged,
        compute_ncc,
        get_correct_binary_multirater,
        get_max_prob_for_pred_classes,
        compute_ace,
        compute_ba_ece,
        compute_aurc,
    )
    
    print("\n" + "="*80)
    print("METRIC VALUES SUMMARY")
    print("="*80)
    
    fold_indices = sorted(ensemble_probs.keys())
    ensemble_probs_array = np.stack([ensemble_probs[f] for f in fold_indices], axis=0)
    mean_probs = np.mean(ensemble_probs_array, axis=0)
    consensus_pred = np.argmax(mean_probs, axis=0)
    consensus_gt = multirater_gt["consensus"]
    num_classes = mean_probs.shape[0]
    
    # Create temp directory for metric outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        case_output_dir = tmp_path / "case_001"
        case_output_dir.mkdir()
        affine = np.eye(4)
        
        # ===== UNCERTAINTY METRICS =====
        print("\n📊 UNCERTAINTY METRICS")
        print("-" * 80)
        
        # Predictive Entropy
        predictive_entropy_map = compute_entropy_map(mean_probs)
        print(f"Predictive Entropy:")
        print(f"  Mean: {predictive_entropy_map.mean():.4f}")
        print(f"  Std:  {predictive_entropy_map.std():.4f}")
        print(f"  Min:  {predictive_entropy_map.min():.4f}")
        print(f"  Max:  {predictive_entropy_map.max():.4f}")
        
        # Expected Entropy
        expected_entropy_map = np.mean([compute_entropy_map(ensemble_probs_array[i]) 
                                       for i in range(len(fold_indices))], axis=0)
        print(f"\nExpected Entropy:")
        print(f"  Mean: {expected_entropy_map.mean():.4f}")
        print(f"  Std:  {expected_entropy_map.std():.4f}")
        print(f"  Min:  {expected_entropy_map.min():.4f}")
        print(f"  Max:  {expected_entropy_map.max():.4f}")
        
        # Mutual Information
        mi_map = compute_mutual_information_wrapper(ensemble_probs_array)
        print(f"\nMutual Information (Epistemic Uncertainty):")
        print(f"  Mean: {mi_map.mean():.4f}")
        print(f"  Std:  {mi_map.std():.4f}")
        print(f"  Min:  {mi_map.min():.4f}")
        print(f"  Max:  {mi_map.max():.4f}")
        
        # ===== SEGMENTATION METRICS =====
        print("\n🎯 SEGMENTATION METRICS")
        print("-" * 80)
        
        # Dice scores vs consensus GT
        dice_scores = compute_dice(consensus_gt, consensus_pred, num_classes, include_background=False)
        print(f"Dice Scores (Consensus Prediction vs Consensus GT):")
        for class_name, dice in dice_scores.items():
            print(f"  {class_name}: {dice:.4f}")
        overall_dice = np.mean(list(dice_scores.values()))
        print(f"  Overall: {overall_dice:.4f}")
        
        # Pairwise Dice between folds
        fold_labels = {f: np.argmax(ensemble_probs[f], axis=0) for f in fold_indices}
        pairwise_dice_list = []
        for i, fold_i in enumerate(fold_indices):
            for fold_j in fold_indices[i+1:]:
                labels_i = fold_labels[fold_i]
                labels_j = fold_labels[fold_j]
                dice = compute_dice(labels_i, labels_j, num_classes, include_background=False)
                pairwise_dice_list.append(dice)
        
        if pairwise_dice_list:
            avg_pairwise = {k: np.mean([d[k] for d in pairwise_dice_list]) 
                           for k in pairwise_dice_list[0].keys()}
            print(f"\nAverage Pairwise Dice (between folds):")
            for class_name, dice in avg_pairwise.items():
                print(f"  {class_name}: {dice:.4f}")
            overall_pairwise = np.mean(list(avg_pairwise.values()))
            print(f"  Overall: {overall_pairwise:.4f}")
        
        # ===== MULTIRATER METRICS =====
        print("\n👥 MULTIRATER METRICS")
        print("-" * 80)
        
        # GED (Generalized Energy Distance)
        ensemble_pred_labels = np.stack([np.argmax(ensemble_probs[f], axis=0) 
                                        for f in fold_indices], axis=0)
        ged_dict = compute_ged(
            gt_raters=multirater_gt["raters"],
            ensemble_pred=ensemble_pred_labels,
            num_classes=num_classes,
            include_background=False
        )
        print(f"Generalized Energy Distance (GED):")
        for key, value in ged_dict.items():
            print(f"  {key}: {value:.4f}")
        
        # NCC (Normalized Cross-Correlation)
        # Save expected entropy for NCC
        import nibabel as nib
        entropy_img = nib.Nifti1Image(expected_entropy_map.astype(np.float32), affine)
        nib.save(entropy_img, str(case_output_dir / "expected_entropy_map.nii.gz"))
        
        gt_var = np.var(multirater_gt["raters"], axis=0)
        ncc_value = compute_ncc(expected_entropy_map, gt_var)
        print(f"\nNormalized Cross-Correlation (NCC):")
        print(f"  NCC: {ncc_value:.4f}")
        
        # ACE (Average Calibration Error)
        # Save consensus seg for ACE
        consensus_img = nib.Nifti1Image(consensus_pred.astype(np.uint8), affine)
        nib.save(consensus_img, str(case_output_dir / "consensus_seg.nii.gz"))
        
        gt_raters = multirater_gt["raters"]
        correct = get_correct_binary_multirater(gt_raters=gt_raters, pred=consensus_pred)
        conf = get_max_prob_for_pred_classes(probs_per_fold=ensemble_probs, consensus_pred=consensus_pred)
        conf_expanded = np.repeat(conf[np.newaxis, ...], gt_raters.shape[0], axis=0).ravel()
        ace_value = compute_ace(correct=correct, calib_confids=conf_expanded, n_bins=20)
        print(f"\nAverage Calibration Error (ACE):")
        print(f"  ACE: {ace_value:.4f}")
        
        # BA-ECE (Boundary-Aware Expected Calibration Error)
        conf_ba = np.repeat(conf[np.newaxis, ...], gt_raters.shape[0], axis=0)
        consensus_pred_ba = np.repeat(consensus_pred[np.newaxis, ...], gt_raters.shape[0], axis=0)
        ba_ece_dict = compute_ba_ece(
            confidence=conf_ba,
            labels=gt_raters,
            pred_labels=consensus_pred_ba
        )
        print(f"\nBoundary-Aware Expected Calibration Error (BA-ECE):")
        print(f"  BA-ECE: {ba_ece_dict['ba_ece']:.4f}")
        print(f"  Number of bands: {len(ba_ece_dict['bands'])}")
        print(f"  Band details:")
        for i, band in enumerate(ba_ece_dict['bands'][:5]):  # Show first 5 bands
            if not np.isnan(band['calibration_error']):
                print(f"    Band {i}: count={band['count']}, "
                      f"mean_unc={band['mean_uncertainty']:.4f}, "
                      f"mean_err={band['mean_error']:.4f}, "
                      f"cal_err={band['calibration_error']:.4f}")
        
        # AURC (Area Under Risk-Coverage curve)
        # Create dummy risk and confidence for AURC
        risks = np.array([1.0 - overall_dice])  # Using overall dice as risk
        confids = np.array([overall_pairwise])  # Using pairwise dice as confidence
        aurc_value = compute_aurc(risks, confids)
        print(f"\nArea Under Risk-Coverage Curve (AURC):")
        print(f"  AURC: {aurc_value:.4f}")
        
        # ===== DATA STATISTICS =====
        print("\n📈 DATA STATISTICS")
        print("-" * 80)
        print(f"Number of raters: {multirater_gt['raters'].shape[0]}")
        print(f"Number of folds: {len(ensemble_probs)}")
        print(f"Number of classes: {num_classes}")
        print(f"Spatial shape: {multirater_gt['raters'].shape[1:]}")
        print(f"\nClass distribution in consensus GT:")
        unique, counts = np.unique(consensus_gt, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} voxels ({100*count/consensus_gt.size:.2f}%)")
        print(f"\nClass distribution in consensus prediction:")
        unique, counts = np.unique(consensus_pred, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} voxels ({100*count/consensus_pred.size:.2f}%)")
        
        print("\n" + "="*80)


def main():
    """Generate all visualizations."""
    print("Generating dummy data...")
    
    # Create output directory
    output_dir = Path("metrics_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate dummy data
    multirater_gt = create_dummy_multirater_labels(
        num_raters=3,
        num_classes=4,
        spatial_shape=(32, 32, 32),
        seed=42,
    )
    
    ensemble_probs = create_dummy_ensemble_probs(
        num_folds=5,
        num_classes=4,
        spatial_shape=(32, 32, 32),
        seed=42,
    )
    
    # Compute and print all metrics
    compute_and_print_metrics(multirater_gt, ensemble_probs, output_dir)
    
    print("\nCreating visualizations...")
    
    # Visualize multirater labels
    visualize_multirater_labels(multirater_gt, output_dir)
    
    # Visualize ensemble probabilities
    visualize_ensemble_probs(ensemble_probs, output_dir)
    
    # Visualize uncertainty metrics
    visualize_uncertainty_metrics(ensemble_probs, output_dir)
    
    # Visualize prediction comparison
    visualize_metric_comparison(multirater_gt, ensemble_probs, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
