#!/usr/bin/env python3
"""Ensemble metric classes for uncertainty and agreement metrics."""

import os
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .metric_functions import (compute_ace, compute_dice,
                               compute_ensemble_entropy, compute_entropy_map,
                               compute_mutual_information_wrapper, compute_ncc,
                               get_correct_binary_multirater,
                               get_max_prob_for_pred_classes)


class BaseMetric:
    """Base class for all ensemble metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = []
        
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute metric for a single case."""
        raise NotImplementedError
        
    def export_summaries(self) -> None:
        """Export summary tables/statistics after processing all cases."""
        pass


class PredictiveEntropyMetric(BaseMetric):
    """Compute mean entropy maps across folds."""
    
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute mean entropy map for a case."""
        fold_indices = sorted(preds_per_fold.keys())
        ensemble_probs = np.stack([preds_per_fold[f] for f in fold_indices], axis=0)
        mean_probs = np.mean(ensemble_probs, axis=0)
        predictive_entropy_map = compute_entropy_map(mean_probs)
        
        if case_output_dir and affine is not None:
            if not os.path.exists(case_output_dir):
                os.makedirs(case_output_dir, exist_ok=True)
            entropy_img = nib.Nifti1Image(predictive_entropy_map.astype(np.float32), affine)
            nib.save(entropy_img, os.path.join(case_output_dir, "predictive_entropy_map.nii.gz"))
        
        return {
            "case_id": case_id,
            "predictive_entropy_mean": float(np.mean(predictive_entropy_map)),
            "predictive_entropy_std": float(np.std(predictive_entropy_map)),
            "predictive_entropy_max": float(np.max(predictive_entropy_map)),
        }


class MutualInformationMetric(BaseMetric):
    """Compute mutual information maps across folds."""
    
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute mutual information map for a case."""
        fold_indices = sorted(preds_per_fold.keys())
        ensemble_probs = np.stack([preds_per_fold[f] for f in fold_indices], axis=0)
        mi_map = compute_mutual_information_wrapper(ensemble_probs)
        
        if case_output_dir and affine is not None:
            if not os.path.exists(case_output_dir):
                os.makedirs(case_output_dir, exist_ok=True)
            mi_img = nib.Nifti1Image(mi_map.astype(np.float32), affine)
            nib.save(mi_img, os.path.join(case_output_dir, "mi_map.nii.gz"))
        
        return {
            "case_id": case_id,
            "mi_mean": float(np.mean(mi_map)),
            "mi_std": float(np.std(mi_map)),
            "mi_max": float(np.max(mi_map)),
        }


class ExpectedEntropyMetric(BaseMetric):
    """Compute mean entropy maps across folds."""
    
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute mean entropy map for a case."""
        fold_indices = sorted(preds_per_fold.keys())
        entropies = [compute_entropy_map(preds_per_fold[f]) for f in fold_indices]
        expected_entropy_map = np.mean(entropies, axis=0)
        
        if case_output_dir and affine is not None:
            if not os.path.exists(case_output_dir):
                os.makedirs(case_output_dir, exist_ok=True)
            entropy_img = nib.Nifti1Image(expected_entropy_map.astype(np.float32), affine)
            nib.save(entropy_img, os.path.join(case_output_dir, "expected_entropy_map.nii.gz"))
        
        return {
            "case_id": case_id,
            "expected_entropy_mean": float(np.mean(expected_entropy_map)),
            "expected_entropy_std": float(np.std(expected_entropy_map)),
            "expected_entropy_max": float(np.max(expected_entropy_map)),
        }


class PairwiseDiceMetric(BaseMetric):
    """Compute pairwise Dice scores between folds."""
    
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.pairwise_results = []
        
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute pairwise Dice scores for a case."""
        fold_indices = sorted(preds_per_fold.keys())
        num_classes = preds_per_fold[fold_indices[0]].shape[0]
        labels_per_fold = {f: np.argmax(preds_per_fold[f], axis=0) for f in fold_indices}
        
        pairwise_dice = []
        for i, fold_i in enumerate(fold_indices):
            labels_i = labels_per_fold[fold_i]
            for fold_j in fold_indices[i+1:]:
                labels_j = labels_per_fold[fold_j]
                dice_scores = {}
                for c in range(num_classes):
                    pred_c_i = (labels_i == c)
                    pred_c_j = (labels_j == c)
                    intersection = np.sum(pred_c_i & pred_c_j)
                    union = np.sum(pred_c_i) + np.sum(pred_c_j)
                    dice = 1.0 if union == 0 else 2.0 * intersection / union
                    dice_scores[f"class_{c}"] = float(dice)
                
                non_bg_classes = range(1, num_classes) if num_classes > 1 else [0]
                overall_dice = float(np.mean([dice_scores[f"class_{c}"] for c in non_bg_classes]))
                
                pairwise_dice.append({
                    "case_id": case_id,
                    "fold_i": fold_i,
                    "fold_j": fold_j,
                    "overall_dice": overall_dice,
                    **dice_scores
                })
        
        self.pairwise_results.extend(pairwise_dice)
        return {"case_id": case_id}
    
    def export_summaries(self) -> None:
        """Export pairwise Dice summary tables."""
        if not self.pairwise_results:
            return
        
        df = pd.DataFrame(self.pairwise_results)
        df.to_csv(os.path.join(self.output_dir, "pairwise_dice_per_case.csv"), index=False)
        
        dice_cols = [c for c in df.columns if c.startswith("class_") or c == "overall_dice"]
        summary_data = []
        for _, row in df.groupby(["fold_i", "fold_j"]):
            summary_row = {
                "fold_i": row["fold_i"].iloc[0],
                "fold_j": row["fold_j"].iloc[0],
            }
            for col in dice_cols:
                summary_row[f"{col}_mean"] = float(row[col].mean())
                summary_row[f"{col}_std"] = float(row[col].std())
            summary_data.append(summary_row)
        
        pd.DataFrame(summary_data).to_csv(
            os.path.join(self.output_dir, "pairwise_dice_summary.csv"), index=False
        )


class ConsensusSegmentationMetric(BaseMetric):
    """Export mean segmentation and compare with ground truth."""
    
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.gt_comparison_results = []
        
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute consensus segmentation and compare with GT."""
        fold_indices = sorted(preds_per_fold.keys())
        ensemble_probs = np.stack([preds_per_fold[f] for f in fold_indices], axis=0)
        mean_probs = np.mean(ensemble_probs, axis=0)
        consensus_seg = np.argmax(mean_probs, axis=0)
        
        if case_output_dir and affine is not None:
            if not os.path.exists(case_output_dir):
                os.makedirs(case_output_dir, exist_ok=True)
            consensus_img = nib.Nifti1Image(consensus_seg.astype(np.uint8), affine)
            nib.save(consensus_img, os.path.join(case_output_dir, "consensus_seg.nii.gz"))
        
        if gt is not None:
            num_classes = mean_probs.shape[0]
            gt_consensus = gt["consensus"]
            dice_scores = compute_dice(gt_consensus, consensus_seg, num_classes, include_background=False)
            fold_dice_scores = {
                f"fold_{f}": compute_dice(gt_consensus, np.argmax(preds_per_fold[f], axis=0), num_classes, include_background=False)
                for f in fold_indices
            }
            
            gt_comparison_row = {
                "case_id": case_id,
                "consensus_overall_dice": float(np.mean(list(dice_scores.values()))),
            }
            for k, v in dice_scores.items():
                gt_comparison_row[f"consensus_{k}"] = float(v)
            for fold_name, fold_dice in fold_dice_scores.items():
                gt_comparison_row[f"{fold_name}_overall_dice"] = float(np.mean(list(fold_dice.values())))
                for k, v in fold_dice.items():
                    gt_comparison_row[f"{fold_name}_{k}"] = float(v)
            
            self.gt_comparison_results.append(gt_comparison_row)
        
        return {"case_id": case_id}
    
    def export_summaries(self) -> None:
        """Export ground truth comparison summaries."""
        if not self.gt_comparison_results:
            return
        
        df = pd.DataFrame(self.gt_comparison_results)
        df.to_csv(os.path.join(self.output_dir, "dice_vs_gt_per_case.csv"), index=False)
        
        summary_cols = [c for c in df.columns if c != "case_id" and c.endswith("_dice")]
        summary_data = {f"{col}_mean": float(df[col].mean()) for col in summary_cols}
        summary_data.update({f"{col}_std": float(df[col].std()) for col in summary_cols})
        
        pd.DataFrame([summary_data]).to_csv(
            os.path.join(self.output_dir, "dice_vs_gt_summary.csv"), index=False
        )


class NCCMetric(BaseMetric):
    """Compute Normalized Cross-Correlation (NCC) metric."""

    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.ncc_results = []
    
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compute NCC for a case. Requires expected entropy to be calculated first."""
        gt_var = np.var(gt["raters"], axis=0)
        expected_entropy_path = os.path.join(case_output_dir, "expected_entropy_map.nii.gz")
        if not os.path.exists(expected_entropy_path):
            raise FileNotFoundError(f"Expected entropy map not found for case {case_id}")
        expected_entropy_pred = nib.load(expected_entropy_path).get_fdata()
        ncc_value = compute_ncc(expected_entropy_pred, gt_var)
        self.ncc_results.append({
            "case_id": case_id,
            "ncc": float(ncc_value),
        })

        return {"case_id": case_id}
    
    def export_summaries(self) -> None:
        """Export NCC summaries."""
        if not self.ncc_results:
            return
        
        self.ncc_results.append({
            "case_id": "mean",
            "ncc": float(np.mean([r["ncc"] for r in self.ncc_results])),
        })
        df = pd.DataFrame(self.ncc_results)
        df.to_csv(os.path.join(self.output_dir, "ncc.csv"), index=False)


class ACEMeric(BaseMetric):
    """Compute Average Calibration Error."""
    
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.ace_results = []

    def compute_case(
            self,
            case_id: str,
            preds_per_fold: Dict[int, np.ndarray],
            gt: Optional[np.ndarray] = None,
            affine: Optional[np.ndarray] = None,
            case_output_dir: Optional[str] = None
        ) -> Dict[str, Any]:
        """Compute ACE for a case. Requires consensus prediction to be calculated first."""
        conensus_pred_path = os.path.join(case_output_dir, "consensus_seg.nii.gz")
        if not os.path.exists(conensus_pred_path):
            raise FileNotFoundError(f"Consensus segmentation not found for case {case_id}")
        consensus_pred = nib.load(conensus_pred_path).get_fdata()
        gt_raters = gt["raters"]
        correct = get_correct_binary_multirater(gt_raters=gt_raters, pred=consensus_pred)
        conf = get_max_prob_for_pred_classes(probs_per_fold=preds_per_fold, consensus_pred=consensus_pred)
        conf = np.repeat(conf[np.newaxis, ...], gt["raters"].shape[0], axis=0).ravel()
        ace_value = compute_ace(correct=correct, calib_confids=conf, n_bins=20)
        self.ace_results.append({
            "case_id": case_id,
            "ace": float(ace_value),
        })
        return {"case_id": case_id}
    
    def export_summaries(self) -> None:
        """Export ACE summaries."""
        if not self.ace_results:
            return
        
        self.ace_results.append({
            "case_id": "mean",
            "ace": float(np.mean([r["ace"] for r in self.ace_results])),
        })
        df = pd.DataFrame(self.ace_results)
        df.to_csv(os.path.join(self.output_dir, "ace.csv"), index=False)


METRICS = {
    "predictive_entropy": PredictiveEntropyMetric,
    "mutual_information": MutualInformationMetric,
    "expected_entropy": ExpectedEntropyMetric,
    "pairwise_dice": PairwiseDiceMetric,
    "consensus_segmentation": ConsensusSegmentationMetric,
    "ncc": NCCMetric,
    "ace": ACEMeric,
}
