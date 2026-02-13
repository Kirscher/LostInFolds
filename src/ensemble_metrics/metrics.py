#!/usr/bin/env python3
"""Ensemble metric classes for uncertainty and agreement metrics."""

import os
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .metric_functions import (compute_ace, compute_aurc, compute_ba_ece,
                               compute_dice, compute_ensemble_entropy,
                               compute_entropy_map, compute_ged,
                               compute_mutual_information_wrapper, compute_ncc)


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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute mean entropy map for a case."""
        mean_probs = precomputed["mean_probs"]
        predictive_entropy_map = compute_entropy_map(mean_probs)
        
        if save_maps and case_output_dir and affine is not None:
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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute mutual information map for a case."""
        mean_probs = precomputed["mean_probs"]
        pred_entropy = compute_entropy_map(mean_probs)
        expected_entropy = precomputed["expected_entropy_map"]
        mi_map = np.maximum(pred_entropy - expected_entropy, 0.0)
        del pred_entropy
        
        if save_maps and case_output_dir and affine is not None:
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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute mean entropy map for a case."""
        expected_entropy_map = precomputed["expected_entropy_map"]
        
        if save_maps and case_output_dir and affine is not None:
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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute pairwise Dice scores for a case."""
        labels_per_fold = precomputed["labels_per_fold"]
        fold_indices = sorted(labels_per_fold.keys())
        num_classes = precomputed["mean_probs"].shape[0]
        
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
        pairwise_dice_df = pd.DataFrame(pairwise_dice)
        pairwise_dice_df.to_csv(
            os.path.join(case_output_dir, "pairwise_dice.csv"), index=False
        )
        
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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute consensus segmentation and compare with GT."""
        consensus_seg = precomputed["consensus_seg"]
        num_classes = precomputed["mean_probs"].shape[0]
        labels_per_fold = precomputed["labels_per_fold"]
        fold_indices = sorted(labels_per_fold.keys())
        
        if save_maps and case_output_dir and affine is not None:
            consensus_img = nib.Nifti1Image(consensus_seg.astype(np.uint8), affine)
            nib.save(consensus_img, os.path.join(case_output_dir, "consensus_seg.nii.gz"))
        
        if gt is not None:
            gt_consensus = gt["consensus"]
            dice_scores = compute_dice(gt_consensus, consensus_seg, num_classes, include_background=False)
            fold_dice_scores = {
                f"fold_{f}": compute_dice(gt_consensus, labels_per_fold[f], num_classes, include_background=False)
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
            
            gt_comparison_df = pd.DataFrame([gt_comparison_row])
            gt_comparison_df.to_csv(
                os.path.join(case_output_dir, "dice_vs_gt.csv"), index=False
            )
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
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute NCC for a case using precomputed expected entropy."""
        gt_var = np.var(gt["raters"], axis=0)
        expected_entropy_pred = precomputed["expected_entropy_map"]
        ncc_value = compute_ncc(expected_entropy_pred, gt_var)
        del gt_var
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


class GEDMetric(BaseMetric):
    """Compute Generalized Energy Distance (GED) metric."""
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.ged_results = []
    
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compute GED for a case."""
        labels_per_fold = precomputed["labels_per_fold"]
        fold_indices = sorted(labels_per_fold.keys())
        num_classes = precomputed["mean_probs"].shape[0]
        ensemble_preds = np.stack([labels_per_fold[f] for f in fold_indices], axis=0)
        ged = compute_ged(gt_raters=gt["raters"], ensemble_pred=ensemble_preds, num_classes=num_classes)
        del ensemble_preds

        ged_row = {
            "case_id": case_id,
            **ged
        }
        self.ged_results.append(ged_row)
        return { "case_id": case_id }
    
    def export_summaries(self) -> None:
        """Export GED summaries."""
        if not self.ged_results:
            return
        
        self.ged_results.append({
            "case_id": "mean",
            **{k: float(np.mean([r[k] for r in self.ged_results])) for k in self.ged_results[0] if k != "case_id"},
        })
        ged_df = pd.DataFrame(self.ged_results)
        ged_df.to_csv(os.path.join(self.output_dir, "ged.csv"), index=False)


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
            case_output_dir: Optional[str] = None,
            save_maps: bool = False,
            precomputed: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        """Compute ACE for a case.

        Process raters one at a time to avoid np.repeat on huge arrays.
        """
        consensus_pred = precomputed["consensus_seg"]
        conf = precomputed["max_conf"]
        gt_raters = gt["raters"]
        # Accumulate correct/conf per rater without tiling
        correct_parts = []
        conf_parts = []
        for r_idx in range(gt_raters.shape[0]):
            correct_parts.append((gt_raters[r_idx] == consensus_pred).ravel())
            conf_parts.append(conf.ravel())
        correct_all = np.concatenate(correct_parts).astype(np.int32)
        conf_all = np.concatenate(conf_parts)
        del correct_parts, conf_parts
        ace_value = compute_ace(correct=correct_all, calib_confids=conf_all, n_bins=20)
        del correct_all, conf_all
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


class BAECEMetric(BaseMetric):
    """Compute Boundary Aware Calibration Error."""
    
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.baece_results = []

    def compute_case(
            self,
            case_id: str,
            preds_per_fold: Dict[int, np.ndarray],
            gt: Optional[np.ndarray] = None,
            affine: Optional[np.ndarray] = None,
            case_output_dir: Optional[str] = None,
            save_maps: bool = False,
            precomputed: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
        """Compute BAECE for a case.

        Process raters one at a time to avoid stacking huge distance/band
        arrays across all raters.
        """
        from .metric_functions import compute_ba_ece_streaming
        consensus_pred = precomputed["consensus_seg"]
        conf = precomputed["max_conf"]
        ba_ece = compute_ba_ece_streaming(
            confidence=conf,
            labels=gt["raters"],
            pred_labels=consensus_pred,
        )
        self.baece_results.append({
            "case_id": case_id,
            **ba_ece
        })
        return {"case_id": case_id}
    
    def export_summaries(self) -> None:
        """Export BACE summaries."""
        if not self.baece_results:
            return
        baece_results_summary = [{
            "case_id": r["case_id"],
            "ba_ece": r["ba_ece"]
        } for r in self.baece_results]
        baece_results_summary.append({
            "case_id": "mean",
            "ba_ece": float(np.mean([r["ba_ece"] for r in self.baece_results])),
        })
        df = pd.DataFrame(baece_results_summary)
        df.to_csv(os.path.join(self.output_dir, "ba_ece.csv"), index=False)


class AURCMetric(BaseMetric):
    """Compute Area Under the Risk-Coverage Curve (AURC) metric."""
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.risks = []
        self.confids = []
        self.num_classes = None
    
    def compute_case(
        self,
        case_id: str,
        preds_per_fold: Dict[int, np.ndarray],
        gt: Optional[np.ndarray] = None,
        affine: Optional[np.ndarray] = None,
        case_output_dir: Optional[str] = None,
        save_maps: bool = False,
        precomputed: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute risks and confids for a case. The final AURC is computed in export_summaries.
        Assumes dice and pairwise dice to be available.
        """
        dice_per_case_path = os.path.join(case_output_dir, "dice_vs_gt.csv")
        pairwise_dice_path = os.path.join(case_output_dir, "pairwise_dice.csv")
        if not os.path.exists(dice_per_case_path):
            raise FileNotFoundError(f"Dice per case file not found for case {case_id}")
        if not os.path.exists(pairwise_dice_path):
            raise FileNotFoundError(f"Pairwise Dice per case file not found for case {case_id}")
        dice_df = pd.read_csv(dice_per_case_path)
        pairwise_dice_df = pd.read_csv(pairwise_dice_path)
        dice = dice_df[dice_df["case_id"] == case_id]
        pairwise_dice = pairwise_dice_df[pairwise_dice_df["case_id"] == case_id]
        pairwise_dice_mean = pairwise_dice.drop(["case_id", "fold_i", "fold_j"], axis=1).mean()
        
        self.num_classes = precomputed["mean_probs"].shape[0]
        risk_dict = {}
        confid_dict = {}
        for c in range(1, self.num_classes):
            dice_c = dice[f"consensus_class_{c}"].values[0]
            pairwise_dice_c = pairwise_dice_mean[f"class_{c}"]
            risk = 1.0 - dice_c
            risk_dict[f"class_{c}"] = risk
            confid_dict[f"class_{c}"] = pairwise_dice_c
        risk_dict["overall_risk"] = 1.0 - dice["consensus_overall_dice"].values[0]
        confid_dict["overall_confid"] = pairwise_dice_mean["overall_dice"]
        
        self.risks.append({
            "case_id": case_id,
            **risk_dict})
        self.confids.append({
            "case_id": case_id,
            **confid_dict})

        return { "case_id": case_id }
    
    def export_summaries(self) -> None:
        """Export AURC summaries."""
        if not self.risks or not self.confids:
            return
        aurc_dict = {}
        for i in range(1, self.num_classes):
            risks = np.array([r[f"class_{i}"] for r in self.risks])
            confids = np.array([c[f"class_{i}"] for c in self.confids])
            aurc_dict[f"class_{i}"] = compute_aurc(risks, confids)
        overall_risks = np.array([r["overall_risk"] for r in self.risks])
        overall_confids = np.array([c["overall_confid"] for c in self.confids])
        aurc_dict["overall_aurc"] = compute_aurc(overall_risks, overall_confids)
        
        aurc_df = pd.DataFrame([aurc_dict])
        aurc_df.to_csv(os.path.join(self.output_dir, "aurc.csv"), index=False)

        # Save per-case risk and confidence data for bootstrap resampling
        per_case_rows = []
        for r, c in zip(self.risks, self.confids):
            row = {"case_id": r["case_id"]}
            for key in r:
                if key != "case_id":
                    row[f"risk_{key}"] = r[key]
            for key in c:
                if key != "case_id":
                    row[f"confid_{key}"] = c[key]
            per_case_rows.append(row)
        per_case_df = pd.DataFrame(per_case_rows)
        per_case_df.to_csv(os.path.join(self.output_dir, "aurc_per_case.csv"), index=False)


METRICS = {
    "predictive_entropy": PredictiveEntropyMetric,
    "mutual_information": MutualInformationMetric,
    "expected_entropy": ExpectedEntropyMetric,
    "pairwise_dice": PairwiseDiceMetric,
    "consensus_segmentation": ConsensusSegmentationMetric,
    "ncc": NCCMetric,
    "ace": ACEMeric,
    "ba_ece": BAECEMetric,
    "ged": GEDMetric,
    "aurc": AURCMetric,
}
