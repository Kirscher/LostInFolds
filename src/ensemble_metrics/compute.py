#!/usr/bin/env python3
"""Main CLI script for computing ensemble metrics from nnUNet multi-fold predictions."""

import argparse
import os
import re
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .metrics import METRICS
from .utils import (
    discover_folds,
    discover_cases,
    load_prediction,
    load_ground_truth,
    standardize_prediction,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute ensemble metrics from nnUNet multi-fold predictions"
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        required=True,
        help="Root directory containing fold subdirectories (e.g., Dataset120_RoadSegmentation)"
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Number of folds to use (e.g., 3 or 5). If not specified, uses all available folds."
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Optional ground truth directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for metric maps and summaries"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="predictive_entropy,mutual_information,expected_entropy,pairwise_dice,consensus_segmentation",
        help="Comma-separated list of metrics to compute"
    )
    parser.add_argument(
        "--case-filter",
        type=str,
        default=None,
        help="Optional regex pattern to filter case IDs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover folds
    print(f"Discovering folds in {args.pred_root}...")
    fold_dirs = discover_folds(args.pred_root, args.num_folds)
    if not fold_dirs:
        raise ValueError(f"No fold directories found in {args.pred_root}")
    
    print(f"Found {len(fold_dirs)} fold(s): {[f[0] for f in fold_dirs]}")
    
    # Discover cases
    fold_paths = [f[1] for f in fold_dirs]
    print("Discovering cases...")
    case_ids = discover_cases(fold_paths)
    
    if args.case_filter:
        pattern = re.compile(args.case_filter)
        case_ids = [c for c in case_ids if pattern.match(c)]
    
    if not case_ids:
        raise ValueError("No common cases found across folds")
    
    print(f"Found {len(case_ids)} case(s)")
    
    # Initialize metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]
    metrics = []
    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
        metric_obj = METRICS[name](args.output_dir)
        metrics.append(metric_obj)
    
    print(f"Computing metrics: {metric_names}")
    
    for case_id in tqdm(case_ids, desc="Processing cases"):
        preds_per_fold = {}
        affine = None
        
        for fold_idx, fold_path in fold_dirs:
            try:
                pred, aff = load_prediction(fold_path, case_id)
                pred = standardize_prediction(pred)
                preds_per_fold[fold_idx] = pred
                if affine is None and aff is not None:
                    affine = aff
            except Exception as e:
                print(f"Warning: Failed to load prediction for case {case_id} from fold {fold_idx}: {e}")
                continue
        
        if not preds_per_fold:
            print(f"Warning: No predictions loaded for case {case_id}, skipping")
            continue
        
        shapes = [p.shape[1:] for p in preds_per_fold.values()]
        if len(set(shapes)) > 1:
            print(f"Warning: Shape mismatch for case {case_id}: {shapes}, skipping")
            continue
        
        ref_shape = shapes[0] if shapes else None
        if affine is None:
            spatial_dims = len(ref_shape) if ref_shape else 3
            affine = np.eye(spatial_dims + 1)
        
        gt = None
        if args.gt_dir:
            gt_data = load_ground_truth(args.gt_dir, case_id)
            if gt_data is not None:
                gt, gt_affine = gt_data
                if affine is None:
                    affine = gt_affine
        
        case_output_dir = os.path.join(args.output_dir, case_id)
        
        for metric in metrics:
            try:
                metric.compute_case(
                    case_id=case_id,
                    preds_per_fold=preds_per_fold,
                    gt=gt,
                    affine=affine,
                    case_output_dir=case_output_dir
                )
            except Exception as e:
                print(f"Warning: Metric {metric.__class__.__name__} failed for case {case_id}: {e}")
                import traceback
                traceback.print_exc()
    
    print("Exporting summaries...")
    for metric in metrics:
        try:
            metric.export_summaries()
        except Exception as e:
            print(f"Warning: Failed to export summary for {metric.__class__.__name__}: {e}")
    
    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
