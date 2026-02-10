#!/usr/bin/env python3
"""Main CLI script for computing ensemble metrics from nnUNet multi-fold predictions."""

import argparse
import ctypes
import gc
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


def _release_memory():
    """Force Python GC and glibc to return freed pages to the OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

from .metrics import METRICS
from .utils import (discover_cases, discover_folds, load_ground_truth,
                    load_prediction, standardize_prediction)


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
        "--num-raters",
        type=int,
        default=3,
        help="Number of raters for ground truth"
    )
    parser.add_argument(
        "--consensus-type",
        type=str,
        default="staple",
        help="Consensus method for ground truth. Can be 'staple', 'majority', or 'none'. If 'none', the majority vote is calculated based on available raters."
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
        default="predictive_entropy,mutual_information,expected_entropy,pairwise_dice,consensus_segmentation,ncc,ace,ged,aurc",
        help="Comma-separated list of metrics to compute"
    )
    parser.add_argument(
        "--case-filter",
        type=str,
        default=None,
        help="Optional regex pattern to filter case IDs"
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        default=False,
        help="Save per-case nifti maps (entropy, MI, etc). Disabled by default to save disk."
    )
    parser.add_argument(
        "--gt-label-offset",
        type=int,
        default=0,
        help="Integer added to every non-zero GT label before comparison. "
             "E.g. --gt-label-offset 1 maps GT {0,1,2} -> {0,2,3} to align "
             "with predictions that include an extra class."
    )
    
    args = parser.parse_args()

    if args.consensus_type not in ["staple", "majority", "none"]:
        raise ValueError("consensus-type must be one of 'staple', 'majority', or 'none'")
    if args.consensus_type == "none":
        args.consensus_type = None
    
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
    
    timing_log = []  # [{case_id, phase, seconds}, ...]
    wall_start = time.time()
    
    for case_id in tqdm(case_ids, desc="Processing cases"):
        case_t0 = time.time()
        affine = None
        first_shape = None
        num_classes = None
        
        # --- PASS 1: Stream folds one at a time ---
        # Accumulate mean_probs, expected_entropy, per-class-max, labels
        # without holding all folds in memory.
        from .metric_functions import compute_entropy_map
        mean_probs_sum = None
        expected_entropy_sum = None
        max_probs = None           # per-class max across folds (for max_conf)
        labels_per_fold = {}
        loaded_folds = []
        
        for fold_idx, fold_path in fold_dirs:
            try:
                pred, aff = load_prediction(fold_path, case_id)
                pred = standardize_prediction(pred)
            except Exception as e:
                print(f"Warning: Failed to load prediction for case {case_id} from fold {fold_idx}: {e}")
                continue
            
            if affine is None and aff is not None:
                affine = aff
            
            if first_shape is None:
                first_shape = pred.shape[1:]
                num_classes = pred.shape[0]
            elif pred.shape[1:] != first_shape:
                print(f"Warning: Shape mismatch for case {case_id} fold {fold_idx}: {pred.shape[1:]} vs {first_shape}, skipping fold")
                del pred
                continue
            
            loaded_folds.append(fold_idx)
            
            # Accumulate mean_probs
            if mean_probs_sum is None:
                mean_probs_sum = pred.copy()  # already float32
            else:
                mean_probs_sum += pred
            
            # Track per-class max probability across folds
            if max_probs is None:
                max_probs = pred.copy()
            else:
                np.maximum(max_probs, pred, out=max_probs)
            
            # Accumulate expected entropy
            fold_entropy = compute_entropy_map(pred)
            if expected_entropy_sum is None:
                expected_entropy_sum = fold_entropy  # float32
            else:
                expected_entropy_sum += fold_entropy
            del fold_entropy
            
            # Store argmax labels (uint8 saves memory)
            labels_per_fold[fold_idx] = np.argmax(pred, axis=0).astype(np.uint8)
            
            # Free this fold's softmax immediately
            del pred
            _release_memory()
        
        if not loaded_folds:
            print(f"Warning: No predictions loaded for case {case_id}, skipping")
            continue
        
        t_pass1 = time.time() - case_t0
        timing_log.append({"case_id": case_id, "phase": "load_folds", "seconds": t_pass1})
        
        num_folds = len(loaded_folds)
        
        if affine is None:
            spatial_dims = len(first_shape) if first_shape else 3
            affine = np.eye(spatial_dims + 1)
        
        # Finalize mean_probs and expected_entropy
        mean_probs = mean_probs_sum
        mean_probs /= num_folds
        del mean_probs_sum
        expected_entropy_map = expected_entropy_sum
        expected_entropy_map /= num_folds
        del expected_entropy_sum
        
        consensus_seg = np.argmax(mean_probs, axis=0).astype(np.uint8)
        
        # Derive max_conf from per-class max probs + consensus
        consensus_int = consensus_seg.astype(np.intp)
        max_conf = np.take_along_axis(
            max_probs, consensus_int[np.newaxis, ...], axis=0
        ).squeeze(axis=0)
        del max_probs, consensus_int
        _release_memory()
        
        # Load ground truth
        gt = None
        t_gt0 = time.time()
        if args.gt_dir:
            gt_data = load_ground_truth(gt_dir=args.gt_dir, case_id=case_id, num_raters=args.num_raters, consensus_type=args.consensus_type)
            if gt_data is not None:
                gt, gt_affine = gt_data
                if affine is None:
                    affine = gt_affine
                # Apply label offset if requested
                if args.gt_label_offset != 0:
                    off = args.gt_label_offset
                    for key in ("raters", "consensus"):
                        arr = gt[key]
                        gt[key] = np.where(arr > 0, arr + off, arr)
        timing_log.append({"case_id": case_id, "phase": "load_gt", "seconds": time.time() - t_gt0})
        
        case_output_dir = os.path.join(args.output_dir, case_id)
        os.makedirs(case_output_dir, exist_ok=True)
        
        precomputed = {
            "mean_probs": mean_probs,
            "consensus_seg": consensus_seg,
            "labels_per_fold": labels_per_fold,
            "expected_entropy_map": expected_entropy_map,
            "max_conf": max_conf,
        }
        del mean_probs, expected_entropy_map, max_conf, consensus_seg
        _release_memory()
        
        for metric in metrics:
            m_t0 = time.time()
            try:
                metric.compute_case(
                    case_id=case_id,
                    preds_per_fold=None,
                    gt=gt,
                    affine=affine,
                    case_output_dir=case_output_dir,
                    save_maps=args.save_maps,
                    precomputed=precomputed
                )
            except Exception as e:
                print(f"Warning: Metric {metric.__class__.__name__} failed for case {case_id}: {e}")
                import traceback
                traceback.print_exc()
            m_elapsed = time.time() - m_t0
            timing_log.append({"case_id": case_id, "phase": metric.__class__.__name__, "seconds": m_elapsed})
        
        case_elapsed = time.time() - case_t0
        timing_log.append({"case_id": case_id, "phase": "TOTAL_CASE", "seconds": case_elapsed})
        tqdm.write(f"  {case_id}: {case_elapsed:.1f}s")
        
        # Free memory between cases
        del gt, precomputed, labels_per_fold
        _release_memory()

        # Log RSS so we can watch for leaks
        try:
            rss_kb = int(open("/proc/self/statm").read().split()[1]) * (os.sysconf("SC_PAGE_SIZE") // 1024)
            tqdm.write(f"    RSS after cleanup: {rss_kb/1024:.0f} MB")
        except Exception:
            pass
    
    print("Exporting summaries...")
    for metric in metrics:
        try:
            metric.export_summaries()
        except Exception as e:
            print(f"Warning: Failed to export summary for {metric.__class__.__name__}: {e}")
    
    # Save timing profile
    import pandas as pd
    wall_elapsed = time.time() - wall_start
    timing_df = pd.DataFrame(timing_log)
    timing_path = os.path.join(args.output_dir, "timing_profile.csv")
    timing_df.to_csv(timing_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Wall-clock time: {wall_elapsed:.1f}s ({wall_elapsed/60:.1f} min)")
    if len(timing_df) > 0:
        phase_totals = timing_df.groupby("phase")["seconds"].sum().sort_values(ascending=False)
        print(f"\nTime by phase (summed over all cases):")
        for phase, secs in phase_totals.items():
            print(f"  {phase:30s} {secs:8.1f}s  ({100*secs/wall_elapsed:5.1f}%)")
    print(f"{'='*60}")
    print(f"Done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
