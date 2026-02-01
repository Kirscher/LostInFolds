#!/usr/bin/env python3
"""
Bootstrap resampling for ensemble metrics with confidence intervals.

This module implements state-of-the-art bootstrap resampling with replacement
to compute confidence intervals for metric statistics. Supports:
- Basic percentile method
- BCa (Bias-Corrected and Accelerated) confidence intervals
- Multiple metrics aggregation per dataset
"""

import argparse
import os
import glob
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    metric_name: str
    statistic: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_samples: int
    n_bootstrap: int
    method: str
    std_error: float


def bootstrap_statistic(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: str = "bca",
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of metric values (one per case)
    statistic_func : callable
        Function to compute the statistic (e.g., np.mean, np.median)
    n_bootstrap : int
        Number of bootstrap resamples
    ci_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        Method for CI computation: 'percentile', 'bca', or 'basic'
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    BootstrapResult
        Container with point estimate, CI bounds, and metadata
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        raise ValueError(f"Need at least 2 samples for bootstrap, got {n}")
    
    # Point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = np.random.randint(0, n, size=n)
        resample = data[resample_idx]
        bootstrap_stats[i] = statistic_func(resample)
    
    # Standard error
    std_error = np.std(bootstrap_stats, ddof=1)
    
    # Compute confidence interval based on method
    alpha = 1 - ci_level
    
    if method == "percentile":
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
    elif method == "basic":
        # Basic bootstrap (reverse percentile)
        q_lower = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        q_upper = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_lower = 2 * point_estimate - q_lower
        ci_upper = 2 * point_estimate - q_upper
        
    elif method == "bca":
        # BCa (Bias-Corrected and Accelerated) - recommended method
        ci_lower, ci_upper = _bca_interval(
            data, bootstrap_stats, point_estimate, statistic_func, alpha
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percentile', 'basic', or 'bca'")
    
    return BootstrapResult(
        metric_name="",  # Will be filled by caller
        statistic=statistic_func.__name__ if hasattr(statistic_func, '__name__') else str(statistic_func),
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_level=ci_level,
        n_samples=n,
        n_bootstrap=n_bootstrap,
        method=method,
        std_error=std_error
    )


def _bca_interval(
    data: np.ndarray,
    bootstrap_stats: np.ndarray,
    theta_hat: float,
    statistic_func: callable,
    alpha: float
) -> Tuple[float, float]:
    """
    Compute BCa (Bias-Corrected and Accelerated) confidence interval.
    
    The BCa method corrects for both bias and skewness in the bootstrap
    distribution, making it more accurate than simple percentile methods.
    
    Parameters
    ----------
    data : np.ndarray
        Original data
    bootstrap_stats : np.ndarray
        Bootstrap statistic values
    theta_hat : float
        Point estimate from original data
    statistic_func : callable
        Function used to compute statistic
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
        
    Returns
    -------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    n = len(data)
    
    # Bias correction factor (z0)
    # Proportion of bootstrap estimates less than the original estimate
    prop_less = np.mean(bootstrap_stats < theta_hat)
    # Handle edge cases
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)
    
    # Acceleration factor (a) using jackknife
    jackknife_stats = np.zeros(n)
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats[i] = statistic_func(jackknife_sample)
    
    jackknife_mean = np.mean(jackknife_stats)
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denominator = np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5
    
    # Handle case where denominator is zero or very small
    if np.abs(denominator) < 1e-10:
        a = 0.0
    else:
        a = numerator / (6.0 * denominator)
    
    # Compute adjusted percentiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
    
    # BCa adjusted percentiles
    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        denominator = 1 - a * (z0 + z_alpha)
        if np.abs(denominator) < 1e-10:
            return z_alpha  # Fall back to unadjusted
        adjusted_z = z0 + numerator / denominator
        return stats.norm.cdf(adjusted_z)
    
    p_lower = adjusted_percentile(z_alpha_lower)
    p_upper = adjusted_percentile(z_alpha_upper)
    
    # Clip to valid range
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)
    
    ci_lower = np.percentile(bootstrap_stats, 100 * p_lower)
    ci_upper = np.percentile(bootstrap_stats, 100 * p_upper)
    
    return ci_lower, ci_upper


def load_metric_csv(filepath: str) -> Tuple[pd.DataFrame, str]:
    """
    Load a metric CSV file and extract the metric name.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
        
    Returns
    -------
    Tuple[pd.DataFrame, str]
        DataFrame with case_id and metric values, and metric name
    """
    df = pd.read_csv(filepath)
    
    # Get metric name from columns (second column typically)
    metric_name = [col for col in df.columns if col != "case_id"][0]
    
    # Filter out summary rows (mean, std, etc.)
    summary_keywords = ["mean", "std", "median", "min", "max", "sum", "count"]
    mask = ~df["case_id"].str.lower().isin(summary_keywords)
    df_cases = df[mask].copy()
    
    return df_cases, metric_name


def bootstrap_metric_file(
    filepath: str,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: str = "bca",
    statistics: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> List[BootstrapResult]:
    """
    Perform bootstrap analysis on a metric CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to metric CSV file
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level (e.g., 0.95)
    method : str
        Bootstrap method ('bca', 'percentile', 'basic')
    statistics : List[str], optional
        Statistics to compute. Default: ['mean', 'median', 'std']
    random_state : int, optional
        Random seed
        
    Returns
    -------
    List[BootstrapResult]
        Bootstrap results for each statistic
    """
    if statistics is None:
        statistics = ["mean", "median", "std"]
    
    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": lambda x: np.std(x, ddof=1),
        "min": np.min,
        "max": np.max,
        "iqr": lambda x: np.percentile(x, 75) - np.percentile(x, 25),
    }
    
    df, metric_name = load_metric_csv(filepath)
    values = df[metric_name].values
    
    results = []
    for stat_name in statistics:
        if stat_name not in stat_funcs:
            raise ValueError(f"Unknown statistic: {stat_name}. Available: {list(stat_funcs.keys())}")
        
        result = bootstrap_statistic(
            values,
            statistic_func=stat_funcs[stat_name],
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            method=method,
            random_state=random_state
        )
        result.metric_name = metric_name
        result.statistic = stat_name
        results.append(result)
    
    return results


def bootstrap_metrics_directory(
    metrics_dir: str,
    output_file: Optional[str] = None,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: str = "bca",
    statistics: Optional[List[str]] = None,
    random_state: Optional[int] = None,
    metric_pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Perform bootstrap analysis on all metric CSV files in a directory.
    
    Parameters
    ----------
    metrics_dir : str
        Directory containing metric CSV files
    output_file : str, optional
        Path to save results CSV
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level
    method : str
        Bootstrap method
    statistics : List[str], optional
        Statistics to compute
    random_state : int, optional
        Random seed
    metric_pattern : str
        Glob pattern for metric files
        
    Returns
    -------
    pd.DataFrame
        DataFrame with bootstrap results for all metrics
    """
    csv_files = glob.glob(os.path.join(metrics_dir, metric_pattern))
    
    # Filter out summary files that might exist
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("bootstrap_")]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {metrics_dir} matching pattern {metric_pattern}")
    
    all_results = []
    
    for filepath in sorted(csv_files):
        try:
            results = bootstrap_metric_file(
                filepath,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                method=method,
                statistics=statistics,
                random_state=random_state
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "metric": r.metric_name,
            "statistic": r.statistic,
            "point_estimate": r.point_estimate,
            f"ci_{int(r.ci_level*100)}_lower": r.ci_lower,
            f"ci_{int(r.ci_level*100)}_upper": r.ci_upper,
            "std_error": r.std_error,
            "n_samples": r.n_samples,
            "n_bootstrap": r.n_bootstrap,
            "method": r.method
        }
        for r in all_results
    ])
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df


def format_results_table(
    df: pd.DataFrame,
    format_style: str = "markdown",
    decimal_places: int = 4
) -> str:
    """
    Format bootstrap results as a readable table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Bootstrap results DataFrame
    format_style : str
        Output format: 'markdown', 'latex', 'plain'
    decimal_places : int
        Number of decimal places for values
        
    Returns
    -------
    str
        Formatted table string
    """
    # Create formatted value column
    df_formatted = df.copy()
    
    # Detect CI columns
    ci_cols = [c for c in df.columns if c.startswith("ci_") and c.endswith("_lower")]
    if ci_cols:
        ci_prefix = ci_cols[0].replace("_lower", "")
        lower_col = f"{ci_prefix}_lower"
        upper_col = f"{ci_prefix}_upper"
        
        df_formatted["value_with_ci"] = df_formatted.apply(
            lambda row: f"{row['point_estimate']:.{decimal_places}f} [{row[lower_col]:.{decimal_places}f}, {row[upper_col]:.{decimal_places}f}]",
            axis=1
        )
    
    # Pivot for better readability
    pivot_df = df_formatted.pivot(
        index="metric",
        columns="statistic",
        values="value_with_ci" if "value_with_ci" in df_formatted.columns else "point_estimate"
    )
    
    if format_style == "markdown":
        return pivot_df.to_markdown()
    elif format_style == "latex":
        return pivot_df.to_latex()
    else:
        return pivot_df.to_string()


def main():
    """CLI entry point for bootstrap analysis."""
    parser = argparse.ArgumentParser(
        description="Bootstrap resampling for ensemble metrics with confidence intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bootstrap all metrics in a directory
  python -m ensemble_metrics.bootstrap --metrics-dir ./metrics_all --output bootstrap_results.csv
  
  # Use specific confidence level and method
  python -m ensemble_metrics.bootstrap --metrics-dir ./metrics_all --ci-level 0.99 --method bca
  
  # Compute specific statistics
  python -m ensemble_metrics.bootstrap --metrics-dir ./metrics_all --statistics mean,median,iqr
  
  # Single file analysis
  python -m ensemble_metrics.bootstrap --metric-file ./metrics_all/ace.csv --output ace_bootstrap.csv
"""
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metrics-dir",
        type=str,
        help="Directory containing metric CSV files"
    )
    input_group.add_argument(
        "--metric-file",
        type=str,
        help="Single metric CSV file to analyze"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for results CSV"
    )
    
    # Bootstrap parameters
    parser.add_argument(
        "--n-bootstrap", "-n",
        type=int,
        default=10000,
        help="Number of bootstrap iterations (default: 10000)"
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95 for 95%% CI)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bca", "percentile", "basic"],
        default="bca",
        help="Bootstrap CI method: 'bca' (recommended), 'percentile', or 'basic' (default: bca)"
    )
    parser.add_argument(
        "--statistics",
        type=str,
        default="mean,median,std",
        help="Comma-separated statistics to compute (default: mean,median,std). Available: mean,median,std,min,max,iqr"
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "latex", "plain"],
        default="markdown",
        help="Output format for summary table (default: markdown)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for metric files (default: *.csv)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress summary output to stdout"
    )
    
    args = parser.parse_args()
    
    # Parse statistics
    statistics = [s.strip() for s in args.statistics.split(",")]
    
    # Determine output file
    output_file = args.output
    if output_file is None:
        if args.metrics_dir:
            output_file = os.path.join(args.metrics_dir, "bootstrap_results.csv")
        else:
            base = os.path.splitext(args.metric_file)[0]
            output_file = f"{base}_bootstrap.csv"
    
    print(f"Bootstrap Analysis")
    print(f"==================")
    print(f"Method: {args.method.upper()}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Confidence level: {args.ci_level * 100:.0f}%")
    print(f"Statistics: {', '.join(statistics)}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()
    
    # Run bootstrap analysis
    if args.metrics_dir:
        print(f"Processing metrics in: {args.metrics_dir}")
        df = bootstrap_metrics_directory(
            metrics_dir=args.metrics_dir,
            output_file=output_file,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            method=args.method,
            statistics=statistics,
            random_state=args.seed,
            metric_pattern=args.pattern
        )
    else:
        print(f"Processing file: {args.metric_file}")
        results = bootstrap_metric_file(
            filepath=args.metric_file,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            method=args.method,
            statistics=statistics,
            random_state=args.seed
        )
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "metric": r.metric_name,
                "statistic": r.statistic,
                "point_estimate": r.point_estimate,
                f"ci_{int(r.ci_level*100)}_lower": r.ci_lower,
                f"ci_{int(r.ci_level*100)}_upper": r.ci_upper,
                "std_error": r.std_error,
                "n_samples": r.n_samples,
                "n_bootstrap": r.n_bootstrap,
                "method": r.method
            }
            for r in results
        ])
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    # Print summary table
    if not args.quiet:
        print()
        print("Results Summary")
        print("---------------")
        print(format_results_table(df, format_style=args.format))
    
    return df


if __name__ == "__main__":
    main()
