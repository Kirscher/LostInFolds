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
    # Use local RNG for thread safety and reproducibility
    rng = np.random.default_rng(random_state)
    
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        raise ValueError(f"Need at least 2 samples for bootstrap, got {n}")
    
    if n_bootstrap < 2:
        raise ValueError(f"n_bootstrap must be at least 2, got {n_bootstrap}")
    
    if not (0 < ci_level < 1):
        raise ValueError(f"ci_level must be between 0 and 1 (exclusive), got {ci_level}")
    
    # Point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
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
            return stats.norm.cdf(z_alpha)  # Fall back to unadjusted percentile
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
    # Convert case_id to string to handle numeric or mixed types
    summary_keywords = ["mean", "std", "median", "min", "max", "sum", "count"]
    df["case_id"] = df["case_id"].fillna("").astype(str)
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


def bootstrap_aurc_file(
    filepath: str,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    method: str = "bca",
    random_state: Optional[int] = None
) -> List[BootstrapResult]:
    """
    Bootstrap AURC by resampling cases and recomputing the global metric.

    The file must be an ``aurc_per_case.csv`` containing per-case risk and
    confidence columns (produced by ``AURCMetric.export_summaries``).

    Parameters
    ----------
    filepath : str
        Path to ``aurc_per_case.csv``.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level.
    method : str
        Bootstrap CI method ('bca', 'percentile', 'basic').
    random_state : int, optional
        Random seed.

    Returns
    -------
    List[BootstrapResult]
        One result per AURC column (per-class + overall).
    """
    from .metric_functions import compute_aurc as _compute_aurc

    df = pd.read_csv(filepath)
    # Detect risk/confid column pairs
    risk_cols = [c for c in df.columns if c.startswith("risk_")]
    confid_cols = [c for c in df.columns if c.startswith("confid_")]

    # Build matched pairs: risk_class_1 <-> confid_class_1, risk_overall_risk <-> confid_overall_confid
    pairs = []
    for rc in risk_cols:
        suffix = rc.replace("risk_", "", 1)  # e.g. "class_1" or "overall_risk"
        # Try to find matching confid column
        if suffix == "overall_risk":
            cc = "confid_overall_confid"
            label = "overall_aurc"
        else:
            cc = f"confid_{suffix}"
            label = f"{suffix}"
        if cc in confid_cols:
            pairs.append((rc, cc, label))

    if not pairs:
        raise ValueError(f"No risk/confid column pairs found in {filepath}")

    results = []
    for risk_col, confid_col, aurc_label in pairs:
        risks_all = df[risk_col].values
        confids_all = df[confid_col].values
        n = len(risks_all)

        def _aurc_from_indices(idx):
            return _compute_aurc(risks_all[idx], confids_all[idx])

        # Wrap as a statistic over an index array
        index_array = np.arange(n)

        def statistic_func(data, _rc=risk_col, _cc=confid_col):
            idx = data.astype(int)
            return _compute_aurc(risks_all[idx], confids_all[idx])

        result = bootstrap_statistic(
            index_array,
            statistic_func=statistic_func,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            method=method,
            random_state=random_state
        )
        result.metric_name = f"aurc_{aurc_label}"
        result.statistic = "global"
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
    
    # Separate global-metric files that need special resampling
    aurc_per_case_file = None
    regular_files = []
    for f in csv_files:
        bname = os.path.basename(f)
        if bname == "aurc_per_case.csv":
            aurc_per_case_file = f
        elif bname == "aurc.csv":
            # Skip the single-row AURC summary; we bootstrap from per-case data
            continue
        else:
            regular_files.append(f)
    
    # Bootstrap regular per-case metric files
    for filepath in sorted(regular_files):
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
    
    # Bootstrap global AURC metric by resampling cases
    if aurc_per_case_file is not None:
        try:
            aurc_results = bootstrap_aurc_file(
                aurc_per_case_file,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                method=method,
                random_state=random_state
            )
            all_results.extend(aurc_results)
        except Exception as e:
            print(f"Warning: Failed to bootstrap AURC from {aurc_per_case_file}: {e}")
    
    
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


@dataclass
class ComparisonResult:
    """Container for method comparison results."""
    metric_name: str
    method_a_name: str
    method_b_name: str
    mean_a: float
    mean_b: float
    difference: float  # B - A
    ci_lower: float
    ci_upper: float
    ci_level: float
    p_value: float
    effect_size: float  # Cohen's d
    n_samples: int
    n_bootstrap: int
    significant: bool


def paired_bootstrap_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: Optional[int] = None,
    alternative: str = "two-sided"
) -> Tuple[float, float, float, float]:
    """
    Perform paired bootstrap test comparing two methods.
    
    Uses bootstrap resampling on paired differences to compute
    confidence interval and p-value for the difference.
    
    Parameters
    ----------
    data_a : np.ndarray
        Metric values for method A (per case)
    data_b : np.ndarray
        Metric values for method B (per case)
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level for CI
    random_state : int, optional
        Random seed
    alternative : str
        'two-sided', 'greater' (B > A), or 'less' (B < A)
        
    Returns
    -------
    Tuple[float, float, float, float]
        (difference, ci_lower, ci_upper, p_value)
    """
    # Use local RNG for thread safety and reproducibility
    rng = np.random.default_rng(random_state)
    
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    
    if len(data_a) != len(data_b):
        raise ValueError(f"Arrays must have same length: {len(data_a)} vs {len(data_b)}")
    
    n = len(data_a)
    
    # Compute paired differences
    differences = data_b - data_a
    observed_diff = np.mean(differences)
    
    # Bootstrap the differences
    bootstrap_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = np.mean(differences[resample_idx])
    
    # Confidence interval
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
    
    # P-value: proportion of bootstrap samples on wrong side of zero
    # Using the shifted bootstrap distribution (null hypothesis: no difference)
    shifted_diffs = bootstrap_diffs - observed_diff
    
    if alternative == "two-sided":
        p_value = np.mean(np.abs(shifted_diffs) >= np.abs(observed_diff)) 
    elif alternative == "greater":  # H1: B > A (diff > 0)
        p_value = np.mean(shifted_diffs >= observed_diff)
    elif alternative == "less":  # H1: B < A (diff < 0)
        p_value = np.mean(shifted_diffs <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    # Ensure p-value is at least 1/n_bootstrap (can't be exactly 0)
    p_value = max(p_value, 1.0 / n_bootstrap)
    
    return observed_diff, ci_lower, ci_upper, p_value


def cohens_d(data_a: np.ndarray, data_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.
    
    Uses the standard deviation of the differences as the denominator
    (appropriate for paired/repeated measures).
    
    Parameters
    ----------
    data_a, data_b : np.ndarray
        Paired metric values
        
    Returns
    -------
    float
        Cohen's d effect size
    """
    differences = np.asarray(data_b) - np.asarray(data_a)
    d = np.mean(differences) / (np.std(differences, ddof=1) + 1e-10)
    return d


def _compare_aurc_per_case(
    filepath_a: str,
    filepath_b: str,
    method_a_name: str = "Method A",
    method_b_name: str = "Method B",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: Optional[int] = None,
) -> List[ComparisonResult]:
    """
    Paired bootstrap comparison of AURC between two methods.

    Resamples cases (rows) with replacement, recomputes AURC for each
    method on the resample, and tests the difference.
    """
    from .metric_functions import compute_aurc as _compute_aurc

    df_a = pd.read_csv(filepath_a)
    df_b = pd.read_csv(filepath_b)
    merged = pd.merge(df_a, df_b, on="case_id", suffixes=("_a", "_b"))
    n = len(merged)
    if n < 2:
        raise ValueError("Insufficient matched cases for AURC comparison")

    risk_cols_a = sorted([c for c in merged.columns if c.startswith("risk_") and c.endswith("_a")])
    results = []

    for rc_a in risk_cols_a:
        suffix = rc_a.replace("risk_", "").replace("_a", "")
        if suffix == "overall_risk":
            cc_a, rc_b, cc_b = "confid_overall_confid_a", "risk_overall_risk_b", "confid_overall_confid_b"
            label = "aurc_overall"
        else:
            cc_a = f"confid_{suffix}_a"
            rc_b = f"risk_{suffix}_b"
            cc_b = f"confid_{suffix}_b"
            label = f"aurc_{suffix}"
        if not all(c in merged.columns for c in [cc_a, rc_b, cc_b]):
            continue

        ra = merged[rc_a].values
        ca = merged[cc_a].values
        rb = merged[rc_b].values
        cb = merged[cc_b].values

        aurc_a_point = _compute_aurc(ra, ca)
        aurc_b_point = _compute_aurc(rb, cb)
        observed_diff = aurc_b_point - aurc_a_point

        rng = np.random.default_rng(random_state)
        boot_diffs = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_diffs[i] = _compute_aurc(rb[idx], cb[idx]) - _compute_aurc(ra[idx], ca[idx])

        alpha = 1 - ci_level
        ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
        shifted = boot_diffs - np.mean(boot_diffs)
        p_value = max(np.mean(np.abs(shifted) >= np.abs(observed_diff)), 1.0 / n_bootstrap)
        significant = (ci_lower > 0) or (ci_upper < 0)
        # Cohen's d on per-case risk differences as proxy
        per_case_diff = rb - ra
        effect = float(np.mean(per_case_diff) / (np.std(per_case_diff, ddof=1) + 1e-10))

        results.append(ComparisonResult(
            metric_name=label,
            method_a_name=method_a_name,
            method_b_name=method_b_name,
            mean_a=aurc_a_point,
            mean_b=aurc_b_point,
            difference=observed_diff,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level,
            p_value=p_value,
            effect_size=effect,
            n_samples=n,
            n_bootstrap=n_bootstrap,
            significant=significant,
        ))

    return results


def compare_methods(
    metrics_dir_a: str,
    metrics_dir_b: str,
    method_a_name: str = "Method A",
    method_b_name: str = "Method B",
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    random_state: Optional[int] = None,
    metric_pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Compare metrics between two methods using paired bootstrap tests.
    
    Parameters
    ----------
    metrics_dir_a : str
        Directory with metrics from method A
    metrics_dir_b : str
        Directory with metrics from method B
    method_a_name : str
        Display name for method A
    method_b_name : str
        Display name for method B
    n_bootstrap : int
        Number of bootstrap iterations
    ci_level : float
        Confidence level
    random_state : int, optional
        Random seed
    metric_pattern : str
        Glob pattern for metric files
        
    Returns
    -------
    pd.DataFrame
        Comparison results for all metrics
    """
    # Find matching metric files
    files_a = {os.path.basename(f): f for f in glob.glob(os.path.join(metrics_dir_a, metric_pattern))}
    files_b = {os.path.basename(f): f for f in glob.glob(os.path.join(metrics_dir_b, metric_pattern))}
    
    # Filter out bootstrap results
    files_a = {k: v for k, v in files_a.items() if not k.startswith("bootstrap_")}
    files_b = {k: v for k, v in files_b.items() if not k.startswith("bootstrap_")}
    
    common_files = set(files_a.keys()) & set(files_b.keys())
    
    if not common_files:
        raise ValueError(f"No common metric files found between {metrics_dir_a} and {metrics_dir_b}")
    
    results = []
    
    for filename in sorted(common_files):
        # Skip single-row AURC summary; we compare via aurc_per_case.csv
        if filename == "aurc.csv":
            continue
        
        # Handle global AURC comparison via per-case resampling
        if filename == "aurc_per_case.csv":
            try:
                results.extend(_compare_aurc_per_case(
                    files_a[filename], files_b[filename],
                    method_a_name=method_a_name,
                    method_b_name=method_b_name,
                    n_bootstrap=n_bootstrap,
                    ci_level=ci_level,
                    random_state=random_state,
                ))
            except Exception as e:
                print(f"Warning: Failed to compare AURC: {e}")
            continue
        
        try:
            df_a, metric_name = load_metric_csv(files_a[filename])
            df_b, _ = load_metric_csv(files_b[filename])
            
            # Merge on case_id to ensure alignment
            merged = pd.merge(df_a, df_b, on="case_id", suffixes=("_a", "_b"))
            
            if len(merged) < 2:
                print(f"Warning: Insufficient matched cases for {filename}, skipping")
                continue
            
            col_a = f"{metric_name}_a"
            col_b = f"{metric_name}_b"
            
            values_a = merged[col_a].values
            values_b = merged[col_b].values
            
            # Perform paired bootstrap test
            diff, ci_lower, ci_upper, p_value = paired_bootstrap_test(
                values_a, values_b,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                random_state=random_state
            )
            
            # Compute effect size
            effect = cohens_d(values_a, values_b)
            
            # Determine significance
            significant = (ci_lower > 0) or (ci_upper < 0)
            
            results.append(ComparisonResult(
                metric_name=metric_name,
                method_a_name=method_a_name,
                method_b_name=method_b_name,
                mean_a=np.mean(values_a),
                mean_b=np.mean(values_b),
                difference=diff,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ci_level=ci_level,
                p_value=p_value,
                effect_size=effect,
                n_samples=len(merged),
                n_bootstrap=n_bootstrap,
                significant=significant
            ))
            
        except Exception as e:
            print(f"Warning: Failed to compare {filename}: {e}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "metric": r.metric_name,
            f"mean_{r.method_a_name}": r.mean_a,
            f"mean_{r.method_b_name}": r.mean_b,
            "difference": r.difference,
            f"ci_{int(r.ci_level*100)}_lower": r.ci_lower,
            f"ci_{int(r.ci_level*100)}_upper": r.ci_upper,
            "p_value": r.p_value,
            "effect_size_d": r.effect_size,
            "significant": r.significant,
            "n_samples": r.n_samples,
        }
        for r in results
    ])
    
    return df


def format_comparison_table(
    df: pd.DataFrame,
    format_style: str = "markdown",
    decimal_places: int = 4
) -> str:
    """
    Format comparison results as a readable table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Comparison results
    format_style : str
        'markdown', 'latex', or 'plain'
    decimal_places : int
        Decimal places for numeric values
        
    Returns
    -------
    str
        Formatted table
    """
    df_fmt = df.copy()
    
    # Format difference with CI
    ci_cols = [c for c in df.columns if c.startswith("ci_") and c.endswith("_lower")]
    if ci_cols:
        ci_prefix = ci_cols[0].replace("_lower", "")
        lower_col = f"{ci_prefix}_lower"
        upper_col = f"{ci_prefix}_upper"
        
        df_fmt["diff_with_ci"] = df_fmt.apply(
            lambda row: f"{row['difference']:.{decimal_places}f} [{row[lower_col]:.{decimal_places}f}, {row[upper_col]:.{decimal_places}f}]",
            axis=1
        )
    
    # Format p-value
    df_fmt["p_value_fmt"] = df_fmt["p_value"].apply(
        lambda p: f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"
    )
    
    # Add significance stars
    df_fmt["sig"] = df_fmt["p_value"].apply(
        lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    )
    
    # Format effect size with interpretation
    def interpret_effect(d):
        d_abs = abs(d)
        if d_abs < 0.2:
            return f"{d:.3f} (negligible)"
        elif d_abs < 0.5:
            return f"{d:.3f} (small)"
        elif d_abs < 0.8:
            return f"{d:.3f} (medium)"
        else:
            return f"{d:.3f} (large)"
    
    df_fmt["effect_interp"] = df_fmt["effect_size_d"].apply(interpret_effect)
    
    # Select columns for display
    display_cols = ["metric"]
    mean_cols = [c for c in df.columns if c.startswith("mean_")]
    display_cols.extend(mean_cols)
    display_cols.extend(["diff_with_ci", "p_value_fmt", "sig", "effect_interp"])
    
    df_display = df_fmt[display_cols].copy()
    df_display.columns = [c.replace("_fmt", "").replace("_interp", "") for c in display_cols]
    
    if format_style == "markdown":
        return df_display.to_markdown(index=False)
    elif format_style == "latex":
        return df_display.to_latex(index=False)
    else:
        return df_display.to_string(index=False)


def main():
    """CLI entry point for bootstrap analysis."""
    parser = argparse.ArgumentParser(
        description="Bootstrap resampling for ensemble metrics with confidence intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bootstrap all metrics in a directory
  python -m src.ensemble_metrics.bootstrap --metrics-dir ./metrics_all --output bootstrap_results.csv
  
  # Use specific confidence level and method
  python -m src.ensemble_metrics.bootstrap --metrics-dir ./metrics_all --ci-level 0.99 --method bca
  
  # Compute specific statistics
  python -m src.ensemble_metrics.bootstrap --metrics-dir ./metrics_all --statistics mean,median,iqr
  
  # Single file analysis
  python -m src.ensemble_metrics.bootstrap --metric-file ./metrics_all/ace.csv --output ace_bootstrap.csv
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
