#!/usr/bin/env python3
"""Tests for bootstrap resampling module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ensemble_metrics.bootstrap import (
    BootstrapResult,
    bootstrap_statistic,
    bootstrap_metric_file,
    bootstrap_metrics_directory,
    load_metric_csv,
    format_results_table,
    _bca_interval,
)


class TestBootstrapStatistic:
    """Tests for the core bootstrap_statistic function."""
    
    def test_basic_mean_bootstrap(self):
        """Test basic bootstrap for mean statistic."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        
        result = bootstrap_statistic(
            data,
            statistic_func=np.mean,
            n_bootstrap=1000,
            ci_level=0.95,
            method="percentile",
            random_state=42
        )
        
        assert isinstance(result, BootstrapResult)
        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.n_samples == 100
        assert result.n_bootstrap == 1000
        assert result.method == "percentile"
        # Point estimate should be close to true mean
        assert abs(result.point_estimate - 10) < 1
    
    def test_bca_method(self):
        """Test BCa bootstrap method."""
        np.random.seed(42)
        data = np.random.normal(5, 1, 50)
        
        result = bootstrap_statistic(
            data,
            statistic_func=np.mean,
            n_bootstrap=2000,
            ci_level=0.95,
            method="bca",
            random_state=42
        )
        
        assert result.method == "bca"
        assert result.ci_lower < result.point_estimate < result.ci_upper
        # CI should contain the true mean with high probability
        assert result.ci_lower < 5.5
        assert result.ci_upper > 4.5
    
    def test_basic_method(self):
        """Test basic (reverse percentile) bootstrap method."""
        np.random.seed(42)
        data = np.random.exponential(2, 80)
        
        result = bootstrap_statistic(
            data,
            statistic_func=np.median,
            n_bootstrap=1000,
            ci_level=0.95,
            method="basic",
            random_state=42
        )
        
        assert result.method == "basic"
        assert result.ci_lower < result.ci_upper
    
    def test_different_ci_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        result_90 = bootstrap_statistic(data, ci_level=0.90, n_bootstrap=2000, random_state=42)
        result_95 = bootstrap_statistic(data, ci_level=0.95, n_bootstrap=2000, random_state=42)
        result_99 = bootstrap_statistic(data, ci_level=0.99, n_bootstrap=2000, random_state=42)
        
        # Higher confidence level should give wider intervals
        width_90 = result_90.ci_upper - result_90.ci_lower
        width_95 = result_95.ci_upper - result_95.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower
        
        assert width_90 < width_95 < width_99
    
    def test_std_error_computed(self):
        """Test that standard error is computed."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 50)
        
        result = bootstrap_statistic(data, n_bootstrap=1000, random_state=42)
        
        assert result.std_error > 0
        # For mean of normal, SE ≈ σ/√n ≈ 1/√50 ≈ 0.14
        assert 0.05 < result.std_error < 0.3
    
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        result1 = bootstrap_statistic(data, n_bootstrap=1000, random_state=12345)
        result2 = bootstrap_statistic(data, n_bootstrap=1000, random_state=12345)
        
        assert result1.point_estimate == result2.point_estimate
        assert result1.ci_lower == result2.ci_lower
        assert result1.ci_upper == result2.ci_upper
    
    def test_insufficient_data_raises(self):
        """Test that insufficient data raises an error."""
        data = np.array([1])
        
        with pytest.raises(ValueError, match="at least 2 samples"):
            bootstrap_statistic(data)
    
    def test_unknown_method_raises(self):
        """Test that unknown method raises an error."""
        data = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_statistic(data, method="invalid_method")
    
    def test_custom_statistic_function(self):
        """Test bootstrap with custom statistic function."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        # Custom: interquartile range
        def iqr(x):
            return np.percentile(x, 75) - np.percentile(x, 25)
        
        result = bootstrap_statistic(
            data,
            statistic_func=iqr,
            n_bootstrap=1000,
            random_state=42
        )
        
        # IQR of standard normal ≈ 1.35
        assert 0.8 < result.point_estimate < 1.8
        assert result.ci_lower < result.ci_upper


class TestLoadMetricCSV:
    """Tests for CSV loading functionality."""
    
    def test_load_basic_csv(self, tmp_path):
        """Test loading a basic metric CSV."""
        csv_content = """case_id,ace
prostate_00,0.106
prostate_01,0.168
prostate_02,0.216
mean,0.163
"""
        csv_file = tmp_path / "ace.csv"
        csv_file.write_text(csv_content)
        
        df, metric_name = load_metric_csv(str(csv_file))
        
        assert metric_name == "ace"
        assert len(df) == 3  # Excludes 'mean' row
        assert "prostate_00" in df["case_id"].values
        assert "mean" not in df["case_id"].values
    
    def test_filters_summary_rows(self, tmp_path):
        """Test that summary rows are filtered out."""
        csv_content = """case_id,dice
case_01,0.85
case_02,0.90
case_03,0.88
mean,0.877
std,0.025
median,0.88
min,0.85
max,0.90
"""
        csv_file = tmp_path / "dice.csv"
        csv_file.write_text(csv_content)
        
        df, metric_name = load_metric_csv(str(csv_file))
        
        assert len(df) == 3
        for kw in ["mean", "std", "median", "min", "max"]:
            assert kw not in df["case_id"].values


class TestBootstrapMetricFile:
    """Tests for bootstrap_metric_file function."""
    
    def test_bootstrap_single_file(self, tmp_path):
        """Test bootstrapping a single metric file."""
        csv_content = """case_id,ncc
case_00,0.95
case_01,0.92
case_02,0.88
case_03,0.91
case_04,0.93
case_05,0.89
case_06,0.94
case_07,0.90
case_08,0.87
case_09,0.91
mean,0.91
"""
        csv_file = tmp_path / "ncc.csv"
        csv_file.write_text(csv_content)
        
        results = bootstrap_metric_file(
            str(csv_file),
            n_bootstrap=500,
            statistics=["mean", "median"],
            random_state=42
        )
        
        assert len(results) == 2
        assert results[0].metric_name == "ncc"
        assert results[0].statistic == "mean"
        assert results[1].statistic == "median"
    
    def test_all_statistics(self, tmp_path):
        """Test all available statistics."""
        np.random.seed(42)
        values = np.random.uniform(0, 1, 20)
        
        csv_content = "case_id,metric\n"
        csv_content += "\n".join([f"case_{i:02d},{v:.4f}" for i, v in enumerate(values)])
        csv_file = tmp_path / "metric.csv"
        csv_file.write_text(csv_content)
        
        results = bootstrap_metric_file(
            str(csv_file),
            n_bootstrap=500,
            statistics=["mean", "median", "std", "min", "max", "iqr"],
            random_state=42
        )
        
        assert len(results) == 6
        stat_names = [r.statistic for r in results]
        assert set(stat_names) == {"mean", "median", "std", "min", "max", "iqr"}


class TestBootstrapMetricsDirectory:
    """Tests for bootstrap_metrics_directory function."""
    
    def test_bootstrap_directory(self, tmp_path):
        """Test bootstrapping all CSVs in a directory."""
        # Create multiple metric files
        np.random.seed(42)
        
        for metric_name in ["ace", "ncc", "dice"]:
            values = np.random.uniform(0, 1, 15)
            csv_content = f"case_id,{metric_name}\n"
            csv_content += "\n".join([f"case_{i:02d},{v:.4f}" for i, v in enumerate(values)])
            csv_content += f"\nmean,{np.mean(values):.4f}"
            (tmp_path / f"{metric_name}.csv").write_text(csv_content)
        
        output_file = tmp_path / "bootstrap_results.csv"
        
        df = bootstrap_metrics_directory(
            str(tmp_path),
            output_file=str(output_file),
            n_bootstrap=500,
            statistics=["mean"],
            random_state=42
        )
        
        assert len(df) == 3  # 3 metrics × 1 statistic
        assert set(df["metric"].values) == {"ace", "ncc", "dice"}
        assert output_file.exists()
    
    def test_excludes_bootstrap_results_file(self, tmp_path):
        """Test that existing bootstrap results files are excluded."""
        # Create metric file
        csv_content = "case_id,ace\ncase_00,0.1\ncase_01,0.2\ncase_02,0.3"
        (tmp_path / "ace.csv").write_text(csv_content)
        
        # Create existing bootstrap results (should be ignored)
        (tmp_path / "bootstrap_results.csv").write_text("metric,statistic,value\nace,mean,0.2")
        
        df = bootstrap_metrics_directory(
            str(tmp_path),
            n_bootstrap=100,
            statistics=["mean"],
            random_state=42
        )
        
        # Should only process ace.csv, not bootstrap_results.csv
        assert len(df) == 1
        assert df["metric"].values[0] == "ace"


class TestFormatResultsTable:
    """Tests for results formatting."""
    
    def test_markdown_format(self):
        """Test markdown table formatting."""
        df = pd.DataFrame({
            "metric": ["ace", "ace", "ncc", "ncc"],
            "statistic": ["mean", "median", "mean", "median"],
            "point_estimate": [0.15, 0.14, 0.92, 0.93],
            "ci_95_lower": [0.12, 0.11, 0.89, 0.90],
            "ci_95_upper": [0.18, 0.17, 0.95, 0.96],
            "std_error": [0.01, 0.01, 0.02, 0.02],
            "n_samples": [10, 10, 10, 10],
            "n_bootstrap": [1000, 1000, 1000, 1000],
            "method": ["bca", "bca", "bca", "bca"]
        })
        
        result = format_results_table(df, format_style="markdown")
        
        assert isinstance(result, str)
        assert "ace" in result
        assert "ncc" in result
        assert "mean" in result
        assert "median" in result


class TestBCAInterval:
    """Tests for BCa interval computation."""
    
    def test_bca_handles_skewed_data(self):
        """Test BCa handles skewed distributions."""
        np.random.seed(42)
        # Exponential distribution is right-skewed
        data = np.random.exponential(2, 50)
        
        # Generate bootstrap samples
        bootstrap_stats = np.zeros(2000)
        for i in range(2000):
            resample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats[i] = np.mean(resample)
        
        theta_hat = np.mean(data)
        
        ci_lower, ci_upper = _bca_interval(
            data, bootstrap_stats, theta_hat, np.mean, 0.05
        )
        
        assert ci_lower < theta_hat < ci_upper
        assert ci_lower > 0  # Exponential mean is always positive
    
    def test_bca_vs_percentile_on_symmetric(self):
        """Test that BCa and percentile give similar results on symmetric data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        result_bca = bootstrap_statistic(
            data, method="bca", n_bootstrap=5000, random_state=42
        )
        result_pct = bootstrap_statistic(
            data, method="percentile", n_bootstrap=5000, random_state=42
        )
        
        # For symmetric distributions, BCa and percentile should be similar
        assert abs(result_bca.ci_lower - result_pct.ci_lower) < 0.1
        assert abs(result_bca.ci_upper - result_pct.ci_upper) < 0.1


class TestIntegration:
    """Integration tests for the full bootstrap workflow."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from CSV to formatted results."""
        np.random.seed(42)
        
        # Create realistic metric files
        metrics = {
            "ace": np.random.uniform(0.1, 0.3, 20),
            "dice": np.random.uniform(0.7, 0.95, 20),
            "ncc": np.random.uniform(0.85, 0.99, 20),
        }
        
        for metric_name, values in metrics.items():
            csv_content = f"case_id,{metric_name}\n"
            csv_content += "\n".join([f"case_{i:02d},{v:.6f}" for i, v in enumerate(values)])
            csv_content += f"\nmean,{np.mean(values):.6f}"
            (tmp_path / f"{metric_name}.csv").write_text(csv_content)
        
        output_file = tmp_path / "bootstrap_summary.csv"
        
        # Run bootstrap analysis
        df = bootstrap_metrics_directory(
            str(tmp_path),
            output_file=str(output_file),
            n_bootstrap=1000,
            ci_level=0.95,
            method="bca",
            statistics=["mean", "median", "std"],
            random_state=42
        )
        
        # Verify results
        assert len(df) == 9  # 3 metrics × 3 statistics
        assert output_file.exists()
        
        # Check that CIs make sense
        for _, row in df.iterrows():
            assert row["ci_95_lower"] <= row["point_estimate"] <= row["ci_95_upper"]
        
        # Format and verify output
        table = format_results_table(df, format_style="markdown")
        assert "ace" in table
        assert "dice" in table
        assert "ncc" in table
