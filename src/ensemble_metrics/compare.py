#!/usr/bin/env python3
"""CLI entry point for comparing metrics between two methods."""

import argparse
import os
import sys

import pandas as pd

from .bootstrap import (
    compare_methods,
    format_comparison_table,
)


def main():
    """CLI entry point for method comparison."""
    parser = argparse.ArgumentParser(
        description="Compare ensemble metrics between two methods using paired bootstrap tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two methods
  python -m ensemble_metrics.compare \\
      --method-a ./results_baseline/metrics_all \\
      --method-b ./results_new/metrics_all \\
      --name-a "Baseline" --name-b "New Method"
  
  # With specific confidence level and output
  python -m ensemble_metrics.compare \\
      --method-a ./method1/metrics \\
      --method-b ./method2/metrics \\
      --ci-level 0.99 \\
      --output comparison_results.csv
  
  # LaTeX output for papers
  python -m ensemble_metrics.compare \\
      --method-a ./baseline/metrics \\
      --method-b ./proposed/metrics \\
      --format latex

Statistical Tests Performed:
  - Paired bootstrap test for mean difference
  - 95% BCa confidence interval for the difference
  - Cohen's d effect size (paired samples)
  - Two-sided p-value from bootstrap distribution

Interpretation Guide:
  - p < 0.05: Significant difference (*)
  - p < 0.01: Highly significant (**)
  - p < 0.001: Very highly significant (***)
  - Effect size: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, >= 0.8 large
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--method-a",
        type=str,
        required=True,
        help="Directory containing metric CSV files for method A"
    )
    parser.add_argument(
        "--method-b",
        type=str,
        required=True,
        help="Directory containing metric CSV files for method B"
    )
    
    # Method names
    parser.add_argument(
        "--name-a",
        type=str,
        default="Method A",
        help="Display name for method A (default: 'Method A')"
    )
    parser.add_argument(
        "--name-b",
        type=str,
        default="Method B",
        help="Display name for method B (default: 'Method B')"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path for comparison results CSV"
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
    
    # Validate directories
    if not os.path.isdir(args.method_a):
        print(f"Error: Method A directory not found: {args.method_a}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.method_b):
        print(f"Error: Method B directory not found: {args.method_b}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output file
    output_file = args.output
    if output_file is None:
        output_file = "comparison_results.csv"
    
    print(f"Method Comparison")
    print(f"=================")
    print(f"{args.name_a}: {args.method_a}")
    print(f"{args.name_b}: {args.method_b}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Confidence level: {args.ci_level * 100:.0f}%")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()
    
    # Run comparison
    try:
        df = compare_methods(
            metrics_dir_a=args.method_a,
            metrics_dir_b=args.method_b,
            method_a_name=args.name_a,
            method_b_name=args.name_b,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            random_state=args.seed,
            metric_pattern=args.pattern
        )
    except Exception as e:
        print(f"Error during comparison: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(df) == 0:
        print("No metrics could be compared. Check that both directories contain matching metric files.")
        sys.exit(1)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary
    if not args.quiet:
        print()
        print("Comparison Results")
        print("------------------")
        print(format_comparison_table(df, format_style=args.format))
        
        # Summary statistics
        n_significant = df["significant"].sum()
        n_total = len(df)
        print()
        print(f"Summary: {n_significant}/{n_total} metrics show significant differences (p < 0.05)")
        
        # List significant differences
        if n_significant > 0:
            print()
            print("Significant differences:")
            for _, row in df[df["significant"]].iterrows():
                direction = "higher" if row["difference"] > 0 else "lower"
                print(f"  - {row['metric']}: {args.name_b} is {direction} (Δ = {row['difference']:.4f}, p = {row['p_value']:.4f})")
    
    return df


if __name__ == "__main__":
    main()
