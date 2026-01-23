#!/usr/bin/env python3
"""Bridge module for importing metrics from the values module."""

import sys
from pathlib import Path
from typing import Optional, Dict, Any

_values_metrics_available = False
_values_metrics = {}

try:
    values_path = Path(__file__).parent.parent.parent / "values"
    if values_path.exists() and str(values_path) not in sys.path:
        sys.path.insert(0, str(values_path))
    try:
        from evaluation.metrics import ncc
        _values_metrics["ncc"] = ncc.compute_ncc
        _values_metrics_available = True
    except ImportError:
        pass
    
    try:
        from evaluation.metrics import auroc
        _values_metrics["auroc"] = auroc
        _values_metrics_available = True
    except ImportError:
        pass
    
    try:
        from evaluation.metrics import aurc
        _values_metrics["aurc"] = aurc
        _values_metrics_available = True
    except ImportError:
        pass
    
    try:
        from evaluation.metrics import ace
        _values_metrics["ace"] = ace
        _values_metrics_available = True
    except ImportError:
        pass
    
    try:
        from evaluation.metrics import al_improvement
        _values_metrics["al_improvement"] = al_improvement
        _values_metrics_available = True
    except ImportError:
        pass

except Exception:
    _values_metrics_available = False


def is_values_available() -> bool:
    """Check if values module metrics are available."""
    return _values_metrics_available


def get_values_metric(metric_name: str) -> Optional[Any]:
    """Get a metric function from the values module if available."""
    return _values_metrics.get(metric_name)


def list_available_values_metrics() -> list:
    """List all available metrics from the values module."""
    return list(_values_metrics.keys())
