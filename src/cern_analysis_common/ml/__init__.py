"""Machine learning utilities for HEP analysis.

Includes:
- Data preprocessing for ML
- Common architectures (VAE, classifiers)
- Evaluation metrics
"""

from cern_analysis_common.ml.preprocessing import (
    standardize,
    normalize,
    train_test_split_events,
    balance_classes,
)
from cern_analysis_common.ml.metrics import (
    roc_curve_with_errors,
    significance_improvement,
    background_rejection,
)

__all__ = [
    "standardize",
    "normalize",
    "train_test_split_events",
    "balance_classes",
    "roc_curve_with_errors",
    "significance_improvement",
    "background_rejection",
]
