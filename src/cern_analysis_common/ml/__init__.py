"""Machine learning utilities for HEP analysis.

Includes:
- Data preprocessing for ML
- Common architectures (VAE, classifiers)
- Evaluation metrics
"""

from cern_analysis_common.ml.metrics import (
    background_rejection,
    roc_curve_with_errors,
    significance_improvement,
)
from cern_analysis_common.ml.preprocessing import (
    balance_classes,
    normalize,
    standardize,
    train_test_split_events,
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
