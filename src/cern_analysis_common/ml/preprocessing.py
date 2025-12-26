"""Data preprocessing for ML in HEP.

Provides utilities for preparing particle physics data for machine learning.
"""

from typing import List, Optional, Tuple

import numpy as np


def standardize(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance.

    Parameters
    ----------
    X : array
        Input features (n_samples, n_features)
    mean : array, optional
        Pre-computed mean (for test data)
    std : array, optional
        Pre-computed std (for test data)
    axis : int
        Axis along which to compute statistics

    Returns
    -------
    tuple
        (standardized_X, mean, std)

    Examples
    --------
    >>> X_train_std, mean, std = standardize(X_train)
    >>> X_test_std, _, _ = standardize(X_test, mean=mean, std=std)
    """
    if mean is None:
        mean = np.mean(X, axis=axis)
    if std is None:
        std = np.std(X, axis=axis)

    # Avoid division by zero
    std = np.where(std > 0, std, 1.0)

    X_std = (X - mean) / std
    return X_std, mean, std


def normalize(
    X: np.ndarray,
    min_val: Optional[np.ndarray] = None,
    max_val: Optional[np.ndarray] = None,
    feature_range: Tuple[float, float] = (0, 1),
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to a given range.

    Parameters
    ----------
    X : array
        Input features
    min_val : array, optional
        Pre-computed minimum
    max_val : array, optional
        Pre-computed maximum
    feature_range : tuple
        Target range (min, max)
    axis : int
        Axis along which to compute statistics

    Returns
    -------
    tuple
        (normalized_X, min_val, max_val)
    """
    if min_val is None:
        min_val = np.min(X, axis=axis)
    if max_val is None:
        max_val = np.max(X, axis=axis)

    scale = max_val - min_val
    scale = np.where(scale > 0, scale, 1.0)

    X_norm = (X - min_val) / scale
    X_norm = X_norm * (feature_range[1] - feature_range[0]) + feature_range[0]

    return X_norm, min_val, max_val


def train_test_split_events(
    *arrays: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> List[np.ndarray]:
    """Split arrays into train and test sets.

    Unlike sklearn's train_test_split, this ensures consistent splitting
    for multiple arrays (important for matched MC truth).

    Parameters
    ----------
    *arrays : array
        Arrays to split (must have same length)
    test_size : float
        Fraction of data for test set
    random_state : int, optional
        Random seed for reproducibility
    shuffle : bool
        Whether to shuffle before splitting

    Returns
    -------
    list
        [train_1, test_1, train_2, test_2, ...]

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split_events(X, y, test_size=0.2)
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required")

    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have the same length")

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    result = []
    for arr in arrays:
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "undersample",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Balance class distribution for classification.

    Parameters
    ----------
    X : array
        Features (n_samples, n_features)
    y : array
        Labels (n_samples,)
    strategy : str
        "undersample" to reduce majority class,
        "oversample" to duplicate minority class
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (balanced_X, balanced_y)
    """
    if random_state is not None:
        np.random.seed(random_state)

    classes, counts = np.unique(y, return_counts=True)

    if strategy == "undersample":
        target_count = np.min(counts)
    elif strategy == "oversample":
        target_count = np.max(counts)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]

        if len(cls_indices) < target_count:
            # Oversample: sample with replacement
            sampled = np.random.choice(cls_indices, size=target_count, replace=True)
        elif len(cls_indices) > target_count:
            # Undersample: sample without replacement
            sampled = np.random.choice(cls_indices, size=target_count, replace=False)
        else:
            sampled = cls_indices

        balanced_indices.extend(sampled)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]


def create_event_mask(
    features: np.ndarray,
    nan_policy: str = "drop",
    inf_policy: str = "drop",
) -> np.ndarray:
    """Create mask for valid events (no NaN/Inf).

    Parameters
    ----------
    features : array
        Feature array (n_samples, n_features)
    nan_policy : str
        "drop" to mask NaN events, "zero" to replace with 0
    inf_policy : str
        "drop" to mask Inf events, "clip" to clip to finite

    Returns
    -------
    array
        Boolean mask of valid events
    """
    if nan_policy == "drop":
        valid_nan = ~np.any(np.isnan(features), axis=1)
    else:
        valid_nan = np.ones(len(features), dtype=bool)

    if inf_policy == "drop":
        valid_inf = ~np.any(np.isinf(features), axis=1)
    else:
        valid_inf = np.ones(len(features), dtype=bool)

    return valid_nan & valid_inf
