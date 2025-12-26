"""ML evaluation metrics for HEP applications.

Includes metrics specific to particle physics like significance improvement
and background rejection at fixed signal efficiency.
"""

from typing import Optional, Tuple

import numpy as np


def roc_curve_with_errors(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_thresholds: int = 100,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve with bootstrap uncertainty bands.

    Parameters
    ----------
    y_true : array
        True labels (0 or 1)
    y_score : array
        Predicted scores/probabilities
    n_thresholds : int
        Number of threshold points
    n_bootstrap : int
        Number of bootstrap samples for error estimation
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (fpr, tpr, tpr_error_low, tpr_error_high)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Thresholds from min to max score
    thresholds = np.linspace(y_score.min(), y_score.max(), n_thresholds)

    # Compute FPR, TPR at each threshold
    fpr = np.zeros(n_thresholds)
    tpr = np.zeros(n_thresholds)

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for i, thresh in enumerate(thresholds):
        pred = (y_score >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))

        tpr[i] = tp / n_pos if n_pos > 0 else 0
        fpr[i] = fp / n_neg if n_neg > 0 else 0

    # Bootstrap for errors
    tpr_samples = np.zeros((n_bootstrap, n_thresholds))

    for b in range(n_bootstrap):
        # Resample
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_b = y_true[idx]
        y_score_b = y_score[idx]

        n_pos_b = np.sum(y_true_b == 1)

        for i, thresh in enumerate(thresholds):
            pred = (y_score_b >= thresh).astype(int)
            tp = np.sum((pred == 1) & (y_true_b == 1))
            tpr_samples[b, i] = tp / n_pos_b if n_pos_b > 0 else 0

    tpr_err_low = tpr - np.percentile(tpr_samples, 16, axis=0)
    tpr_err_high = np.percentile(tpr_samples, 84, axis=0) - tpr

    return fpr, tpr, tpr_err_low, tpr_err_high


def significance_improvement(
    y_true: np.ndarray,
    y_score: np.ndarray,
    signal_efficiency: float = 0.5,
) -> float:
    """Compute significance improvement at given signal efficiency.

    Significance improvement = (S/sqrt(B))_cut / (S/sqrt(B))_no_cut
                            = sqrt(background_rejection) * signal_efficiency

    Parameters
    ----------
    y_true : array
        True labels (1=signal, 0=background)
    y_score : array
        Predicted scores (higher = more signal-like)
    signal_efficiency : float
        Target signal efficiency

    Returns
    -------
    float
        Significance improvement factor
    """
    # Find threshold for target signal efficiency
    signal_scores = y_score[y_true == 1]
    threshold = np.percentile(signal_scores, 100 * (1 - signal_efficiency))

    # Background rejection at this threshold
    bkg_scores = y_score[y_true == 0]
    bkg_rejection = np.mean(bkg_scores < threshold)

    # Significance improvement
    if bkg_rejection > 0:
        return signal_efficiency / np.sqrt(1 - bkg_rejection)
    return np.inf


def background_rejection(
    y_true: np.ndarray,
    y_score: np.ndarray,
    signal_efficiencies: np.ndarray,
) -> np.ndarray:
    """Compute background rejection at given signal efficiencies.

    Parameters
    ----------
    y_true : array
        True labels (1=signal, 0=background)
    y_score : array
        Predicted scores
    signal_efficiencies : array
        Target signal efficiencies

    Returns
    -------
    array
        Background rejection values (1 - FPR)
    """
    signal_scores = y_score[y_true == 1]
    bkg_scores = y_score[y_true == 0]

    rejections = np.zeros(len(signal_efficiencies))

    for i, eff in enumerate(signal_efficiencies):
        threshold = np.percentile(signal_scores, 100 * (1 - eff))
        rejections[i] = np.mean(bkg_scores < threshold)

    return rejections


def auc_with_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute AUC with bootstrap uncertainty.

    Parameters
    ----------
    y_true : array
        True labels
    y_score : array
        Predicted scores
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (auc, auc_error)
    """
    if random_state is not None:
        np.random.seed(random_state)

    def compute_auc(y_t, y_s):
        """Simple AUC computation using Mann-Whitney U statistic."""
        n_pos = np.sum(y_t == 1)
        n_neg = np.sum(y_t == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        pos_scores = y_s[y_t == 1]
        neg_scores = y_s[y_t == 0]

        # Count pairs where positive > negative
        count = 0
        for ps in pos_scores:
            count += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)

        return count / (n_pos * n_neg)

    auc = compute_auc(y_true, y_score)

    # Bootstrap
    auc_samples = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        auc_samples[b] = compute_auc(y_true[idx], y_score[idx])

    auc_error = np.std(auc_samples)

    return auc, auc_error
