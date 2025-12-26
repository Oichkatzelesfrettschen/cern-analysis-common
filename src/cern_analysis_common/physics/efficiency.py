"""Efficiency and acceptance calculations for HEP analysis.

Implements proper statistical treatment of efficiencies including
binomial errors and Clopper-Pearson confidence intervals.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats


def efficiency_ratio(
    numerator: Union[int, np.ndarray],
    denominator: Union[int, np.ndarray],
    weights_num: Optional[np.ndarray] = None,
    weights_den: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute efficiency with binomial uncertainty.

    Parameters
    ----------
    numerator : int or array
        Number of events passing (or boolean mask)
    denominator : int or array
        Total number of events (or full sample)
    weights_num : array, optional
        Weights for numerator events
    weights_den : array, optional
        Weights for denominator events

    Returns
    -------
    tuple
        (efficiency, uncertainty)

    Notes
    -----
    For weighted events, uses Poisson-like error propagation.
    """
    if isinstance(numerator, np.ndarray):
        if weights_num is not None:
            k = np.sum(weights_num[numerator])
        else:
            k = np.sum(numerator)
    else:
        k = numerator

    if isinstance(denominator, np.ndarray):
        if weights_den is not None:
            n = np.sum(weights_den)
        else:
            n = len(denominator)
    else:
        n = denominator

    if n == 0:
        return 0.0, 0.0

    eff = k / n

    # Binomial error: sqrt(eff * (1-eff) / n)
    # Handles edge cases where eff = 0 or 1
    var = eff * (1 - eff) / n
    err = np.sqrt(max(var, 0))

    return float(eff), float(err)


def binomial_efficiency(
    k: int,
    n: int,
) -> Tuple[float, float, float]:
    """Compute efficiency with asymmetric binomial errors.

    Uses Wilson score interval for better coverage at extremes.

    Parameters
    ----------
    k : int
        Number of successes
    n : int
        Number of trials

    Returns
    -------
    tuple
        (efficiency, error_low, error_high)
    """
    if n == 0:
        return 0.0, 0.0, 0.0

    eff = k / n

    # Wilson score interval (68% CL for 1-sigma equivalent)
    z = 1.0  # 1-sigma
    z2 = z * z

    denom = 1 + z2 / n
    center = (eff + z2 / (2 * n)) / denom
    delta = z * np.sqrt(eff * (1 - eff) / n + z2 / (4 * n**2)) / denom

    err_low = eff - (center - delta)
    err_high = (center + delta) - eff

    return float(eff), float(err_low), float(err_high)


def clopper_pearson_interval(
    k: int,
    n: int,
    confidence: float = 0.68,
) -> Tuple[float, float, float]:
    """Compute Clopper-Pearson exact confidence interval.

    This is the standard method for efficiency uncertainties in HEP.

    Parameters
    ----------
    k : int
        Number of successes
    n : int
        Number of trials
    confidence : float
        Confidence level (default 0.68 for 1-sigma equivalent)

    Returns
    -------
    tuple
        (efficiency, lower_bound, upper_bound)

    Examples
    --------
    >>> eff, lo, hi = clopper_pearson_interval(95, 100)
    >>> print(f"Efficiency: {eff:.1%} (+{hi-eff:.1%} -{eff-lo:.1%})")
    """
    if n == 0:
        return 0.0, 0.0, 1.0

    eff = k / n
    alpha = 1 - confidence

    # Lower bound: Beta distribution quantile
    if k == 0:
        lower = 0.0
    else:
        lower = stats.beta.ppf(alpha / 2, k, n - k + 1)

    # Upper bound: Beta distribution quantile
    if k == n:
        upper = 1.0
    else:
        upper = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)

    return float(eff), float(lower), float(upper)


def acceptance_correction(
    reconstructed: np.ndarray,
    generated: np.ndarray,
    bins: np.ndarray,
    variable_reco: np.ndarray,
    variable_gen: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute acceptance correction factors per bin.

    Parameters
    ----------
    reconstructed : array
        Boolean mask for reconstructed events
    generated : array
        Boolean mask for generated events (MC truth)
    bins : array
        Bin edges for the variable
    variable_reco : array
        Variable values at reconstruction level
    variable_gen : array
        Variable values at generator level

    Returns
    -------
    tuple
        (bin_centers, correction_factors, uncertainties)

    Notes
    -----
    Correction = N_gen / N_reco per bin.
    Apply by multiplying data yields by correction factors.
    """
    # Count in bins
    n_reco, _ = np.histogram(variable_reco[reconstructed], bins=bins)
    n_gen, _ = np.histogram(variable_gen[generated], bins=bins)

    # Correction factors
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(n_reco > 0, n_gen / n_reco, 1.0)

    # Uncertainty (propagated from binomial)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(n_gen > 0, n_reco / n_gen, 0.0)
        eff_err = np.sqrt(eff * (1 - eff) / np.maximum(n_gen, 1))
        # Error on correction = correction * (eff_err / eff)
        corr_err = np.where(eff > 0, correction * eff_err / eff, 0.0)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return bin_centers, correction, corr_err


def unfolding_response_matrix(
    reco_values: np.ndarray,
    gen_values: np.ndarray,
    reco_bins: np.ndarray,
    gen_bins: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build detector response matrix for unfolding.

    Parameters
    ----------
    reco_values : array
        Reconstructed variable values
    gen_values : array
        Generator-level (true) values
    reco_bins : array
        Bin edges for reconstructed axis
    gen_bins : array
        Bin edges for generator axis
    weights : array, optional
        Event weights

    Returns
    -------
    array
        Response matrix R[i,j] = P(reco bin i | gen bin j)
        Shape: (n_reco_bins, n_gen_bins)

    Notes
    -----
    Columns are normalized so each column sums to 1 (probability).
    """
    # 2D histogram
    response, _, _ = np.histogram2d(
        reco_values, gen_values, bins=[reco_bins, gen_bins], weights=weights
    )

    # Normalize columns (probability of reco bin given gen bin)
    col_sums = response.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        response_norm = np.where(col_sums > 0, response / col_sums, 0.0)

    return response_norm
