"""Selection cuts for HEP analysis.

Provides functions to apply kinematic cuts to particle data.
Designed for use with numpy arrays or pandas DataFrames.
"""

from typing import List, Optional, Tuple, cast

import numpy as np


def apply_pt_cut(
    pt: np.ndarray,
    pt_min: Optional[float] = None,
    pt_max: Optional[float] = None,
) -> np.ndarray:
    """Apply transverse momentum cut.

    Parameters
    ----------
    pt : array
        Transverse momentum values
    pt_min : float, optional
        Minimum pt (inclusive)
    pt_max : float, optional
        Maximum pt (exclusive)

    Returns
    -------
    array
        Boolean mask where True = passes cut
    """
    mask: np.ndarray = np.ones(len(pt), dtype=bool)
    if pt_min is not None:
        mask &= pt >= pt_min
    if pt_max is not None:
        mask &= pt < pt_max
    return mask


def apply_eta_cut(
    eta: np.ndarray,
    eta_min: Optional[float] = None,
    eta_max: Optional[float] = None,
    symmetric: bool = True,
) -> np.ndarray:
    """Apply pseudorapidity cut.

    Parameters
    ----------
    eta : array
        Pseudorapidity values
    eta_min : float, optional
        Minimum eta. If symmetric=True, this is -|eta_max|.
    eta_max : float, optional
        Maximum eta (or symmetric bound if symmetric=True)
    symmetric : bool
        If True, apply |eta| < eta_max

    Returns
    -------
    array
        Boolean mask where True = passes cut

    Examples
    --------
    >>> # |eta| < 0.9 (ALICE central barrel)
    >>> mask = apply_eta_cut(eta, eta_max=0.9)
    >>> # -2.5 < eta < 2.5 (ATLAS/CMS tracker)
    >>> mask = apply_eta_cut(eta, eta_min=-2.5, eta_max=2.5, symmetric=False)
    """
    if symmetric and eta_max is not None:
        return cast(np.ndarray, np.abs(eta) < eta_max)

    mask: np.ndarray = np.ones(len(eta), dtype=bool)
    if eta_min is not None:
        mask &= eta >= eta_min
    if eta_max is not None:
        mask &= eta < eta_max
    return mask


def apply_rapidity_cut(
    y: np.ndarray,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    symmetric: bool = True,
) -> np.ndarray:
    """Apply rapidity cut.

    Parameters
    ----------
    y : array
        Rapidity values
    y_min : float, optional
        Minimum rapidity
    y_max : float, optional
        Maximum rapidity (or symmetric bound)
    symmetric : bool
        If True, apply |y| < y_max

    Returns
    -------
    array
        Boolean mask where True = passes cut
    """
    if symmetric and y_max is not None:
        return cast(np.ndarray, np.abs(y) < y_max)

    mask: np.ndarray = np.ones(len(y), dtype=bool)
    if y_min is not None:
        mask &= y >= y_min
    if y_max is not None:
        mask &= y < y_max
    return mask


def apply_mass_window(
    mass: np.ndarray,
    center: float,
    width: float,
    n_sigma: Optional[float] = None,
) -> np.ndarray:
    """Apply invariant mass window cut.

    Parameters
    ----------
    mass : array
        Invariant mass values
    center : float
        Central mass value (e.g., PDG mass)
    width : float
        Window half-width in mass units, OR mass resolution if n_sigma given
    n_sigma : float, optional
        If given, width is interpreted as resolution and cut is center +/- n_sigma*width

    Returns
    -------
    array
        Boolean mask where True = passes cut

    Examples
    --------
    >>> # J/psi mass window: 3.0 - 3.2 GeV
    >>> mask = apply_mass_window(m_mumu, center=3.097, width=0.1)
    >>> # 3-sigma window around Z mass
    >>> mask = apply_mass_window(m_ll, center=91.2, width=2.5, n_sigma=3)
    """
    if n_sigma is not None:
        half_width = n_sigma * width
    else:
        half_width = width

    return cast(np.ndarray, np.abs(mass - center) < half_width)


def apply_delta_r_cut(
    delta_r: np.ndarray,
    dr_min: Optional[float] = None,
    dr_max: Optional[float] = None,
) -> np.ndarray:
    """Apply angular separation cut.

    Parameters
    ----------
    delta_r : array
        Delta R values
    dr_min : float, optional
        Minimum Delta R (isolation cut)
    dr_max : float, optional
        Maximum Delta R (matching cut)

    Returns
    -------
    array
        Boolean mask where True = passes cut
    """
    mask: np.ndarray = np.ones(len(delta_r), dtype=bool)
    if dr_min is not None:
        mask &= delta_r >= dr_min
    if dr_max is not None:
        mask &= delta_r < dr_max
    return mask


def combine_cuts(*masks: np.ndarray, logic: str = "and") -> np.ndarray:
    """Combine multiple selection masks.

    Parameters
    ----------
    *masks : array
        Boolean masks to combine
    logic : str
        "and" for intersection, "or" for union

    Returns
    -------
    array
        Combined boolean mask

    Examples
    --------
    >>> pt_cut = apply_pt_cut(pt, pt_min=1.0)
    >>> eta_cut = apply_eta_cut(eta, eta_max=0.9)
    >>> final_mask = combine_cuts(pt_cut, eta_cut)
    """
    if not masks:
        raise ValueError("At least one mask required")

    result: np.ndarray = masks[0].copy()

    if logic == "and":
        for mask in masks[1:]:
            result &= mask
    elif logic == "or":
        for mask in masks[1:]:
            result |= mask
    else:
        raise ValueError(f"Unknown logic: {logic}. Use 'and' or 'or'.")

    return result


def cut_efficiency(mask: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Compute efficiency of a selection cut.

    Parameters
    ----------
    mask : array
        Boolean selection mask
    weights : array, optional
        Event weights

    Returns
    -------
    float
        Cut efficiency (fraction passing)
    """
    if weights is not None:
        return float(np.sum(weights[mask]) / np.sum(weights))
    return float(np.mean(mask))


def cut_flow(
    masks: List[Tuple[str, np.ndarray]],
    weights: Optional[np.ndarray] = None,
) -> List[Tuple[str, int, float]]:
    """Generate cut flow table.

    Parameters
    ----------
    masks : list of (name, mask) tuples
        Named selection masks in order of application
    weights : array, optional
        Event weights

    Returns
    -------
    list of (name, n_pass, efficiency) tuples
        Cut flow entries

    Examples
    --------
    >>> cuts = [
    ...     ("pt > 1", pt_mask),
    ...     ("|eta| < 0.9", eta_mask),
    ...     ("mass window", mass_mask),
    ... ]
    >>> flow = cut_flow(cuts)
    >>> for name, n, eff in flow:
    ...     print(f"{name}: {n} ({eff:.1%})")
    """
    n_total = len(masks[0][1])
    cumulative: np.ndarray = np.ones(n_total, dtype=bool)

    flow = [("Initial", n_total, 1.0)]

    for name, mask in masks:
        cumulative &= mask
        n_pass = int(np.sum(cumulative))
        eff = float(n_pass / n_total)
        flow.append((name, n_pass, eff))

    return flow
