"""Histogram plotting utilities for HEP analysis."""

from typing import List, Literal, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    data: np.ndarray,
    bins: Union[int, np.ndarray, List[float]] = 50,
    weights: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    ax: Optional["plt.Axes"] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    logy: bool = False,
    **kwargs,
) -> Tuple["plt.Axes", np.ndarray, np.ndarray]:
    """Plot a 1D histogram in HEP style.

    Parameters
    ----------
    data : array
        Data to histogram
    bins : int or array
        Number of bins or bin edges
    weights : array, optional
        Event weights
    label : str, optional
        Legend label
    color : str, optional
        Line/fill color
    histtype : str
        Histogram type ("step", "stepfilled", "bar")
    ax : matplotlib Axes, optional
        Axes to plot on
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    logy : bool
        Use log scale on y-axis
    **kwargs
        Additional arguments to plt.hist

    Returns
    -------
    tuple
        (axes, bin_contents, bin_edges)
    """
    if ax is None:
        fig, ax = plt.subplots()

    bins_arg: Union[int, List[float]] = bins.tolist() if isinstance(bins, np.ndarray) else bins
    counts, edges, _ = ax.hist(
        data,
        bins=bins_arg,
        weights=weights,
        label=label,
        color=color,
        histtype=histtype,
        **kwargs,
    )

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if label:
        ax.legend()

    return ax, cast(np.ndarray, np.asarray(counts)), cast(np.ndarray, edges)


def plot_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    bins: np.ndarray,
    num_weights: Optional[np.ndarray] = None,
    den_weights: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Ratio",
    color: str = "black",
    marker: str = "o",
    reference_line: float = 1.0,
) -> "plt.Axes":
    """Plot ratio of two histograms with errors.

    Parameters
    ----------
    numerator : array
        Numerator data
    denominator : array
        Denominator data
    bins : array
        Bin edges
    num_weights : array, optional
        Numerator weights
    den_weights : array, optional
        Denominator weights
    ax : matplotlib Axes, optional
        Axes to plot on
    xlabel : str, optional
        X-axis label
    ylabel : str
        Y-axis label
    color : str
        Marker color
    marker : str
        Marker style
    reference_line : float
        Y-value for reference line

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Histogram both
    num_counts, _ = np.histogram(numerator, bins=bins, weights=num_weights)
    den_counts, _ = np.histogram(denominator, bins=bins, weights=den_weights)

    # Compute ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den_counts > 0, num_counts / den_counts, np.nan)

    # Errors (assuming Poisson)
    with np.errstate(divide="ignore", invalid="ignore"):
        num_err = np.sqrt(num_counts)
        den_err = np.sqrt(den_counts)
        # Error propagation for ratio
        ratio_err = np.where(
            den_counts > 0,
            ratio * np.sqrt((num_err / num_counts) ** 2 + (den_err / den_counts) ** 2),
            np.nan,
        )

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = 0.5 * np.diff(bins)

    ax.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_err,
        xerr=bin_widths,
        fmt=marker,
        color=color,
        capsize=0,
    )

    ax.axhline(reference_line, color="gray", linestyle="--", linewidth=1)

    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_efficiency(
    passed: np.ndarray,
    total: np.ndarray,
    bins: np.ndarray,
    variable_passed: np.ndarray,
    variable_total: np.ndarray,
    ax: Optional["plt.Axes"] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Efficiency",
    color: str = "blue",
    marker: str = "o",
    confidence: float = 0.68,
) -> "plt.Axes":
    """Plot efficiency with Clopper-Pearson errors.

    Parameters
    ----------
    passed : array
        Boolean mask for events passing selection
    total : array
        Boolean mask for total events (usually all True)
    bins : array
        Bin edges
    variable_passed : array
        Variable values for passed events
    variable_total : array
        Variable values for total events
    ax : matplotlib Axes, optional
        Axes to plot on
    xlabel : str, optional
        X-axis label
    ylabel : str
        Y-axis label
    color : str
        Marker color
    marker : str
        Marker style
    confidence : float
        Confidence level for error bars

    Returns
    -------
    matplotlib Axes
    """
    from cern_analysis_common.physics.efficiency import clopper_pearson_interval

    if ax is None:
        fig, ax = plt.subplots()

    # Count in bins
    n_passed, _ = np.histogram(variable_passed[passed], bins=bins)
    n_total, _ = np.histogram(variable_total[total], bins=bins)

    # Compute efficiency and errors per bin
    eff = np.zeros(len(bins) - 1)
    err_low = np.zeros(len(bins) - 1)
    err_high = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        e, lo, hi = clopper_pearson_interval(int(n_passed[i]), int(n_total[i]), confidence)
        eff[i] = e
        err_low[i] = e - lo
        err_high[i] = hi - e

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_widths = 0.5 * np.diff(bins)

    ax.errorbar(
        bin_centers,
        eff,
        yerr=[err_low, err_high],
        xerr=bin_widths,
        fmt=marker,
        color=color,
        capsize=0,
    )

    ax.set_ylim(0, 1.1)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def plot_2d_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: Union[int, Tuple[int, int], List[np.ndarray]] = 50,
    weights: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    logz: bool = False,
    colorbar: bool = True,
    **kwargs,
) -> "plt.Axes":
    """Plot 2D histogram.

    Parameters
    ----------
    x : array
        X-axis data
    y : array
        Y-axis data
    bins : int, tuple, or list
        Bin specification
    weights : array, optional
        Event weights
    ax : matplotlib Axes, optional
        Axes to plot on
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    cmap : str
        Colormap
    logz : bool
        Log scale for color axis
    colorbar : bool
        Add colorbar
    **kwargs
        Additional arguments to pcolormesh

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    from matplotlib.colors import LogNorm

    norm = LogNorm() if logz else None

    h, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)

    mesh = ax.pcolormesh(
        xedges, yedges, h.T, cmap=cmap, norm=norm, **kwargs
    )

    if colorbar:
        plt.colorbar(mesh, ax=ax)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax


def invariant_mass_plot(
    mass: np.ndarray,
    bins: Union[int, np.ndarray] = 100,
    weights: Optional[np.ndarray] = None,
    ax: Optional["plt.Axes"] = None,
    xlabel: str = r"$m$ [GeV/$c^2$]",
    ylabel: str = "Counts",
    particle_name: Optional[str] = None,
    pdg_mass: Optional[float] = None,
    color: str = "blue",
    **kwargs,
) -> "plt.Axes":
    """Plot invariant mass distribution with optional PDG reference.

    Parameters
    ----------
    mass : array
        Invariant mass values
    bins : int or array
        Bin specification
    weights : array, optional
        Event weights
    ax : matplotlib Axes, optional
        Axes to plot on
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    particle_name : str, optional
        Particle name for legend (e.g., "J/psi", "Z")
    pdg_mass : float, optional
        PDG mass value to show as vertical line
    color : str
        Histogram color
    **kwargs
        Additional arguments to hist

    Returns
    -------
    matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    label = particle_name if particle_name else None

    ax.hist(
        mass,
        bins=bins.tolist() if isinstance(bins, np.ndarray) else bins,
        weights=weights,
        histtype="step",
        color=color,
        label=label,
        linewidth=1.5,
        **kwargs,
    )

    if pdg_mass is not None:
        ax.axvline(pdg_mass, color="red", linestyle="--", linewidth=1,
                   label=f"PDG: {pdg_mass:.3f} GeV")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if label or pdg_mass:
        ax.legend()

    return ax
