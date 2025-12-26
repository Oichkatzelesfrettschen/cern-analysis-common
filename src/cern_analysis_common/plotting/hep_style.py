"""HEP plotting styles.

Provides matplotlib style configurations matching major experiments.
Uses mplhep when available, falls back to built-in styles.
"""

from typing import Optional

import matplotlib.pyplot as plt

try:
    import mplhep as hep

    HAS_MPLHEP = True
except ImportError:
    HAS_MPLHEP = False


# Default HEP style parameters (when mplhep not available)
HEP_STYLE_PARAMS = {
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.minor.width": 1.0,
    "ytick.minor.width": 1.0,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.formatter.use_mathtext": True,
    "legend.frameon": False,
    "errorbar.capsize": 0,
}


def set_hep_style(experiment: Optional[str] = None) -> None:
    """Set HEP-compatible plotting style.

    Parameters
    ----------
    experiment : str, optional
        Experiment style: "ALICE", "ATLAS", "CMS", "LHCb", or None for generic

    Examples
    --------
    >>> set_hep_style("ALICE")
    >>> plt.hist(data, bins=50)
    >>> plt.savefig("plot.pdf")
    """
    if HAS_MPLHEP:
        if experiment is None:
            hep.style.use("ROOT")
        elif experiment.upper() == "ALICE":
            hep.style.use("ALICE")
        elif experiment.upper() == "ATLAS":
            hep.style.use("ATLAS")
        elif experiment.upper() == "CMS":
            hep.style.use("CMS")
        elif experiment.upper() == "LHCB":
            hep.style.use("LHCb")
        else:
            hep.style.use("ROOT")
    else:
        plt.rcParams.update(HEP_STYLE_PARAMS)


def alice_style() -> None:
    """Set ALICE experiment plotting style."""
    set_hep_style("ALICE")


def atlas_style() -> None:
    """Set ATLAS experiment plotting style."""
    set_hep_style("ATLAS")


def cms_style() -> None:
    """Set CMS experiment plotting style."""
    set_hep_style("CMS")


def add_experiment_label(
    ax: Optional["plt.Axes"] = None,
    experiment: str = "ALICE",
    status: str = "Preliminary",
    energy: Optional[str] = None,
    position: str = "upper left",
    fontsize: int = 14,
) -> None:
    """Add experiment label to plot.

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Axes to add label to (uses current axes if None)
    experiment : str
        Experiment name
    status : str
        Publication status ("Preliminary", "Work in Progress", "")
    energy : str, optional
        Collision energy (e.g., "pp, sqrt(s) = 13.6 TeV")
    position : str
        Label position ("upper left", "upper right", etc.)
    fontsize : int
        Font size for label

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.hist(pt, bins=50)
    >>> add_experiment_label(ax, "ALICE", "Preliminary", "Pb-Pb, sqrt(s_NN) = 5.36 TeV")
    """
    if ax is None:
        ax = plt.gca()

    # Position mapping
    positions = {
        "upper left": (0.05, 0.95),
        "upper right": (0.95, 0.95),
        "lower left": (0.05, 0.05),
        "lower right": (0.95, 0.05),
    }
    x, y = positions.get(position, (0.05, 0.95))
    ha = "left" if "left" in position else "right"
    va = "top" if "upper" in position else "bottom"

    if HAS_MPLHEP:
        if experiment.upper() == "ALICE":
            hep.alice.label(ax=ax, data=status != "", preliminary=(status == "Preliminary"))
        elif experiment.upper() == "ATLAS":
            hep.atlas.label(ax=ax, data=status != "")
        elif experiment.upper() == "CMS":
            hep.cms.label(ax=ax, data=status != "")
        else:
            # Generic label
            label = f"{experiment}"
            if status:
                label += f" {status}"
            ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                    fontweight="bold", ha=ha, va=va)
    else:
        # Fallback without mplhep
        label = f"{experiment}"
        if status:
            label += f" {status}"
        ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold", ha=ha, va=va)

    # Add energy label below
    if energy:
        y_offset = -0.05 if va == "top" else 0.05
        ax.text(x, y + y_offset, energy, transform=ax.transAxes,
                fontsize=fontsize - 2, ha=ha, va=va)
