"""HEP-style plotting utilities.

Provides functions for creating publication-quality plots following
HEP conventions (ALICE, ATLAS, CMS styles).
"""

from cern_analysis_common.plotting.hep_style import (
    set_hep_style,
    alice_style,
    atlas_style,
    cms_style,
    add_experiment_label,
)
from cern_analysis_common.plotting.histograms import (
    plot_histogram,
    plot_ratio,
    plot_efficiency,
    plot_2d_histogram,
    invariant_mass_plot,
)

__all__ = [
    "set_hep_style",
    "alice_style",
    "atlas_style",
    "cms_style",
    "add_experiment_label",
    "plot_histogram",
    "plot_ratio",
    "plot_efficiency",
    "plot_2d_histogram",
    "invariant_mass_plot",
]
