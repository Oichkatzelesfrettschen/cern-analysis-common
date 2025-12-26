"""Physics utilities for HEP analysis.

Includes:
- Four-vector operations and Lorentz transformations
- Kinematic variables (pt, eta, phi, rapidity)
- Selection cuts and event filtering
- Efficiency and acceptance calculations
"""

from cern_analysis_common.physics.four_vectors import (
    FourVector,
    invariant_mass,
    transverse_momentum,
    pseudorapidity,
    rapidity,
    delta_r,
    delta_phi,
    boost_to_cm,
)
from cern_analysis_common.physics.cuts import (
    apply_pt_cut,
    apply_eta_cut,
    apply_rapidity_cut,
    apply_mass_window,
    combine_cuts,
)
from cern_analysis_common.physics.efficiency import (
    efficiency_ratio,
    binomial_efficiency,
    clopper_pearson_interval,
    acceptance_correction,
)

__all__ = [
    # Four vectors
    "FourVector",
    "invariant_mass",
    "transverse_momentum",
    "pseudorapidity",
    "rapidity",
    "delta_r",
    "delta_phi",
    "boost_to_cm",
    # Cuts
    "apply_pt_cut",
    "apply_eta_cut",
    "apply_rapidity_cut",
    "apply_mass_window",
    "combine_cuts",
    # Efficiency
    "efficiency_ratio",
    "binomial_efficiency",
    "clopper_pearson_interval",
    "acceptance_correction",
]
