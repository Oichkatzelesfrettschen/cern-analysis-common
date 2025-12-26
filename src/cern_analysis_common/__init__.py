"""
cern-analysis-common: Shared infrastructure for CERN/HEP data analysis.

This library provides common tools for particle physics data analysis:

- io: Data I/O for ROOT, HDF5, and tabular formats
- physics: Four-vectors, kinematic cuts, efficiency calculations
- plotting: HEP-style visualization with mplhep
- ml: Machine learning utilities (VAE, classifiers)
"""

from cern_analysis_common.constants import (
    PROTON_MASS,
    PION_MASS,
    KAON_MASS,
    ELECTRON_MASS,
    MUON_MASS,
    C_LIGHT,
    GEV_TO_MEV,
    MEV_TO_GEV,
)

__version__ = "0.1.0"
__all__ = [
    "PROTON_MASS",
    "PION_MASS",
    "KAON_MASS",
    "ELECTRON_MASS",
    "MUON_MASS",
    "C_LIGHT",
    "GEV_TO_MEV",
    "MEV_TO_GEV",
]
