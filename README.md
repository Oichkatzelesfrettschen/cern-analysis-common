# cern-analysis-common

Shared infrastructure for CERN/HEP data analysis.

## Overview

`cern-analysis-common` provides Python tools for particle physics data analysis:

- **io**: Data I/O for ROOT, HDF5, and tabular formats (uproot-based)
- **physics**: Four-vectors, kinematic cuts, efficiency calculations
- **plotting**: HEP-style visualization with mplhep support
- **ml**: Machine learning utilities for classification and preprocessing

## Installation

```bash
# Basic installation
pip install cern-analysis-common

# With all optional dependencies
pip install cern-analysis-common[all]

# Development installation
git clone https://github.com/Oichkatzelesfrettschen/cern-analysis-common.git
cd cern-analysis-common
pip install -e ".[dev]"
```

## Quick Start

### Load ROOT file and apply cuts

```python
from cern_analysis_common.io import load_root, tree_to_dataframe
from cern_analysis_common.physics import (
    FourVector,
    invariant_mass,
    apply_pt_cut,
    apply_eta_cut,
    combine_cuts,
)

# Load ALICE data
data = load_root("AO2D.root", "O2mcparticle")

# Apply kinematic cuts
pt_cut = apply_pt_cut(data["pt"], pt_min=0.5)
eta_cut = apply_eta_cut(data["eta"], eta_max=0.9)
selection = combine_cuts(pt_cut, eta_cut)

# Compute invariant mass
p1 = FourVector.from_pt_eta_phi_m(
    data["pt"][selection],
    data["eta"][selection],
    data["phi"][selection],
    PION_MASS
)
```

### Plot with HEP style

```python
from cern_analysis_common.plotting import (
    set_hep_style,
    plot_histogram,
    add_experiment_label,
)

set_hep_style("ALICE")

fig, ax = plt.subplots()
plot_histogram(pt_data, bins=50, xlabel=r"$p_\mathrm{T}$ [GeV/$c$]", ax=ax)
add_experiment_label(ax, "ALICE", "Preliminary", "pp, sqrt(s) = 13.6 TeV")
plt.savefig("pt_spectrum.pdf")
```

### Compute efficiency with proper errors

```python
from cern_analysis_common.physics import clopper_pearson_interval

# 95 events passed out of 100
eff, lower, upper = clopper_pearson_interval(95, 100, confidence=0.68)
print(f"Efficiency: {eff:.1%} (+{upper-eff:.1%} -{eff-lower:.1%})")
# Output: Efficiency: 95.0% (+2.3% -3.1%)
```

## Modules

### io (Data I/O)

| Function | Description |
|----------|-------------|
| `load_root` | Load ROOT file (TTree or branches) |
| `load_tree` | Load TTree as awkward array |
| `tree_to_dataframe` | Convert TTree to pandas DataFrame |
| `load_hdf5` | Load HDF5 datasets |
| `save_hdf5` | Save to HDF5 with compression |
| `load_csv`, `load_excel` | Tabular data loaders |

### physics

| Class/Function | Description |
|----------------|-------------|
| `FourVector` | Lorentz four-vector with (+,-,-,-) metric |
| `invariant_mass` | Two-particle invariant mass |
| `delta_r`, `delta_phi` | Angular separations |
| `apply_pt_cut`, `apply_eta_cut` | Kinematic selections |
| `clopper_pearson_interval` | Exact efficiency confidence interval |
| `acceptance_correction` | Binned acceptance factors |

### plotting

| Function | Description |
|----------|-------------|
| `set_hep_style` | Apply ALICE/ATLAS/CMS style |
| `plot_histogram` | 1D histogram with HEP conventions |
| `plot_ratio` | Ratio plot with errors |
| `plot_efficiency` | Efficiency vs variable |
| `invariant_mass_plot` | Mass distribution with PDG reference |

### ml

| Function | Description |
|----------|-------------|
| `standardize`, `normalize` | Feature scaling |
| `train_test_split_events` | Split preserving event structure |
| `balance_classes` | Handle class imbalance |
| `significance_improvement` | S/sqrt(B) improvement factor |
| `background_rejection` | 1-FPR at fixed signal efficiency |

## Related Projects

This library is part of the [OpenUniverse](https://github.com/Oichkatzelesfrettschen/openuniverse) collection:

- [grb-common](https://github.com/Oichkatzelesfrettschen/grb-common): GRB afterglow analysis
- [compact-common](https://github.com/Oichkatzelesfrettschen/compact-common): Neutron star/black hole physics
- [CERN-Data-Analysis-ALICE-Run3](https://github.com/Oichkatzelesfrettschen/CERN-Data-Analysis-ALICE-Run3): ALICE Run 3 analysis

## License

GPL-3.0-or-later

## Citation

```bibtex
@software{cern_analysis_common,
  author = {Afrauthihinngreygaard, Deirikr Jaiusadastra},
  title = {cern-analysis-common: HEP data analysis tools},
  year = {2025},
  url = {https://github.com/Oichkatzelesfrettschen/cern-analysis-common}
}
```
