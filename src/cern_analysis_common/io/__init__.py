"""Data I/O utilities for HEP analysis.

Supports:
- ROOT files via uproot
- HDF5 files via h5py
- Tabular data (CSV, Excel) via pandas
"""

from cern_analysis_common.io.hdf5 import (
    load_hdf5,
    load_hdf5_dataset,
    save_hdf5,
)
from cern_analysis_common.io.root import (
    list_branches,
    list_trees,
    load_root,
    load_tree,
    tree_to_dataframe,
)
from cern_analysis_common.io.tabular import (
    load_csv,
    load_excel,
    load_parquet,
)

__all__ = [
    "load_root",
    "load_tree",
    "tree_to_dataframe",
    "list_trees",
    "list_branches",
    "load_hdf5",
    "save_hdf5",
    "load_hdf5_dataset",
    "load_csv",
    "load_excel",
    "load_parquet",
]
