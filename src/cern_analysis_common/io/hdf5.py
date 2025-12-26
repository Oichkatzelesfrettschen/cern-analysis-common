"""HDF5 file I/O for large datasets.

HDF5 is commonly used for preprocessed HEP data and ML training sets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _check_h5py():
    """Verify h5py is available."""
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required for HDF5 I/O. "
            "Install with: pip install h5py"
        )


def load_hdf5(
    filepath: Union[str, Path],
    datasets: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load datasets from an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    datasets : list of str, optional
        Specific datasets to load. If None, loads all.

    Returns
    -------
    dict
        Mapping of dataset names to numpy arrays

    Examples
    --------
    >>> data = load_hdf5("jets.h5", datasets=["jet_pt", "jet_eta", "jet_constituents"])
    """
    _check_h5py()

    result = {}
    with h5py.File(filepath, "r") as f:
        if datasets is None:
            datasets = list(f.keys())

        for name in datasets:
            if name in f:
                result[name] = f[name][:]
            else:
                raise KeyError(f"Dataset '{name}' not found in {filepath}")

    return result


def load_hdf5_dataset(
    filepath: Union[str, Path],
    dataset: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
) -> np.ndarray:
    """Load a single dataset with optional slicing.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    dataset : str
        Name of dataset
    start : int, optional
        Start index for slicing
    stop : int, optional
        Stop index for slicing

    Returns
    -------
    np.ndarray
        Dataset array
    """
    _check_h5py()

    with h5py.File(filepath, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset '{dataset}' not found in {filepath}")

        ds = f[dataset]
        if start is not None or stop is not None:
            return ds[start:stop]
        return ds[:]


def save_hdf5(
    filepath: Union[str, Path],
    data: Dict[str, np.ndarray],
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
    overwrite: bool = False,
) -> None:
    """Save datasets to an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file
    data : dict
        Mapping of dataset names to numpy arrays
    compression : str, optional
        Compression algorithm ("gzip", "lzf", None)
    compression_opts : int
        Compression level (1-9 for gzip)
    overwrite : bool
        If True, overwrite existing file

    Examples
    --------
    >>> save_hdf5("output.h5", {"jet_pt": pt_array, "jet_mass": mass_array})
    """
    _check_h5py()

    filepath = Path(filepath)
    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} already exists. Use overwrite=True.")

    mode = "w" if overwrite else "x"

    with h5py.File(filepath, mode) as f:
        for name, array in data.items():
            if compression:
                f.create_dataset(
                    name,
                    data=array,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                f.create_dataset(name, data=array)


def list_hdf5_datasets(filepath: Union[str, Path]) -> Dict[str, tuple]:
    """List datasets and their shapes in an HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file

    Returns
    -------
    dict
        Mapping of dataset names to (shape, dtype) tuples
    """
    _check_h5py()

    result = {}
    with h5py.File(filepath, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                result[name] = (obj.shape, obj.dtype)

        f.visititems(visitor)

    return result
