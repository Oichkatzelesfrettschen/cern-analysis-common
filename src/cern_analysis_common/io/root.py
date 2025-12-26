"""ROOT file I/O using uproot.

Provides unified interface for reading ROOT files commonly used in HEP.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import uproot
    import awkward as ak

    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False


def _check_uproot():
    """Verify uproot is available."""
    if not HAS_UPROOT:
        raise ImportError(
            "uproot and awkward are required for ROOT I/O. "
            "Install with: pip install uproot awkward"
        )


def load_root(
    filepath: Union[str, Path],
    tree_name: Optional[str] = None,
    branches: Optional[List[str]] = None,
    cut: Optional[str] = None,
    library: str = "ak",
) -> Any:
    """Load data from a ROOT file.

    Parameters
    ----------
    filepath : str or Path
        Path to ROOT file
    tree_name : str, optional
        Name of TTree to load. If None, loads first tree found.
    branches : list of str, optional
        Specific branches to load. If None, loads all.
    cut : str, optional
        Selection cut string (ROOT TTree::Draw syntax)
    library : str
        Output format: "ak" (awkward), "np" (numpy), "pd" (pandas)

    Returns
    -------
    array
        Data in requested format (awkward array by default)

    Examples
    --------
    >>> data = load_root("alice_data.root", "Events", branches=["pt", "eta", "phi"])
    >>> df = load_root("higgs.root", library="pd")
    """
    _check_uproot()

    with uproot.open(filepath) as f:
        if tree_name is None:
            # Find first TTree
            trees = [k for k in f.keys() if "TTree" in str(type(f[k]))]
            if not trees:
                raise ValueError(f"No TTrees found in {filepath}")
            tree_name = trees[0].split(";")[0]

        tree = f[tree_name]

        if branches is None:
            branches = tree.keys()

        return tree.arrays(branches, cut=cut, library=library)


def load_tree(
    filepath: Union[str, Path],
    tree_name: str,
    branches: Optional[List[str]] = None,
) -> "ak.Array":
    """Load a TTree as an awkward array.

    Parameters
    ----------
    filepath : str or Path
        Path to ROOT file
    tree_name : str
        Name of TTree
    branches : list of str, optional
        Branches to load

    Returns
    -------
    ak.Array
        Awkward array with tree data
    """
    _check_uproot()
    return load_root(filepath, tree_name, branches, library="ak")


def tree_to_dataframe(
    filepath: Union[str, Path],
    tree_name: str,
    branches: Optional[List[str]] = None,
    flatten: bool = True,
) -> "pd.DataFrame":
    """Convert TTree to pandas DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to ROOT file
    tree_name : str
        Name of TTree
    branches : list of str, optional
        Branches to load
    flatten : bool
        If True, flatten jagged arrays

    Returns
    -------
    pd.DataFrame
        DataFrame with tree data
    """
    _check_uproot()

    data = load_root(filepath, tree_name, branches, library="ak")

    if flatten:
        # Flatten any jagged arrays
        flat_data = {}
        for field in ak.fields(data):
            arr = data[field]
            if arr.ndim > 1:
                flat_data[field] = ak.flatten(arr)
            else:
                flat_data[field] = arr
        data = ak.Array(flat_data)

    return ak.to_dataframe(data)


def list_trees(filepath: Union[str, Path]) -> List[str]:
    """List all TTrees in a ROOT file.

    Parameters
    ----------
    filepath : str or Path
        Path to ROOT file

    Returns
    -------
    list of str
        Names of TTrees
    """
    _check_uproot()

    with uproot.open(filepath) as f:
        trees = []
        for key in f.keys():
            name = key.split(";")[0]
            obj = f[key]
            if hasattr(obj, "keys"):  # TTree has keys() method
                trees.append(name)
        return list(set(trees))


def list_branches(filepath: Union[str, Path], tree_name: str) -> Dict[str, str]:
    """List branches and their types in a TTree.

    Parameters
    ----------
    filepath : str or Path
        Path to ROOT file
    tree_name : str
        Name of TTree

    Returns
    -------
    dict
        Mapping of branch name to type string
    """
    _check_uproot()

    with uproot.open(filepath) as f:
        tree = f[tree_name]
        return {name: str(branch.typename) for name, branch in tree.items()}
