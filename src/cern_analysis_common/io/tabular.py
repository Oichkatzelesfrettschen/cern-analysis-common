"""Tabular data I/O (CSV, Excel, Parquet).

Common formats for preprocessed or summarized HEP data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def load_csv(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    nrows: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load CSV file as DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    columns : list of str, optional
        Specific columns to load
    dtype : dict, optional
        Column dtype specifications
    nrows : int, optional
        Number of rows to read
    **kwargs
        Additional arguments to pandas.read_csv

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    df = pd.read_csv(filepath, usecols=columns, dtype=dtype, nrows=nrows, **kwargs)
    return df


def load_excel(
    filepath: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load Excel file as DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to Excel file
    sheet_name : str or int
        Sheet to load (name or index)
    columns : list of str, optional
        Specific columns to load
    **kwargs
        Additional arguments to pandas.read_excel

    Returns
    -------
    pd.DataFrame
        Loaded data

    Notes
    -----
    Requires openpyxl for .xlsx files.
    """
    df = pd.read_excel(
        filepath, sheet_name=sheet_name, usecols=columns, **kwargs
    )
    return df


def load_parquet(
    filepath: Union[str, Path],
    columns: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load Parquet file as DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to Parquet file
    columns : list of str, optional
        Specific columns to load
    **kwargs
        Additional arguments to pandas.read_parquet

    Returns
    -------
    pd.DataFrame
        Loaded data

    Notes
    -----
    Parquet is efficient for columnar data and preserves types well.
    """
    df = pd.read_parquet(filepath, columns=columns, **kwargs)
    return df


def save_parquet(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    compression: str = "snappy",
    **kwargs,
) -> None:
    """Save DataFrame to Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save
    filepath : str or Path
        Output path
    compression : str
        Compression algorithm
    **kwargs
        Additional arguments to DataFrame.to_parquet
    """
    df.to_parquet(filepath, compression=compression, **kwargs)
