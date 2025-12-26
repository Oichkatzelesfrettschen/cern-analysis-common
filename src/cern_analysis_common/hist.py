import numpy as np
from typing import Union, Optional, Tuple

class Histogram1D:
    """
    High-performance, incremental 1D histogramming.
    Designed for streaming data processing and parallel reduction.
    """
    def __init__(self, bins: Union[int, np.ndarray, list], range: Optional[Tuple[float, float]] = None):
        """
        Initialize histogram.
        
        Parameters
        ----------
        bins : int or sequence
            If int, defines number of equal-width bins in given range.
            If sequence, defines bin edges.
        range : (float, float), optional
            The lower and upper range of bins. Required if bins is int.
        """
        if isinstance(bins, int):
            if range is None:
                raise ValueError("range must be provided when bins is an integer")
            self.edges = np.linspace(range[0], range[1], bins + 1)
        else:
            self.edges = np.asarray(bins)
            if not np.all(np.diff(self.edges) > 0):
                raise ValueError("Bin edges must be monotonically increasing")
        
        self.counts = np.zeros(len(self.edges) - 1, dtype=np.float64)
        self.sum_w2 = np.zeros(len(self.edges) - 1, dtype=np.float64) # Sum of weights squared
        self.underflow = 0.0
        self.overflow = 0.0
        self.n_entries = 0
        
    def fill(self, data: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Fill histogram with data.
        
        Parameters
        ----------
        data : array-like
            Values to histogram
        weights : array-like, optional
            Weights for each value
        """
        data = np.asarray(data)
        if weights is None:
            weights = np.ones_like(data, dtype=np.float64)
        else:
            weights = np.asarray(weights)
            
        # Numpy histogram is fast enough for chunks
        # But we need to handle under/overflow manually if we want ROOT compatibility
        # np.histogram ignores outliers
        
        c, _ = np.histogram(data, bins=self.edges, weights=weights)
        self.counts += c
        
        # Track errors (sum of weights squared)
        # np.histogram doesn't do this, so we need to trick it or loop?
        # Trick: histogram(data, weights=weights**2)
        sw2, _ = np.histogram(data, bins=self.edges, weights=weights**2)
        self.sum_w2 += sw2
        
        self.n_entries += len(data)
        
        # Simple overflow/underflow (unweighted for now, or full check)
        # Optimization: Don't do full mask if not needed
        # For strict correctness:
        self.underflow += np.sum(weights[data < self.edges[0]])
        self.overflow += np.sum(weights[data >= self.edges[-1]])
        
    def merge(self, other: 'Histogram1D'):
        """Add another histogram to this one."""
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("Cannot merge histograms with different binning")
        
        self.counts += other.counts
        self.sum_w2 += other.sum_w2
        self.underflow += other.underflow
        self.overflow += other.overflow
        self.n_entries += other.n_entries
        
    @property
    def variances(self):
        """Return variance per bin (sumw2)."""
        return self.sum_w2
        
    @property
    def errors(self):
        """Return standard error per bin."""
        return np.sqrt(self.sum_w2)
        
    def plot(self, ax=None, **kwargs):
        """Plot using matplotlib."""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
            
        return ax.stairs(self.counts, self.edges, **kwargs)

    def __repr__(self):
        return f"Histogram1D(bins={len(self.counts)}, entries={self.n_entries})"
