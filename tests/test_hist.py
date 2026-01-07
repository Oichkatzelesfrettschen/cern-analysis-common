import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath("cern-analysis-common/src"))
from cern_analysis_common.hist import Histogram1D


class TestHistogram(unittest.TestCase):

    def test_fill_and_merge(self):
        h1 = Histogram1D(bins=[0, 1, 2, 3])
        h1.fill([0.5, 1.5, 2.5])

        expected = np.array([1., 1., 1.])
        np.testing.assert_array_equal(h1.counts, expected)

        h2 = Histogram1D(bins=[0, 1, 2, 3])
        h2.fill([0.5, 0.5])

        h1.merge(h2)

        expected_merged = np.array([3., 1., 1.])
        np.testing.assert_array_equal(h1.counts, expected_merged)

    def test_weighted(self):
        h = Histogram1D(bins=10, range=(0, 10))
        data = np.array([5.0, 5.0])
        weights = np.array([0.5, 0.5])

        h.fill(data, weights)

        # Bin index for 5.0 in [0,10] with 10 bins is index 5 (5.0-6.0)
        self.assertEqual(h.counts[5], 1.0)
        self.assertEqual(h.variances[5], 0.5**2 + 0.5**2) # 0.25 + 0.25 = 0.5

if __name__ == '__main__':
    unittest.main()
