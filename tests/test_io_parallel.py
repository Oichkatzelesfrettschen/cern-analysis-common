import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("cern-analysis-common/src"))
from cern_analysis_common.io.root import load_dataset

class TestParallelIO(unittest.TestCase):
    
    @patch('cern_analysis_common.io.root.load_root')
    @patch('cern_analysis_common.io.root._check_uproot')
    @patch('glob.glob')
    def test_load_dataset_numpy(self, mock_glob, mock_check, mock_load):
        # Setup mocks
        mock_glob.return_value = ['file1.root', 'file2.root']
        
        # Mock return data for each file
        data1 = {'pt': np.array([1, 2]), 'eta': np.array([0.1, 0.2])}
        data2 = {'pt': np.array([3, 4]), 'eta': np.array([0.3, 0.4])}
        
        # Configure side effect for load_root
        mock_load.side_effect = [data1, data2]
        
        # Run function
        result = load_dataset("*.root", library="np", max_workers=2)
        
        # Verify calls
        self.assertEqual(mock_load.call_count, 2)
        
        # Verify concatenation
        np.testing.assert_array_equal(result['pt'], np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(result['eta'], np.array([0.1, 0.2, 0.3, 0.4]))

    @patch('cern_analysis_common.io.root.load_root')
    @patch('cern_analysis_common.io.root._check_uproot')
    @patch('glob.glob')
    def test_load_dataset_pandas(self, mock_glob, mock_check, mock_load):
        # Setup mocks
        mock_glob.return_value = ['file1.root', 'file2.root']
        
        df1 = pd.DataFrame({'pt': [1, 2]})
        df2 = pd.DataFrame({'pt': [3, 4]})
        
        mock_load.side_effect = [df1, df2]
        
        result = load_dataset("*.root", library="pd", max_workers=2)
        
        self.assertEqual(len(result), 4)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), 
                                      pd.DataFrame({'pt': [1, 2, 3, 4]}))

if __name__ == '__main__':
    unittest.main()
