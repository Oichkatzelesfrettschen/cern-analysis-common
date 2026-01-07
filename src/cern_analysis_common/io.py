
import pandas as pd
import uproot


def load_root(file_path, tree_name=None):
    """
    Load a ROOT file into a pandas DataFrame or awkward array.
    """
    with uproot.open(file_path) as file:
        if tree_name is None:
            # Try to guess tree name or return file keys
            keys = file.keys()
            if not keys:
                return None
            tree_name = keys[0] # Naive guess

        tree = file[tree_name]
        return tree.arrays(library="pd")

def load_csv(file_path, **kwargs):
    """
    Load a CSV file using pandas.
    """
    return pd.read_csv(file_path, **kwargs)
