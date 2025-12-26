# cern-analysis-common Documentation

Shared infrastructure for CERN/HEP data analysis.

## Modules

- `cern_analysis_common.plotting`: HEP experiment plotting styles.
- `cern_analysis_common.io`: ROOT and data loading helpers.
- `cern_analysis_common.physics`: High energy physics utilities.

## Quick Start

```python
from cern_analysis_common.plotting import set_hep_style
import matplotlib.pyplot as plt

set_hep_style("ALICE")
plt.plot([1, 2, 3], [1, 4, 9])
plt.show()
```
