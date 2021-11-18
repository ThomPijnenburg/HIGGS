# HIGGS
Finding HIGGS in a Haystack


## Data

The HIGGS dataset: https://archive.ics.uci.edu/ml/datasets/HIGGS 

As published in this paper: 
```
Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics with Deep Learning.” Nature Communications 5 (July 2, 2014).
```


### Data description
- First column is the target label, 0 is background, 1 is signal
- There are 28 features in total, of which 21 low-level and 7 high-level features: lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb
- Here we are restricting to the **raw** features to build our model.
- The last 500000 samples are used for the test set.
- The second to last 500000 samples are used for validation set.