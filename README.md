# Finding HIGGS in a Haystack

The Higgs boson is one of the elementary particles in the [Standard Model (SM) of Particle Physics](https://en.wikipedia.org/wiki/Standard_Model). It is produced by quantum excitations of the Higgs field, following the principles of Quantum Field Theory. Its role in the SM is crucial as it is proposed to explain how some particles acquire mass. Specifically, bosons have properties as described by the Higgs mechanism. The Higgs particle is very unstable and hence decays into other particles almost immediately.

In the Large Hadron Collider (LHC) we can search for exotic particles like the higgs by combing through the collision data. Of the roughly 10^11 collisions produced per hour, approximately 300 result in a Higgs event [[1](#[1])]. Finding the _signal_ events (with Higgs boson) among the _background_ events is a challenging task. 

In this project we will look at machine learning methods to assist in the discovery of signal events in the collision data.

## Data

The HIGGS dataset [[2](#[2])] is available on the [UCI archive](https://archive.ics.uci.edu/ml/datasets/HIGGS). The data is generated using Monte Carlo simulations and contains 11M records. 


### Data description
- First column is the target label, 0 is background, 1 is signal
- There are 28 features in total, of which 21 low-level and 7 high-level features: lepton pT, lepton eta, lepton phi, missing energy magnitude, missing energy phi, jet 1 pt, jet 1 eta, jet 1 phi, jet 1 b-tag, jet 2 pt, jet 2 eta, jet 2 phi, jet 2 b-tag, jet 3 pt, jet 3 eta, jet 3 phi, jet 3 b-tag, jet 4 pt, jet 4 eta, jet 4 phi, jet 4 b-tag, m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb
- Here we are restricting to the **raw** features to build our model.
- The last 500k samples are used for the test set.
- The second to last 500k samples are used for validation set.


## References

* <a name="[1]">[1]</a> Baldi, P., P. Sadowski, and D. Whiteson. “Searching for Exotic Particles in High-energy Physics with Deep Learning.” Nature Communications 5 (July 2, 2014).
* <a name="[2]">[2]</a> https://archive.ics.uci.edu/ml/datasets/HIGGS 