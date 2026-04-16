# Structure

### Helper Files
* `mot_helper.py`
  File containing the helper functions related to the Entropic MOT algorithm.
* `rnd_helper.py`
  File containing helper functions for extracting Risk-Neutral Densities. This includes Implied Volatility calculation, SVI Parametrization, and application of the Breeden-Litzenberger Formula.

### Execution Files
* `rnd_calculation.ipynb`
  This notebook is used to test changes to the way we extract Risk-Neutral Densities.
* `entropic_MOT_logreturns.ipynb`
  The execution file for the Entropic MOT algorithm, applied to the density of log returns extracted from Nvidia call option data. Volatility controlling is included here.
* `entropic_MOT_synthetic_data.ipynb`
  The execution file for the Entropic MOT algorithm, used with synthetic data generated from a Geometric Brownian Motion. Volatility controlling is included here.