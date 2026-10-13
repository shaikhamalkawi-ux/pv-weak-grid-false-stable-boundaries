# Corrected profile-cluster resampling on the frozen 89-case ledger

**Direct replay:** `python bootstrap/run_public_cluster_bootstrap.py`

Requirements: Python 3.12+ and versions pinned in `requirements.txt`.

The 89-row ledger in `data/extended_guard_validation_89cases.csv` is copied byte-for-byte from the verified source selection; SHA-256 `25dc10cc6cd95a95299475f5c4211ff2a741e2a3eb41b22e3ac68b6ea813f2c6`.

The public script implements the recorded corrected sampling convention independently of the historical 29-file verification closure. It does not execute any PV model, fit any guard, or alter any sample, result, seed or parameter. It uses two stochastic families, each with 16 paired profile identifiers and two delta levels, 10,000 replicates with NumPy seed 20260917 and linear quantiles. The 21 capacitance and four PLL cases remain fixed in each replicate. Its output is cross-checked against `expected/cluster_bootstrap_correction.json`.

The retained 0/89 guarded failures describe the **fixed set**, not a population failure-probability estimate. The nominal empirical 2.5% guard fails in a separate localized-cloud transfer test requiring 4.577% correction. The historical archived analysis script is provided for source review only and is not invoked by the public runner; older row-wise bootstrap results are not the corrected inferential summaries.
