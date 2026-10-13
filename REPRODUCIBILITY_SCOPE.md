# Reproducibility scope and claim boundaries

This **public** repository currently contains project documentation. No claim of full source-code release or complete end-to-end reproduction is made until the scientific files are separately reviewed and uploaded.

- **Benchmark and scope:** 40 × 10 PV/DC-optimizer benchmark, 1971-dimensional physical tangent state space, and physical weak-grid boundary decisions.
- **Sampled guard:** a 2.5% one-sided decision guard calibrated on 28 cases covered 89 held-out evaluated boundaries. This is **not** a universal certificate or a failure-probability guarantee.
- **Transfer counterexample:** a retained localized-cloud case requires a 4.577% correction and defeats the 2.5% guard.
- **Selected A6 check:** the Python explicit-DCO switching evidence supports finite-time growth at one selected point with averaged global VSC/PLL. It is not full EMT, HIL, hardware validation, or successful native A6 Simulink execution. A6 belonged to guard calibration and is not an independent held-out guard test.
- **Bootstrap:** descriptive 10,000-replicate profile-cluster resampling of a fixed 89-case ledger, with deterministic parameter grids held fixed; not population inference.
- **Source limitations:** some legacy drivers, fits, source reconstruction chains, and numerical diagnostics are unavailable. Any eventual public code selection must state what it can independently reproduce.
- **External records:** measurements from other PV/inverter systems are not co-located hardware validation of the benchmark's physical stability boundary.

See the manuscript and its Supplementary Material for the authoritative interpretation. A separately permissioned, unpublished Zenodo draft may be used for confidential editorial review; no secret link or access token will be published here.
