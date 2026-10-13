# PV weak-grid stability-boundary research — public computational source

Associated manuscript (not yet published): **False-Stable Decisions in Reduced-Order PV Models Near Weak-Grid Stability Boundaries**.

This public source selection contains the bounded Python DCO-switching code and frozen inputs for the selected A6 point, plus the fixed 89-case validation ledger and a portable Python script that replays the published descriptive profile-cluster bootstrap. Source files that were copied from the retained scientific staging archive are byte-identical unless explicitly identified as new public documentation or wrapper code.

## Scientific scope

- Benchmark: 40 parallel strings × 10 DCO positions; 1971-dimensional physical tangent FOM.
- The one-sided **2.5% guard** was calibrated on 28 cases and covered **89 held-out sampled boundaries**, but it is **not a universal certificate**.
- A localized-cloud geometry defeats the frozen guard: **4.577%** required correction.
- The A6 Python implementation supports finite-time growth at one selected disagreement point. The global VSC/PLL remains **averaged**, the grid algebraic, and direct native A6 Simulink remains unresolved; **not full EMT or hardware validation**.
- The chosen A6 case belongs to guard calibration, so it is not independent validation of that guard.

## What can be reproduced here

**Bootstrap (directly runnable; no PV simulation):**

```bash
python -m pip install -r bootstrap/requirements.txt
python bootstrap/run_public_cluster_bootstrap.py
```

The script checks the original frozen CSV digest, samples complete two-level profiles with replacement within each stochastic family, fixes the deterministic parameter grids, and compares the output against stored V24 results. The 10,000-replicate statistics are descriptive conditional resampling sensitivity, not confidence intervals for an unspecified population or failure-probability guarantees.

**A6 source:** `a6_switching/` includes original Python switching code and a hash-locked MATLAB parameter export, but the selected publication subset omits full trace arrays and native MATLAB/Simulink session records. The original `scripts/run_switching.py` was designed for a larger retained evidence structure and its evidence gates refer to files not included here. Therefore **this selection is not a self-contained execution or independent verification of the 12-second A6 run**. For its exact protocol, numerical refinements and recorded rates, consult the Supplementary Material and authorized reviewer evidence.

## Rights and availability

Raw third-party datasets are **not redistributed** here. See [DATA_SOURCES.md](DATA_SOURCES.md) for their official identifiers. The **journal manuscript and the confidential unpublished reviewer-draft link** are not in this public repository. The research includes explicit gaps in historical execution/source recovery and does not claim full end-to-end independent reproducibility.

Public visibility is not a grant of an open-source license; see [LICENSE_STATUS.md](LICENSE_STATUS.md).

**Authors:** Ghassan Malkawi (corresponding), Ahmed Elsayed, Mohammed Alhagyan and Bakeel Hussein.
