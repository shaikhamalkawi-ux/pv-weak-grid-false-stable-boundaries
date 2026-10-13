# Python A6 implementation — selected public source

The original retained `g7_python_explicit_switching.py` module and its three `trackB_code/` dependencies are byte-identical to the V24 staging archive. Only their **folder arrangement** was adapted for imports expected by the original `locked_case.py`, which loads the SHA-256-locked `.mat` and scientific-input JSON.

This is **selected code and compact expected results**, not a complete self-contained archived trace package. In particular, `run_switching.py` requires saved stable-side replay and report files that are deliberately absent from this public selection. The full pointwise refinement and verification use private/reviewer-authorized evidence records; this public tree is not an independent A6 simulation rerun.

- Selected A6 profile: seed 20260913, random 02, delta 0.30; held out of SPPK35 basis construction but later included in guard calibration.
- Frozen A6 +0.2% perturbation, 25-kHz DCO ideal switching, 16/32 RK4 subdivisions.
- The 16/32 result supports finite-time positive growth at approximately 1.093 Hz; switching-vs-averaged-FOM rate discrepancy 18.12%.
- The global VSC/PLL remains averaged and the grid algebraic; no successful native A6 Simulink switching trace, full EMT, Floquet proof or hardware validation is claimed.

See `expected/switching_results_table.csv` and `expected/convergence_report.json` for saved output values. The supplied numerical input MAT file is a retained frozen parameter export, not a new measurement.
