# Reader and reviewer guide — PV Matrix Stability

**Associated work:** *False-Stable Decisions in Reduced-Order PV Models Near Weak-Grid Stability Boundaries* (author-prepared journal manuscript; **not represented here as accepted or published**).

## Where to start

- [`README.md`](README.md): scope, study boundaries and run instructions.
- [`bootstrap/README.md`](bootstrap/README.md): **directly replayable** descriptive profile-cluster resampling of the 89-case ledger. Run `python -m pip install -r bootstrap/requirements.txt`, then `python bootstrap/run_public_cluster_bootstrap.py`. Verify saved expected output and seed `20260917`.
- [`a6_switching/README.md`](a6_switching/README.md): selected Python explicit-DCO switching implementation, frozen input and expected compact result summaries. It is **not a fully self-contained independent rerun** of the original A6 12-second simulation.
- [`REPRODUCIBILITY_SCOPE.md`](REPRODUCIBILITY_SCOPE.md): detailed inclusions, exclusions and historical source limits.
- [`DATA_SOURCES.md`](DATA_SOURCES.md): original third-party dataset identifiers, scientific roles and access conditions.
- [`LICENSE_STATUS.md`](LICENSE_STATUS.md): source licensing/reuse status. Public access is not automatically permission to relicense.

## Scientific interpretation

The study's 40 × 10 PV-DCO benchmark has a 1,971-dimensional physical tangent model. A one-sided 2.5% correction calibrated on 28 cases covered 89 held-out sampled boundaries but failed on a separately retained localized-cloud transfer case requiring 4.577%. Neither the empirical guard nor the descriptive bootstrap is a universal or population-level guarantee.

The bounded A6 explicit-DCO switching evidence retains **averaged global VSC/PLL and an algebraic grid**. It is not full EMT, HIL, field or hardware validation. Subsequent October 2026 MATLAB Online attempts produced **zero new usable native Simulink time traces** and were classified HOLD; archived editor screenshots cannot be read as new simulation results.

## Evidence available separately

The manuscript's Supplementary Material and separately supplied reviewer evidence archive provide additional historical audit material beyond this *selected* public repository. Full benchmark workflows, some historical raw traces, third-party raw datasets, journal-private correspondence and confidential Zenodo draft preview credentials are intentionally not placed in this public repository.

**No article DOI has yet been registered here.** Please cite the peer-reviewed article if and when published; [`CITATION.cff`](CITATION.cff) states its current status.

**Contact:** Ghassan Malkawi, corresponding author, `gmalkawi@hct.ac.ae`.
