"""Replay the V24 descriptive profile-cluster bootstrap on its frozen 89-case CSV.

This is an independently packaged implementation of the declared resampling
algorithm, not a PV simulation, new result, population CI, or guard certificate.
The original hash-bound historical replay is a separate research record.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "extended_guard_validation_89cases.csv"
EXPECTED = ROOT / "expected" / "cluster_bootstrap_correction.json"
SEED = 20260917
REPLICATES = 10000
GUARD = .025
STOCHASTIC = {"unconstrained_random_new_seeds", "correlated_positive"}
FIXED = {"Cinv_axis", "PLL_axis"}
SOURCE_LEDGER_SHA256 = "25dc10cc6cd95a95299475f5c4211ff2a741e2a3eb41b22e3ac68b6ea813f2c6"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replay():
    if sha256(DATA) != SOURCE_LEDGER_SHA256:
        raise ValueError("Frozen 89-case CSV hash mismatch")
    df = pd.read_csv(DATA)
    if len(df) != 89 or not df.case_id.is_unique:
        raise ValueError("The input ledger is not the frozen 89-case set")
    if set(df.family) != STOCHASTIC | FIXED:
        raise ValueError("Unexpected evaluation family")
    nums = df[["fom_scr", "rom_scr", "required_guard"]].to_numpy(float)
    if not np.isfinite(nums).all():
        raise ValueError("Nonfinite source values")
    if np.max(abs(np.maximum(df.fom_scr / df.rom_scr - 1, 0) - df.required_guard)) >= 2e-14:
        raise ValueError("Required correction identity mismatch")
    guarded = df.rom_scr*(1+GUARD) >= df.fom_scr-1e-12
    if not np.array_equal(guarded.to_numpy(), df.guard_safe.to_numpy()):
        raise ValueError("Saved guard decisions differ")
    grouped = {}
    for family, g in df.groupby("family", sort=True):
        if family in STOCHASTIC:
            split = g.case_id.str.rsplit("|d", n=1, expand=True)
            g = g.assign(profile=split[0], delta=split[1].astype(float))
            pairs = []
            for profile, pair in g.groupby("profile", sort=True):
                pair = pair.sort_values("delta")
                if len(pair) != 2 or pair.delta.tolist() != [.2, .3]:
                    raise ValueError("Stochastic profile pair is incomplete")
                pairs.append(pair.required_guard.to_numpy(float))
            if len(pairs) != 16: raise ValueError("Expected 16 profiles per stochastic family")
            grouped[family] = np.asarray(pairs)
        else:
            expected_n = 21 if family == "Cinv_axis" else 4
            if len(g) != expected_n: raise ValueError("Fixed family count changed")
            grouped[family] = g.required_guard.to_numpy(float)

    rng = np.random.default_rng(SEED)
    means = np.empty(REPLICATES)
    p95s = np.empty(REPLICATES)
    for i in range(REPLICATES):
        values = []
        for family in sorted(grouped):
            arr = grouped[family]
            if family in STOCHASTIC:
                selection = rng.integers(0,16,size=16)
                values.extend(arr[selection].reshape(-1))
            else: values.extend(arr)
        values = np.asarray(values,float)
        if len(values) != 89: raise ValueError("Resample size changed")
        means[i] = values.mean()
        p95s[i] = np.quantile(values,.95,method="linear")
    result = {
        "n_cases": len(df),
        "seed": SEED,
        "replicates": REPLICATES,
        "guard_fraction": GUARD,
        "guard_failures": int((~guarded).sum()),
        "observed_mean_correction_fraction": float(df.required_guard.mean()),
        "observed_p95_correction_fraction": float(np.quantile(df.required_guard,.95,method="linear")),
        "observed_max_correction_fraction": float(df.required_guard.max()),
        "cluster_mean_interval_fraction": np.quantile(means,[.025,.975],method="linear").tolist(),
        "cluster_p95_interval_fraction": np.quantile(p95s,[.025,.975],method="linear").tolist(),
        "interpretation": "Descriptive fixed-family profile-cluster sensitivity, not a population CI or probability guarantee"
    }
    e = json.loads(EXPECTED.read_text())
    targets = {
       "cluster_mean_interval_fraction": e["cluster_sensitivity_mean_interval_fraction"],
       "cluster_p95_interval_fraction": e["cluster_sensitivity_p95_interval_fraction"],
       "observed_mean_correction_fraction": e["unchanged_finite_set_statistics"]["mean_required_guard"],
       "observed_p95_correction_fraction": e["unchanged_finite_set_statistics"]["p95_required_guard"],
       "observed_max_correction_fraction": e["unchanged_finite_set_statistics"]["max_required_guard"],
    }
    for key, reference in targets.items():
        if not np.allclose(result[key], reference,atol=1e-14,rtol=1e-12):
            raise AssertionError(f"Saved output mismatch: {key}")
    result["matches_saved_v24_summary"] = True
    return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="Optional JSON output file")
    a=parser.parse_args()
    result=replay()
    print(json.dumps(result,indent=2,allow_nan=False))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
