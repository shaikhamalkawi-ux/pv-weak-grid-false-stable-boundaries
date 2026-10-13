"""Frozen-protocol trace analysis and read-only replay of archived G7 evidence.

No model integration is performed here.  For valid, positive-frequency traces,
the filtering, windows, and three envelope estimators reproduce the archived
``analyze_traces.py`` algorithm without changing its numerical operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt
from scipy.stats import linregress


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "original_python" / "g7_python_package"
REPLAY_WINDOWS = {
    "A1_stable_subdiv16.csv": (1.0, 11.4),
    "A1_stable_half_amp_subdiv8.csv": (1.0, 11.4),
    "A3_false_stable_subdiv8.csv": (2.0, 20.0),
    "A3_half_amp_subdiv8.csv": (2.0, 20.0),
    "A3_refine_subdiv10_15s.csv": (2.0, 13.0),
    "A3_refine_subdiv12_10s.csv": (2.0, 9.0),
}


def _fit(x, y, label, warnings):
    """Do not silently fit an empty/nonfinite envelope or invent a rate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        warnings.append(f"{label}: fewer than three fit points ({len(x)}).")
    if len(x) < 2:
        return None
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        warnings.append(f"{label}: nonfinite fit input; rate not estimated.")
        return None
    if np.ptp(x) == 0:
        warnings.append(f"{label}: all fit times identical; rate not estimated.")
        return None
    fit = linregress(x, y)
    if not np.isfinite(fit.rvalue):
        warnings.append(f"{label}: correlation is undefined; R2 not reported.")
    return fit


def _slope(fit):
    return None if fit is None or not np.isfinite(fit.slope) else float(fit.slope)


def _r2(fit):
    return None if fit is None or not np.isfinite(fit.rvalue) else float(fit.rvalue**2)


def analyze_trace(t, vdc, t1=1.0, t2=11.4):
    """Return JSON-native metrics from the original fixed-window algorithm.

    Envelope rates are not classified as stable/unstable here.  No samples are
    removed, no window is chosen from the result, and no reference value enters
    the fitting.  Unavailable fits are represented by ``None`` plus a warning.
    """
    t, x = np.asarray(t, float), np.asarray(vdc, float)
    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x) or len(t) < 2:
        raise ValueError("t and vdc must be equal-length one-dimensional traces.")
    if not (np.isfinite(t).all() and np.isfinite(x).all()):
        raise ValueError("Trace contains nonfinite time or voltage samples.")
    if not np.all(np.diff(t) > 0):
        raise ValueError("Trace times must be strictly increasing.")
    if not (np.isfinite(t1) and np.isfinite(t2) and t2 > t1):
        raise ValueError("Analysis window must be finite and increasing.")

    warnings = []
    if t1 < t[0] or t2 > t[-1]:
        warnings.append("Trace does not cover the full requested analysis window.")
    dt = np.median(np.diff(t))
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        warnings.append("Sampling is nonuniform; archived estimator uses median dt without resampling.")
    fs = 1 / dt
    sos = butter(4, [.2, 3], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x - x.mean())
    m = (t >= t1) & (t <= t2)
    h = hilbert(xf)
    env = np.abs(h)
    with np.errstate(divide="ignore", invalid="ignore"):
        rh = _fit(t[m], np.log(env[m]), "Hilbert envelope", warnings)
    ph = np.unwrap(np.angle(h))
    rp = _fit(t[m], ph[m], "Hilbert phase", warnings)

    if m.any():
        pk, _ = find_peaks(
            np.abs(xf[m]),
            distance=int(fs * .30),
            prominence=max(1e-9, .02 * np.std(xf[m])),
        )
        pt, pa = t[m][pk], np.abs(xf[m][pk])
        q = (pt >= t1 + .5) & (pt <= t2 - .5) & (pa > 0)
        rpk = _fit(pt[q], np.log(pa[q]), "Peak envelope", warnings)
        peak_points = int(q.sum())
    else:
        rpk = _fit([], [], "Peak envelope", warnings)
        peak_points = 0

    f = None if rp is None else float(rp.slope / (2 * np.pi))
    cc, rr = [], []
    if f is None or not np.isfinite(f) or f <= 0:
        warnings.append("Phase frequency is unavailable or <= 0; cycle-RMS fit not estimated.")
        rrfit = None
    else:
        ncy = max(8, int(round(fs / f)))
        step = max(4, ncy // 4)
        for i in range(0, len(xf) - ncy + 1, step):
            ct = t[i + ncy // 2]
            if t1 <= ct <= t2:
                cc.append(ct)
                rr.append(np.sqrt(np.mean(xf[i:i + ncy]**2)))
        with np.errstate(divide="ignore", invalid="ignore"):
            rrfit = _fit(np.asarray(cc), np.log(np.asarray(rr)), "Cycle-RMS envelope", warnings)

    return {
        "hilbert_alpha_s_1": _slope(rh),
        "hilbert_r2": _r2(rh),
        "peak_alpha_s_1": _slope(rpk),
        "peak_r2": _r2(rpk),
        "cycle_rms_alpha_s_1": _slope(rrfit),
        "cycle_rms_r2": _r2(rrfit),
        "phase_frequency_Hz": f,
        "phase_r2": _r2(rp),
        "analysis_window_s": [float(t1), float(t2)],
        "sampling_frequency_Hz": float(fs),
        "trace_samples": int(len(t)),
        "fit_points": {
            "hilbert_envelope": int(m.sum()),
            "hilbert_phase": int(m.sum()),
            "peak_envelope": peak_points,
            "cycle_rms": len(cc),
        },
        "warnings": warnings,
    }


def _comparison(actual, expected, atol, rtol):
    finite = actual is not None and np.isfinite(actual)
    difference = float(abs(actual - expected)) if finite else None
    tolerance = float(atol + rtol * abs(expected))
    return {
        "actual": None if not finite else float(actual),
        "expected": float(expected),
        "absolute_difference": difference,
        "allowed_difference": tolerance,
        "pass": bool(finite and difference <= tolerance),
    }


def replay_archived(archive_dir=DEFAULT_ARCHIVE, *, atol=1e-10, rtol=1e-9):
    """Re-analyze six archived traces, comparing all archived replay metrics.

    Also independently check the stable-side metrics printed in KEY_RESULTS.
    Files under ``archive_dir`` are read only; this function writes nothing.
    """
    archive_dir = Path(archive_dir).resolve()
    expected_replay = json.loads((archive_dir / "TRACE_ANALYSIS_REPLAY.json").read_text())
    key_results = json.loads((archive_dir / "KEY_RESULTS.json").read_text())
    results, by_file = [], {}
    for expected in expected_replay:
        filename = expected["file"]
        if filename not in REPLAY_WINDOWS:
            raise ValueError(f"No frozen window supplied for archived trace {filename!r}.")
        t1, t2 = REPLAY_WINDOWS[filename]
        path = archive_dir / filename
        a = np.genfromtxt(path, delimiter=",", names=True)
        names = a.dtype.names
        if names is None or len(names) < 2:
            raise ValueError(f"Archived CSV lacks the expected time/voltage columns: {path}")
        actual = analyze_trace(a[names[0]], a[names[1]], t1, t2)
        checks = {
            key: _comparison(actual[key], value, atol, rtol)
            for key, value in expected.items() if key != "file"
        }
        entry = {
            "file": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "analysis": actual,
            "checks": checks,
            "pass": bool(all(check["pass"] for check in checks.values())),
        }
        results.append(entry)
        by_file[filename] = actual

    if set(by_file) != set(REPLAY_WINDOWS):
        raise ValueError("Archived replay list does not contain the exact six frozen traces.")
    stable = by_file["A1_stable_subdiv16.csv"]
    stable_key_checks = {
        key: _comparison(stable["phase_frequency_Hz" if key == "freq_Hz" else key], value, atol, rtol)
        for key, value in key_results["stable_python_subdiv16"].items()
    }
    return {
        "operation": "archived_trace_reanalysis_only_no_simulation",
        "archive_directory": str(archive_dir),
        "algorithm": "Original archived fourth-order 0.2-3 Hz bandpass, Hilbert/peak/cycle-RMS fits",
        "numeric_tolerance": {"absolute": float(atol), "relative": float(rtol)},
        "trace_replay": results,
        "stable_KEY_RESULTS_checks": stable_key_checks,
        "pass": bool(
            all(entry["pass"] for entry in results)
            and all(check["pass"] for check in stable_key_checks.values())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="replay", choices=["replay"])
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "archive_replay.json")
    args = parser.parse_args()
    report = replay_archived(args.archive_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "traces": len(report["trace_replay"]), "report": str(args.output)}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
