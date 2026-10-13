"""Load the frozen A6 export and audit its direct Python representation.

This adapter does not solve an MPP or an equilibrium, and never time-integrates
the model.  The historical source and parameter export are read-only inputs.
``prepare_a6`` returns the original switching module's eight values.  Frozen P
is available through ``load_locked()`` or the returned config's ``_locked_P``.
"""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "original_python" / "g7_python_package"
TRACKB = PACKAGE / "trackB_code"
sys.dont_write_bytecode = True
for _path in (PACKAGE, TRACKB):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import p001_mxn_reconstructed as pm
import p001_coherent_nonlinear as cn
from p001_model_scaffold import PVPanel

CASE_ID = "G7_A6_SPPK35_OOS_MAX_FALSE_STABLE"
MAT_PATH = ROOT / "inputs" / "G7_R3_A6_RECONSTRUCTED_P_MATLAB_V2.mat"
MAT_SHA256 = "4cb73d46af1cd944bc67163388fcd229b00a664bb24bec3daba78356446e3ace"
LOCK_PATH = ROOT / "inputs" / "G7_R3_A6_SCIENTIFIC_LOCK.json"
LOCK_SHA256 = "e273f81a8a1d3e16eba1fe9608380118aee4b8985a28703df7a947b0933b17ec"

# Exact order unpacked by the original rhs_gate implementation.
PACK_FIELDS = (
    "pv_multiplier", "pv.Np", "pv.Ns", "Cpv", "L_dco", "Cdc", "M",
    "Rline", "Lline", "Cinv", "Ka", "T1", "T2", "Kpwm", "Vs_mag",
    "Rg", "Xg", "Vdc_ref", "Vpcc_ref_pu", "Vdq_base", "Kp1", "Ki1",
    "Kp2", "Ki2", "Kp3", "Ki3", "Kp4", "Ki4", "Kppll", "Kipll",
    "Ipk_base", "Rf", "Lf",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_inputs():
    for path, expected in ((MAT_PATH, MAT_SHA256), (LOCK_PATH, LOCK_SHA256)):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen input hash mismatch for {path}: {actual}")
    P = loadmat(MAT_PATH, simplify_cells=True)["cases"][CASE_ID]
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8-sig"))
    if lock["caseId"] != CASE_ID or (P["M"], P["N"]) != (40, 10):
        raise ValueError("Frozen A6 case identity or dimensions do not match")
    if P["Xg_pu"] != lock["Xg_pu"] or P["Cinv"] != lock["Cinv_F"]:
        raise ValueError("Frozen parameter dictionary and scientific lock differ")
    if not np.array_equal(P["irradiance"], lock["irradiance_by_position_Wm2"]):
        raise ValueError("Frozen irradiance values differ from the scientific lock")
    return P, lock


def _get_attr(c, path: str):
    value = c
    for key in path.split("."):
        value = getattr(value, key)
    return value


def _get_p(P, path: str):
    value = P
    for key in path.split("."):
        value = value[key]
    return value


def load_locked():
    """Return a freshly loaded, SHA-256-verified frozen parameter dictionary."""
    return _verified_inputs()[0]


def prepare_a6(perturb_fraction=0.002):
    """Return (c, z0, C1, C2, Isc, Voc, vref, p) from locked bytes.

    ``z0`` uses one representative string's current, while the MAT stores
    iL0_total over all M identical strings.  Only global vDC is perturbed.
    A zero perturbation exposes the frozen averaged equilibrium for auditing.
    """
    if not np.isfinite(perturb_fraction):
        raise ValueError("Perturbation fraction must be finite")
    P, _ = _verified_inputs()
    pv = PVPanel(**{f.name: P["pv"][f.name] for f in fields(PVPanel)})
    special = {"pv", "dco_voltage_error_base_V", "irradiance_matrix"}
    kwargs = {f.name: P[f.name] for f in fields(pm.MXNCfg) if f.name not in special}
    kwargs["pv"] = pv
    kwargs["dco_voltage_error_base_V"] = P["DCO_voltage_error_base"]
    kwargs["irradiance_matrix"] = np.tile(P["irradiance"], (P["M"], 1))
    c = pm.MXNCfg(**kwargs)
    c.Vs_mag = float(P["Vs_mag"])
    c._locked_P = P
    constants = tuple(np.asarray(P[name], dtype=float).copy()
                      for name in ("pvC1", "pvC2", "pvIsc", "pvVoc"))
    c._pv_const = tuple(np.tile(value, (c.M, 1)) for value in constants)
    vref = np.asarray(P["vref"], dtype=float).copy()
    c._vref = np.tile(vref, (c.M, 1))
    c._S_matrix = c.irradiance_matrix.copy()
    dco = np.column_stack((P["vpv0"], np.asarray(P["iL0_total"]) / c.M,
                           P["vout0"], P["xint0"], P["xlag0"]))
    z0 = np.concatenate((dco.reshape(-1), np.asarray(P["g0"], dtype=float)))
    z0[5*c.N+1] *= 1 + float(perturb_fraction)
    p = np.asarray([_get_attr(c, name) for name in PACK_FIELDS], dtype=float)
    C1, C2, Isc, Voc = constants
    return c, z0, C1, C2, Isc, Voc, vref, p


def _duty_averaged_rhs(z, C1, C2, Isc, Voc, vref, p):
    """Average the original on/off RHS algebraically, with no integration."""
    import g7_python_explicit_switching as sw
    n = C1.size
    duty = sw.duty_from_state.py_func(z, n, p)
    off = sw.rhs_gate.py_func(z, np.zeros(n), C1, C2, Isc, Voc, vref, p)
    on = sw.rhs_gate.py_func(z, np.ones(n), C1, C2, Isc, Voc, vref, p)
    averaged = off.copy()
    for k in range(n):
        block = slice(5*k, 5*k+5)
        averaged[block] = duty[k]*on[block] + (1-duty[k])*off[block]
    return averaged


def parity_report():
    """Return auditable preflight tests; no scientific simulation is run."""
    c, z, C1, C2, Isc, Voc, vref, p = prepare_a6(0.0)
    P = c._locked_P
    _, lock = _verified_inputs()
    rhs = cn.coherent_rhs(z, c)
    residual = float(np.linalg.norm(rhs, ord=np.inf))
    jacobian = cn.numerical_jacobian(z, c, rel=2e-7)
    eigenvalues = np.linalg.eigvals(jacobian)
    governing = eigenvalues[np.argmax(eigenvalues.real)]
    alpha = float(governing.real)
    frequency = float(abs(governing.imag)/(2*math.pi))
    alpha_error = abs(alpha-lock["FOM_alpha_1_per_s"])
    frequency_error = abs(frequency-lock["FOM_frequency_Hz"])
    expected_pack = np.asarray([_get_p(P, name) for name in PACK_FIELDS], dtype=float)
    packing_errors = {name: float(abs(actual-expected)) for name, actual, expected
                      in zip(PACK_FIELDS, p, expected_pack)}
    property_names = ("Vdq_base", "Ipk_base", "Zbase", "Rg", "Xg",
                      "DCO_voltage_error_base")
    property_errors = {name: float(abs(getattr(c, name)-P[name]))
                       for name in property_names}
    perturbed = prepare_a6(0.002)[1]
    expected_initial = z.copy()
    expected_initial[5*c.N+1] *= 1.002
    averaged_errors = {}
    averaged_status = "PASS"
    try:
        for label, state in (("unperturbed", z), ("initial_0p2pct", perturbed)):
            actual = _duty_averaged_rhs(state, C1, C2, Isc, Voc, vref, p)
            averaged_errors[label] = float(np.max(np.abs(actual-cn.coherent_rhs(state, c))))
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        averaged_status = "NOT_RUN: optional original switching RHS import requires numba"
    power = float(np.dot(P["vpv0"], P["iL0_total"]))
    expected_power = 10011518.655395424
    checks = {
        "input_sha256": True,
        "state_shape_60": z.shape == (60,),
        "finite_parameters_and_state": bool(np.isfinite(p).all() and np.isfinite(z).all()),
        "equilibrium_residual_below_1e_5": residual < 1e-5,
        "alpha_error_below_1e_5": alpha_error < 1e-5,
        "frequency_error_below_1e_5_Hz": frequency_error < 1e-5,
        "parameter_pack_matches_frozen_values": bool(np.allclose(p, expected_pack, rtol=1e-14, atol=1e-14)),
        "derived_properties_match_frozen_values": max(property_errors.values()) < 1e-10,
        "DCO_error_base_equals_Vdc_ref": c.DCO_voltage_error_base == c.Vdc_ref,
        "coherent_physical_scaling": bool(np.allclose(
            [c.M*c.Cpv, c.L_dco/c.M, c.M*c.Cdc],
            [P["Cpv_eq"], P["L_eq"], P["Cdc_eq"]], rtol=1e-14, atol=0)),
        "only_initial_vDC_perturbed": bool(np.array_equal(perturbed, expected_initial)),
        "power_error_below_1e_6_W": abs(power-expected_power) < 1e-6,
    }
    if averaged_errors:
        checks["averaged_original_switching_rhs_agreement_below_1e_6"] = max(averaged_errors.values()) < 1e-6
        if not checks["averaged_original_switching_rhs_agreement_below_1e_6"]:
            averaged_status = "FAIL"
    return {
        "case_id": CASE_ID,
        "profile_uid": lock["profileUid"],
        "simulation_executed": False,
        "MPP_or_equilibrium_resolved": False,
        "parameter_mat_path": str(MAT_PATH),
        "parameter_mat_sha256": MAT_SHA256,
        "scientific_lock_sha256": LOCK_SHA256,
        "checks": checks,
        "passed": all(checks.values()),
        "equilibrium_residual_inf": residual,
        "jacobian_relative_difference_step": 2e-7,
        "coherent_alpha_per_s": alpha,
        "coherent_frequency_Hz": frequency,
        "locked_FOM_alpha_per_s": lock["FOM_alpha_1_per_s"],
        "locked_FOM_frequency_Hz": lock["FOM_frequency_Hz"],
        "alpha_absolute_error": alpha_error,
        "frequency_absolute_error_Hz": frequency_error,
        "parameter_pack_fields": list(PACK_FIELDS),
        "parameter_pack_absolute_errors": packing_errors,
        "derived_property_absolute_errors": property_errors,
        "duty_averaged_rhs_absolute_errors": averaged_errors,
        "duty_averaged_rhs_check_status": averaged_status,
        "total_PV_power_W": power,
        "total_PV_power_absolute_error_W": abs(power-expected_power),
        "irradiance_Wm2": P["irradiance"].tolist(),
        "initial_vDC_V": float(perturbed[5*c.N+1]),
        "frozen_stop_time_s": lock["stopTime_s"],
        "implementation_boundary": "Detailed DCO switching; averaged VSC/PLL; algebraic grid. This preflight only audits the coherent averaged state and RHS, not switching outcomes.",
    }


if __name__ == "__main__":
    report = parity_report()
    (ROOT / 'reports' / 'a6_parameter_parity.json').write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)
