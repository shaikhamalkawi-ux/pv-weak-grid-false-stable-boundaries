"""New, isolated runner; original scientific module is imported unchanged."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'original_python' / 'g7_python_package'
sys.path.insert(0, str(ORIGINAL))
import g7_python_explicit_switching as sw


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@njit(cache=True)
def monitored(z0, C1, C2, Isc, Voc, vref, p, tstop, subdiv):
    """Identical event/RK4 arithmetic; diagnostics at accepted substep endpoints."""
    nper = int(round(tstop * sw.FSW))
    save_per = 25
    n = C1.size
    out = np.empty((nper // save_per + 1, z0.size))
    out[0] = z0
    z = z0.copy()
    lo = z.copy()
    hi = z.copy()
    duty_lo = 1.0
    duty_hi = 0.0
    j = 1
    for k in range(nper):
        d = sw.duty_from_state(z, n, p)
        duty_lo = min(duty_lo, np.min(d))
        duty_hi = max(duty_hi, np.max(d))
        edges = np.empty(n + 2)
        edges[0] = 0.0
        edges[n+1] = 1.0
        for q in range(n):
            edges[q+1] = d[q]
        edges = np.sort(edges)
        for e in range(edges.size - 1):
            a, b = edges[e], edges[e+1]
            if b-a < 1e-12:
                continue
            mid = 0.5 * (a+b)
            gates = np.empty(n)
            for q in range(n):
                gates[q] = 1.0 if mid < d[q] else 0.0
            h = (b-a) * sw.TSW / subdiv
            for q in range(subdiv):
                z = sw.rk4_gate(z, h, gates, C1, C2, Isc, Voc, vref, p)
                if not np.isfinite(z).all():
                    raise ValueError('Nonfinite integration state')
                lo = np.minimum(lo, z)
                hi = np.maximum(hi, z)
        if (k+1) % save_per == 0:
            out[j] = z
            j += 1
    return out[:j], lo, hi, duty_lo, duty_hi


def prepare(case):
    if case == 'stable':
        return sw.prepare_case(.62, .2)
    from locked_case import prepare_a6
    return prepare_a6()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', choices=['stable','a6'], required=True)
    ap.add_argument('--subdiv', type=int, choices=[16,32,64], default=16)
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    if args.case == 'a6':
        from locked_case import parity_report
        parity = parity_report()
        assert parity['passed'], 'Frozen A6 parameter parity gate failed'
        stable = json.loads((ROOT/'runs'/'stable_sd16'/'result.json').read_text())
        assert stable['admissible'] and stable['stable_replay_gate']['passed'], 'Stable replay gate failed'
    c, z0, C1, C2, Isc, Voc, vref, p = prepare(args.case)
    outdir = ROOT/'runs'/f'{args.case}_sd{args.subdiv}{"_probe" if args.probe else ""}'
    outdir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    source_hashes = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in ORIGINAL.rglob('*.py')}
    write_json(outdir/'run_contract.json', dict(case=args.case, subdiv=args.subdiv,
        started_utc=started, stop_s=.01 if args.probe else 12., sample_s=.001,
        perturb_fraction=.002, scientific_source_hashes=source_hashes,
        initial_state=z0.tolist(), parameter_vector=p.tolist(),
        irradiance=np.asarray(c._S_matrix[0]).tolist(),
        claim_boundary='explicit coherent DCO switching; averaged global VSC/PLL and algebraic grid'))
    print('Compiling and verifying instrumented arithmetic', flush=True)
    tic=time.perf_counter()
    t_ref,y_ref=sw.simulate(z0,C1,C2,Isc,Voc,vref,p,.01,.001,args.subdiv)
    y_test,_,_,_,_=monitored(z0,C1,C2,Isc,Voc,vref,p,.01,args.subdiv)
    error=float(np.max(np.abs(y_test-y_ref)))
    assert np.array_equal(y_test,y_ref), f'Instrumented mismatch {error}'
    print(f'Probe parity exact; setup_seconds={time.perf_counter()-tic:.2f}',flush=True)
    tic=time.perf_counter()
    _,_,_,_,_=monitored(z0,C1,C2,Isc,Voc,vref,p,.01,args.subdiv)
    probe_seconds=time.perf_counter()-tic
    write_json(outdir/'probe.json',dict(max_abs_parity_error=error,
        probe_wall_seconds=probe_seconds, estimated_12s_wall_seconds=1200*probe_seconds))
    if args.probe:
        np.savez_compressed(outdir/'probe.npz',t=t_ref,y=y_ref)
        print('PROBE_COMPLETE',flush=True)
        return
    print(f'Estimated full run {1200*probe_seconds:.1f} seconds; fixed 12 s horizon',flush=True)
    tparts=[np.array([0.])]; yparts=[z0[None,:]]
    z=z0.copy(); lo=z.copy(); hi=z.copy(); dlo=1.; dhi=0.
    tic=time.perf_counter()
    for second in range(12):
        yy,ll,hh,ddlo,ddhi=monitored(z,C1,C2,Isc,Voc,vref,p,1.,args.subdiv)
        tt=second+np.arange(1,1001)*.001
        tparts.append(tt); yparts.append(yy[1:]); z=yy[-1].copy()
        lo=np.minimum(lo,ll); hi=np.maximum(hi,hh); dlo=min(dlo,ddlo); dhi=max(dhi,ddhi)
        np.savez_compressed(outdir/f'checkpoint_{second+1:02d}.npz',t=tt,y=yy[1:])
        print(f'{args.case} sd{args.subdiv}: t={second+1}/12s wall={time.perf_counter()-tic:.1f}s vDC={z[51]:.6f}',flush=True)
    t=np.concatenate(tparts); y=np.concatenate(yparts)
    np.savez_compressed(outdir/'raw_all60states.npz',t=t,y=y)
    np.savetxt(outdir/'trace_vdc.csv',np.column_stack([t,y[:,51]]),delimiter=',',header='time_s,vdc_V',comments='')
    from analyze_validation import analyze_trace
    metrics=analyze_trace(t,y[:,51],1.,11.4)
    sensitivity={f'{a}-{b}s':analyze_trace(t,y[:,51],a,b) for a,b in [(2.,10.),(3.,9.)]}
    exponent_lo=lo[:50:5]/(c.pv.Ns*C2*Voc)
    exponent_hi=hi[:50:5]/(c.pv.Ns*C2*Voc)
    diagnostics=dict(min_inductor_current_A=float(np.min(lo[1:50:5])),
        min_pv_voltage_V=float(np.min(lo[:50:5])), min_output_voltage_V=float(np.min(lo[2:50:5])),
        min_vDC_V=float(lo[51]), max_vDC_V=float(hi[51]), min_duty=float(dlo),max_duty=float(dhi),
        min_exponent=float(np.min(exponent_lo)),max_exponent=float(np.max(exponent_hi)),
        state_min=lo.tolist(),state_max=hi.tolist(),monitor_scope='accepted RK4 substep endpoints; duty each carrier start')
    admissible=bool(np.min(lo[1:50:5])>0 and lo[51]>1 and dlo>1e-6 and dhi<.999999
                    and np.min(exponent_lo)>-50 and np.max(exponent_hi)<50)
    result=dict(case=args.case,subdiv=args.subdiv,completed=True,admissible=admissible,
        wall_seconds=time.perf_counter()-tic,metrics=metrics,sensitivity=sensitivity,diagnostics=diagnostics)
    if args.case=='stable':
        archived=np.genfromtxt(ORIGINAL/'A1_stable_subdiv16.csv',delimiter=',',names=True)
        ay=archived[archived.dtype.names[1]]
        assert len(ay)==len(t)
        prior=json.loads((ORIGINAL/'KEY_RESULTS.json').read_text())['stable_python_subdiv16']
        rmse=float(np.sqrt(np.mean((y[:,51]-ay)**2)))
        fd=abs(metrics['phase_frequency_Hz']-prior['freq_Hz'])
        ad=abs(metrics['cycle_rms_alpha_s_1']-prior['cycle_rms_alpha_s_1'])
        result['stable_replay_gate']=dict(passed=bool(rmse<=.05 and fd<=.002 and ad<=.002),
            vdc_rmse_V=rmse,frequency_error_Hz=fd,cycle_rms_alpha_error=ad)
    write_json(outdir/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
