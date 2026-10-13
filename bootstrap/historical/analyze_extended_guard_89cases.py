#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd

GUARD = 0.025
BOOT_SEED = 20260917
N_BOOT = 10000
ALPHA = 0.05

def build_ledger(data_dir: Path) -> pd.DataFrame:
    # Existing independent random validation (seeds 20260915/16)
    r = pd.read_csv(data_dir / 'trackB_cm_sppk45_guardband_validation_seed4_seed5.csv')
    a = pd.DataFrame({
        'family': 'unconstrained_random_new_seeds',
        'case_id': r['profile'].astype(str) + '|d' + r['delta_std'].astype(str),
        'fom_scr': r['fom_SCR'],
        'rom_scr': r['rom_SCR'],
    })

    # Cinv-axis validation
    c = pd.read_csv(data_dir / 'trackB_delta20_Xg_Cinv_cm_sppk45_boundary.csv')
    b = pd.DataFrame({
        'family': 'Cinv_axis',
        'case_id': 'Cmult=' + c['Cinv_mult'].astype(str),
        'fom_scr': c['fom_SCRcrit'],
        'rom_scr': c['cm_sppk45_SCRcrit'],
    })

    # PLL-axis validation
    p = pd.read_csv(data_dir / 'trackB_cm_sppk45_pll_boundary.csv')
    d = pd.DataFrame({
        'family': 'PLL_axis',
        'case_id': 'PLLscale=' + p['pll_gain_factor'].astype(str),
        'fom_scr': p['fom_SCRcrit'],
        'rom_scr': p['cm_sppk45_SCRcrit'],
    })

    # Positive correlated profiles
    q = pd.read_csv(data_dir / 'trackB_correlated_positive_oos_32cases.csv')
    e = pd.DataFrame({
        'family': 'correlated_positive',
        'case_id': q['profile'].astype(str) + '|d' + q['delta_cv_target'].astype(str),
        'fom_scr': q['fom_SCR'],
        'rom_scr': q['cm_sppk45_SCR'],
    })

    out = pd.concat([a,b,d,e], ignore_index=True)
    out['required_raw'] = out['fom_scr'] / out['rom_scr'] - 1.0
    out['required_guard'] = np.maximum(out['required_raw'], 0.0)
    out['guarded_scr'] = out['rom_scr'] * (1.0 + GUARD)
    out['guard_overshoot'] = out['guarded_scr'] / out['fom_scr'] - 1.0
    out['guard_safe'] = out['guarded_scr'] >= out['fom_scr'] - 1e-12
    return out

def summarize(df: pd.DataFrame) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    groups = [g['required_guard'].to_numpy(float) for _,g in df.groupby('family', sort=True)]
    means = np.empty(N_BOOT)
    p95s = np.empty(N_BOOT)
    for k in range(N_BOOT):
        vals=[]
        for g in groups:
            vals.extend(g[rng.integers(0,len(g),size=len(g))])
        vals=np.asarray(vals,float)
        means[k]=vals.mean()
        p95s[k]=np.quantile(vals,0.95)
    n=len(df)
    zero_failure_upper = 1.0 - ALPHA**(1.0/n)
    by=[]
    for fam,g in df.groupby('family',sort=True):
        by.append({
            'family':fam,
            'n':int(len(g)),
            'max_required_guard':float(g.required_guard.max()),
            'p95_required_guard':float(g.required_guard.quantile(.95)),
            'mean_required_guard':float(g.required_guard.mean()),
            'guard_failures':int((~g.guard_safe).sum()),
            'mean_guarded_overshoot':float(g.guard_overshoot.mean()),
        })
    return {
        'n_cases':n,
        'guard_fraction':GUARD,
        'guard_failures':int((~df.guard_safe).sum()),
        'max_required_guard':float(df.required_guard.max()),
        'p95_required_guard':float(df.required_guard.quantile(.95)),
        'mean_required_guard':float(df.required_guard.mean()),
        'mean_guarded_overshoot':float(df.guard_overshoot.mean()),
        'min_guarded_overshoot':float(df.guard_overshoot.min()),
        'max_guarded_overshoot':float(df.guard_overshoot.max()),
        'zero_failure_one_sided_95_binomial_upper_bound_iid_interpretation':float(zero_failure_upper),
        'bootstrap_seed':BOOT_SEED,
        'bootstrap_replicates':N_BOOT,
        'stratified_bootstrap_mean_required_guard_95ci':[float(x) for x in np.quantile(means,[.025,.975])],
        'stratified_bootstrap_p95_required_guard_95ci':[float(x) for x in np.quantile(p95s,[.025,.975])],
        'by_family':by,
        'caveat':'Binomial and bootstrap summaries are descriptive under sampling assumptions; they are not a global stability or probability guarantee.'
    }

def main():
    here=Path(__file__).resolve().parent
    data=here.parent/'data'
    # Source-family CSVs should be placed in data/ for a fully raw rebuild.
    # When only the frozen 89-case ledger is distributed, recompute summary from it.
    source_names=[
        'trackB_cm_sppk45_guardband_validation_seed4_seed5.csv',
        'trackB_delta20_Xg_Cinv_cm_sppk45_boundary.csv',
        'trackB_cm_sppk45_pll_boundary.csv',
        'trackB_correlated_positive_oos_32cases.csv']
    if all((data/n).exists() for n in source_names):
        df=build_ledger(data)
    else:
        df=pd.read_csv(data/'extended_guard_validation_89cases.csv')
    summary=summarize(df)
    df.to_csv(data/'extended_guard_validation_89cases_recomputed.csv',index=False)
    (data/'extended_guard_validation_89cases_summary_recomputed.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
