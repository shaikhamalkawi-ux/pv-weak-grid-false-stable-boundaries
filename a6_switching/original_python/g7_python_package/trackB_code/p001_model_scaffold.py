"""P001 source-locked structural scaffold.

This is NOT yet a full reproduction of the authors' Simulink model.
It implements only equations that are explicit in Wang et al. (2020), plus
clearly labelled algebraic inferences used for consistency checks.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize_scalar

@dataclass(frozen=True)
class PVPanel:
    Sref: float = 1000.0
    Tref: float = 25.0
    Isc: float = 9.16
    Voc: float = 35.97
    Im: float = 8.45
    Vm: float = 29.62
    Ns: int = 4
    Np: int = 25
    alpha_T_I: float = 0.0025
    gamma_T_V: float = 0.00288
    beta_S_V: float = 0.5


def env_corrected(p: PVPanel, S_wm2: float, T_c: float, normalized_irradiance_in_log: bool=True):
    """Eq. (1) environmental correction.

    Printed Eq. (1) writes ln[e + beta(S-Sref)] while Table 1 labels S in W/m^2.
    That literal interpretation gives implausibly huge voltage multipliers for a
    +50/+100 W/m^2 step. `normalized_irradiance_in_log=True` instead evaluates
    the irradiance difference in units of Sref. This is an AUDITED INFERENCE,
    not a silently asserted paper convention.
    """
    dI=(1+p.alpha_T_I*(T_c-p.Tref))*S_wm2/p.Sref
    delta=(S_wm2-p.Sref)/p.Sref if normalized_irradiance_in_log else (S_wm2-p.Sref)
    dV=(1-p.gamma_T_V*(T_c-p.Tref))*math.log(math.e+p.beta_S_V*delta)
    return dict(Isc=p.Isc*dI, Im=p.Im*dI, Voc=p.Voc*dV, Vm=p.Vm*dV, dI=dI, dV=dV)


def pv_coefficients(p: PVPanel, S_wm2=1000.0, T_c=25.0, normalized_irradiance_in_log=True):
    q=env_corrected(p,S_wm2,T_c,normalized_irradiance_in_log)
    C2=(q['Vm']/q['Voc']-1.0)/math.log(1.0-q['Im']/q['Isc'])
    C1=(1.0-q['Im']/q['Isc'])*math.exp(-q['Vm']/(C2*q['Voc']))
    return C1,C2,q


def ipv_from_vpv(vpv_v: float, p: PVPanel=PVPanel(), S_wm2=1000.0, T_c=25.0,
                 normalized_irradiance_in_log=True):
    C1,C2,q=pv_coefficients(p,S_wm2,T_c,normalized_irradiance_in_log)
    expo=vpv_v/(p.Ns*C2*q['Voc'])
    return p.Np*q['Isc']*(1-C1*(math.exp(expo)-1))


def numerical_mpp(p: PVPanel=PVPanel(), S_wm2=1000.0, T_c=25.0,
                  normalized_irradiance_in_log=True):
    _,_,q=pv_coefficients(p,S_wm2,T_c,normalized_irradiance_in_log)
    vmax=p.Ns*q['Voc']*0.999999
    f=lambda v: -v*max(0.0,ipv_from_vpv(v,p,S_wm2,T_c,normalized_irradiance_in_log))
    res=minimize_scalar(f,bounds=(0.0,vmax),method='bounded',options={'xatol':1e-12})
    v=float(res.x); i=float(ipv_from_vpv(v,p,S_wm2,T_c,normalized_irradiance_in_log))
    return dict(Vmpp_V=v,Impp_A=i,Pmpp_W=v*i)


def dco_boost_rhs(vpv, iL, vdc_out, duty, idc_out, ipv, Cpv, L, Cdc):
    """Eq. (5), physical-unit form (without the paper's later per-unit omega_B factor)."""
    return np.array([
        (ipv-iL)/Cpv,
        (vpv-(1-duty)*vdc_out)/L,
        ((1-duty)*iL-idc_out)/Cdc,
    ],float)


def dco_compensator_rhs(x_int, x_lag, error, T2):
    """One minimal state-space realization of Eq. (9), explicitly labelled inference.

    Gcomp = Ka*(1+s*T1)/(s*(1+s*T2))
          = Ka/s + Ka*(T1-T2)/(1+s*T2).
    States: x_int_dot=e, T2*x_lag_dot=-x_lag+e.
    """
    return np.array([error,(-x_lag+error)/T2],float)


def dco_compensator_output(x_int,x_lag,Ka,T1,T2):
    return Ka*x_int + Ka*(T1-T2)*x_lag


def expected_state_count(M:int,N:int):
    """Source-locked count: Sec.5.1/5.2 says 2000 DCO + 10 global states for 40x10.
    This implies 5 states/DCO, hence 5*M*N+10. The decomposition of the five DCO
    states as 3 plant + 2 compensator states is an inference from Eqs. (5),(9).
    """
    return 5*M*N+10


def state_layout(M:int,N:int):
    out=[]
    for m in range(M):
        for n in range(N):
            prefix=f'dco[{m+1},{n+1}]'
            out += [prefix+'.vpv',prefix+'.iL',prefix+'.vdc_out',prefix+'.ctrl_int',prefix+'.ctrl_lag']
    out += ['iDC','vDC','x1_PI1','x2_PI2','x3_PI3','x4_PI4','x5_PLL_PI','theta_pcc','id','iq']
    return out


def published_mode_metrics(real,imag):
    w=abs(imag); f=w/(2*math.pi); zeta=-real/math.sqrt(real*real+imag*imag)
    return f,zeta

def lossless_dc_equilibrium_ideal_boost(S_matrix_wm2, vdc_link_v: float, rline_ohm: float,
                                        tracking_error_v: float=0.1, T_c: float=25.0,
                                        p: PVPanel=PVPanel(), normalized_irradiance_in_log=True,
                                        pv_current_multiplier: float=1.0):
    """DC-side steady-state consequence of Eqs. (1),(5),(10) under explicit assumptions.

    Assumptions (not hidden): each DCO regulates its PV input voltage to numerical MPP
    plus the published tracking error; converter switching losses are neglected as in
    the paper's power-balance discussion; all output capacitors have zero derivative;
    equal series current and equal parallel string voltage constraints are enforced;
    DC-line derivative is zero. This does not use VSC/PLL dynamics and is therefore
    a DC steady-state structural check, not a full G2 reproduction.
    """
    Sm=np.asarray(S_matrix_wm2,dtype=float)
    M,N=Sm.shape
    unit=[]
    P=np.zeros_like(Sm); Vin=np.zeros_like(Sm); Iin=np.zeros_like(Sm)
    for m in range(M):
        row=[]
        for n in range(N):
            mp=numerical_mpp(p,float(Sm[m,n]),T_c,normalized_irradiance_in_log)
            v=mp['Vmpp_V']+tracking_error_v
            i=ipv_from_vpv(v,p,float(Sm[m,n]),T_c,normalized_irradiance_in_log)*pv_current_multiplier
            Vin[m,n]=v; Iin[m,n]=i; P[m,n]=v*i
    Pstr=P.sum(axis=1); Ptot=float(Pstr.sum())
    # Vprime = Vdc + 2*Rline*IDC, IDC=Ptot/Vprime
    Vprime=0.5*(vdc_link_v+math.sqrt(vdc_link_v**2+8*rline_ohm*Ptot))
    Istr=Pstr/Vprime
    Vout=np.zeros_like(P); duty=np.zeros_like(P)
    for m in range(M):
        for n in range(N):
            Vout[m,n]=P[m,n]/Istr[m]
            duty[m,n]=1.0-Vin[m,n]/Vout[m,n]
    return dict(M=M,N=N,Vin_V=Vin,Iin_A=Iin,Pin_W=P,Vdc_out_V=Vout,duty=duty,
                string_current_A=Istr,IDC_A=float(Istr.sum()),VDCprime_V=float(Vprime),
                VDC_link_V=float(vdc_link_v),Ptotal_W=Ptot)
