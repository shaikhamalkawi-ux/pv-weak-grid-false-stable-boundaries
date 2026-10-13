"""Scalable source-constrained P001 reconstruction for MxN boost-DCO arrays.

This is an independent reconstruction from Wang et al. (2020) equations and figures.
It is not the authors' original Simulink implementation. Every inferred base/sign is
kept explicit in the configuration.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from p001_model_scaffold import PVPanel, numerical_mpp, pv_coefficients, lossless_dc_equilibrium_ideal_boost

@dataclass
class MXNCfg:
    M:int
    N:int
    pv:PVPanel=field(default_factory=PVPanel)
    pv_multiplier:float=1.0
    tracking_error_v:float=0.1
    # DCO/line/DC-link
    L_dco:float=318.3e-6
    Cpv:float=5e-6
    Cdc:float=159.2e-6
    Rline:float=0.0946
    Lline:float=1.0635e-3
    Cinv:float=25000e-6
    Ka:float=5.0
    T1:float=1.0
    T2:float=0.5
    Kpwm:float=0.5
    # VSC / PLL
    Lf:float=2.865e-3
    Rf:float=0.001
    Kp1:float=0.1; Ki1:float=10.0
    Kp2:float=0.1; Ki2:float=10.0
    Kp3:float=0.2; Ki3:float=20.0
    Kp4:float=0.2; Ki4:float=20.0
    Kppll:float=10.0; Kipll:float=1500.0
    # system bases/operating point
    Sbase:float=10e6
    Vll_base:float=3000.0
    Vdc_ref:float=1500.0
    f:float=50.0
    Vpcc_ref_pu:float=1.0
    Q_operating:float=3e6
    Rg_pu:float=0.01
    Xg_pu:float=0.5
    dco_voltage_error_base_V:float|None=None
    irradiance_matrix:object|None=None  # optional MxN W/m^2 array for mismatch studies

    @property
    def Vdq_base(self): return math.sqrt(2/3)*self.Vll_base
    @property
    def Ipk_base(self): return math.sqrt(2)*self.Sbase/(math.sqrt(3)*self.Vll_base)
    @property
    def Zbase(self): return self.Vll_base**2/self.Sbase
    @property
    def Rg(self): return self.Rg_pu*self.Zbase
    @property
    def Xg(self): return self.Xg_pu*self.Zbase
    @property
    def DCO_voltage_error_base(self): return self.Vdc_ref if self.dco_voltage_error_base_V is None else self.dco_voltage_error_base_V


def unpack(y,c:MXNCfg):
    k=5*c.M*c.N
    return y[:k].reshape(c.M,c.N,5), y[k:]


def _prepare_pv(c:MXNCfg,S=1000.0,T=25.0):
    Sm=np.asarray(c.irradiance_matrix if c.irradiance_matrix is not None else S,dtype=float)
    if Sm.ndim==0: Sm=np.full((c.M,c.N),float(Sm))
    if Sm.shape!=(c.M,c.N): raise ValueError(f"irradiance shape {Sm.shape} != {(c.M,c.N)}")
    C1=np.empty_like(Sm); C2=np.empty_like(Sm); Isc=np.empty_like(Sm); Voc=np.empty_like(Sm); vref=np.empty_like(Sm)
    for m in range(c.M):
        for n in range(c.N):
            c1,c2,q=pv_coefficients(c.pv,float(Sm[m,n]),T,True)
            mp=numerical_mpp(c.pv,float(Sm[m,n]),T,True)
            C1[m,n]=c1; C2[m,n]=c2; Isc[m,n]=q['Isc']; Voc[m,n]=q['Voc']; vref[m,n]=mp['Vmpp_V']+c.tracking_error_v
    c._pv_const=(C1,C2,Isc,Voc)
    c._vref=vref
    c._S_matrix=Sm


def rhs(y,c:MXNCfg):
    if not hasattr(c,'_pv_const'): _prepare_pv(c)
    a,g=unpack(y,c)
    vpv=a[:,:,0]; iL=a[:,:,1]; vout=a[:,:,2]; xint=a[:,:,3]; xlag=a[:,:,4]
    iDC,vDC,x1,x2,x3,x4,x5,theta,id_,iq_=g
    C1,C2,Isc,Voc=c._pv_const
    expo=vpv/(c.pv.Ns*C2*Voc)
    ipv=c.pv_multiplier*c.pv.Np*Isc*(1-C1*(np.exp(expo)-1))
    # Fig.3 sign: vpv positive, vpvm negative
    e=(vpv-c._vref)/c.DCO_voltage_error_base
    dxint=e
    dxlag=(-xlag+e)/c.T2
    ucomp=c.Ka*xint+c.Ka*(c.T1-c.T2)*xlag
    d=np.clip(c.Kpwm*ucomp,1e-6,0.999999)
    idc_in=(1-d)*iL
    dvpv=(ipv-iL)/c.Cpv
    diL=(vpv-(1-d)*vout)/c.L_dco
    # Physical series-current and equal-parallel-string-voltage constraints.
    invC=1.0/c.Cdc
    aa=np.sum(idc_in*invC,axis=1)
    bb=np.full(c.M,c.N*invC)
    q_common=(np.sum(aa/bb)-iDC)/np.sum(1.0/bb)
    i_string=(aa-q_common)/bb
    dvout=(idc_in-i_string[:,None])/c.Cdc
    vprime=float(np.mean(np.sum(vout,axis=1)))
    diDC=(vprime-vDC-2*c.Rline*iDC)/(2*c.Lline)

    I=id_+1j*iq_
    V=c.Vs_mag*np.exp(-1j*theta)+(c.Rg+1j*c.Xg)*I
    vgd=float(V.real); vgq=float(V.imag); vmag=abs(V)
    Pac=1.5*(vgd*id_+vgq*iq_)
    iinv=Pac/max(vDC,1.0)
    dvDC=(iDC-iinv)/c.Cinv

    # Fig.4(b) outer-loop signs: actual signal +, reference -.
    evdc=(vDC-c.Vdc_ref)/c.Vdc_ref
    dx1=evdc
    idref=(c.Kp1*evdc+c.Ki1*x1)*c.Ipk_base
    ev=(vmag-c.Vpcc_ref_pu*c.Vdq_base)/c.Vdq_base
    dx2=ev
    iqref=(c.Kp2*ev+c.Ki2*x2)*c.Ipk_base
    eid=(idref-id_)/c.Ipk_base; eiq=(iqref-iq_)/c.Ipk_base
    dx3=eid; dx4=eiq
    ud=(c.Kp3*eid+c.Ki3*x3)*c.Vdq_base
    uq=(c.Kp4*eiq+c.Ki4*x4)*c.Vdq_base
    did=(ud-c.Rf*id_)/c.Lf
    diq=(uq-c.Rf*iq_)/c.Lf
    epll=vgq/c.Vdq_base
    dx5=epll
    dtheta=c.Kppll*epll+c.Kipll*x5
    return np.r_[np.stack([dvpv,diL,dvout,dxint,dxlag],axis=2).reshape(-1),
                 [diDC,dvDC,dx1,dx2,dx3,dx4,dx5,dtheta,did,diq]]


def equilibrium(c:MXNCfg,S=1000.0):
    _prepare_pv(c,S)
    dc=lossless_dc_equilibrium_ideal_boost(c._S_matrix,c.Vdc_ref,c.Rline,
                                            c.tracking_error_v,pv_current_multiplier=c.pv_multiplier)
    a=np.zeros((c.M,c.N,5))
    for m in range(c.M):
        for n in range(c.N):
            duty=dc['duty'][m,n]
            a[m,n]=[dc['Vin_V'][m,n],dc['Iin_A'][m,n],dc['Vdc_out_V'][m,n],
                    duty/(c.Kpwm*c.Ka),0.0]
    iDC=dc['IDC_A']; vDC=c.Vdc_ref; P=vDC*iDC
    Vd=c.Vpcc_ref_pu*c.Vdq_base
    id_=P/(1.5*Vd); iq_=-c.Q_operating/(1.5*Vd)
    Z=c.Rg+1j*c.Xg
    Vs_rot=Vd-Z*(id_+1j*iq_)
    c.Vs_mag=abs(Vs_rot); theta=-math.atan2(Vs_rot.imag,Vs_rot.real)
    x1=(id_/c.Ipk_base)/c.Ki1
    x2=(iq_/c.Ipk_base)/c.Ki2
    x3=((c.Rf*id_)/c.Vdq_base)/c.Ki3
    x4=((c.Rf*iq_)/c.Vdq_base)/c.Ki4
    x5=0.0
    g=np.array([iDC,vDC,x1,x2,x3,x4,x5,theta,id_,iq_])
    return np.r_[a.reshape(-1),g],dc,dict(Pdc_W=P,Q_var=c.Q_operating,Vpcc_V=Vd,
                                          Vs_mag_V=c.Vs_mag,theta_rad=theta,id_A=id_,iq_A=iq_)


def numerical_jacobian(y,c:MXNCfg,rel=2e-6,absstep=1e-8):
    y=np.asarray(y,float); n=len(y); J=np.empty((n,n),float)
    for j in range(n):
        h=max(absstep,rel*max(1.0,abs(y[j])))
        yp=y.copy(); ym=y.copy(); yp[j]+=h; ym[j]-=h
        J[:,j]=(rhs(yp,c)-rhs(ym,c))/(2*h)
    return J


def state_labels(c:MXNCfg):
    labels=[]
    for m in range(c.M):
        for n in range(c.N):
            labels += [f'vpv_{m+1}_{n+1}',f'iL_{m+1}_{n+1}',f'vout_{m+1}_{n+1}',
                       f'xint_{m+1}_{n+1}',f'xlag_{m+1}_{n+1}']
    return labels+['iDC','vDC','x1_vdc','x2_vpcc','x3_id','x4_iq','x5_pll','theta','id','iq']


def parallel_voltage_constraint_reduction(J,c:MXNCfg):
    """Eliminate M-1 redundant string-total-voltage coordinates.

    The ambient ODE preserves equality of total parallel-string voltages but permits
    off-manifold perturbations, which appear as near-zero modes. This returns a
    coordinate-reduced Jacobian on the tangent manifold sum_n dvout_mn = sum_n dvout_1n.
    It is a reconstruction device, not a claim about the authors' exact Simulink state basis.
    """
    from scipy.sparse import lil_matrix, csr_matrix
    n=J.shape[0]
    def vout_idx(m,nn): return 5*(m*c.N+nn)+2
    eliminated=[vout_idx(m,0) for m in range(1,c.M)]
    elimset=set(eliminated)
    keep=[i for i in range(n) if i not in elimset]
    pos={full:j for j,full in enumerate(keep)}
    T=lil_matrix((n,len(keep)),dtype=float)
    for full,j in pos.items(): T[full,j]=1.0
    for m,e in enumerate(eliminated,start=1):
        for nn in range(c.N): T[e,pos[vout_idx(0,nn)]] += 1.0
        for nn in range(1,c.N): T[e,pos[vout_idx(m,nn)]] -= 1.0
    T=csr_matrix(T)
    Jred=np.asarray(J[keep,:] @ T)
    labels_full=state_labels(c)
    labels_red=[labels_full[i] for i in keep]
    return Jred,keep,T,labels_red


def coherent_average_projection(c:MXNCfg):
    """15-state coherent averaging projection for a uniform MxN array.

    Five columns represent coherent motion of each DCO state type across all M*N
    units; ten columns preserve the global states exactly. W^T V = I. This is a
    simple symmetry/aggregation baseline, not claimed as a new MOR method.
    """
    n=5*c.M*c.N+10
    V=np.zeros((n,15)); W=np.zeros((n,15)); K=c.M*c.N
    for typ in range(5):
        inds=[5*q+typ for q in range(K)]
        V[inds,typ]=1.0
        W[inds,typ]=1.0/K
    base=5*K
    for j in range(10):
        V[base+j,5+j]=1.0; W[base+j,5+j]=1.0
    return V,W
