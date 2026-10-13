import sys, math, time, numpy as np
from pathlib import Path
from numba import njit
HERE=Path(__file__).resolve().parent
# In the manuscript source tree, Track-B code is two levels above this folder.
# In the standalone audit ZIP, copy/symlink the trackB_code folder next to this script or set TRACKB_CODE.
import os
trackb=os.environ.get('TRACKB_CODE')
if trackb is None:
    candidate=HERE/'trackB_code' if (HERE/'trackB_code').exists() else HERE.parent.parent/'trackB_code'
    if candidate.exists(): trackb=str(candidate)
if trackb is None:
    raise RuntimeError('Set TRACKB_CODE to the bundled Track-B code directory')
sys.path.insert(0,trackb)
import p001_mxn_reconstructed as pm
import p001_coherent_nonlinear as cn
FSW=25000.0; TSW=1/FSW

def prepare_case(Xg_pu=0.663515, pert_pct=0.2):
    hi=1193.285870824; lo=795.523913883
    row=np.r_[np.full(5,hi),np.full(5,lo)]
    Sm=np.tile(row,(40,1))
    c=pm.MXNCfg(M=40,N=10,irradiance_matrix=Sm,Xg_pu=Xg_pu,Cinv=0.025,Vdc_ref=8300.0,Kp1=0.5,Ki1=10.0)
    y,dc,op=pm.equilibrium(c); z=cn.coherent_state_from_full(y,c)
    z[5*c.N+1] *= 1+pert_pct/100
    C1=c._pv_const[0][0].copy(); C2=c._pv_const[1][0].copy(); Isc=c._pv_const[2][0].copy(); Voc=c._pv_const[3][0].copy(); vref=c._vref[0].copy()
    p=np.array([c.pv_multiplier,c.pv.Np,c.pv.Ns,c.Cpv,c.L_dco,c.Cdc,c.M,c.Rline,c.Lline,c.Cinv,c.Ka,c.T1,c.T2,c.Kpwm,c.Vs_mag,c.Rg,c.Xg,c.Vdc_ref,c.Vpcc_ref_pu,c.Vdq_base,c.Kp1,c.Ki1,c.Kp2,c.Ki2,c.Kp3,c.Ki3,c.Kp4,c.Ki4,c.Kppll,c.Kipll,c.Ipk_base,c.Rf,c.Lf],float)
    return c,z,C1,C2,Isc,Voc,vref,p

@njit(cache=True)
def duty_from_state(z,N,p):
    Ka=p[10]; T1=p[11]; T2=p[12]; Kpwm=p[13]
    d=np.empty(N)
    for n in range(N):
        k=5*n; u=Ka*z[k+3]+Ka*(T1-T2)*z[k+4]; x=Kpwm*u
        if x<1e-6: x=1e-6
        elif x>0.999999: x=0.999999
        d[n]=x
    return d

@njit(cache=True)
def rhs_gate(z,gates,C1,C2,Isc,Voc,vref,p):
    pv_mult,Np,Ns,Cpv,Ldco,Cdc,M,Rline,Lline,Cinv,Ka,T1,T2,Kpwm,Vs_mag,Rg,Xg,Vdc_ref,Vpcc_ref,Vdq_base,Kp1,Ki1,Kp2,Ki2,Kp3,Ki3,Kp4,Ki4,Kppll,Kipll,Ipk_base,Rf,Lf=p
    N=C1.size; dz=np.zeros_like(z)
    iDC=z[5*N]; vDC=z[5*N+1]; x1=z[5*N+2]; x2=z[5*N+3]; x3=z[5*N+4]; x4=z[5*N+5]; x5=z[5*N+6]; theta=z[5*N+7]; id_=z[5*N+8]; iq_=z[5*N+9]
    i_string=iDC/M; vprime=0.0
    for n in range(N):
        k=5*n; vpv=z[k]; iL=z[k+1]; vout=z[k+2]; xint=z[k+3]; xlag=z[k+4]
        expo=vpv/(Ns*C2[n]*Voc[n]); expo=min(50.0,max(-50.0,expo))
        ipv=pv_mult*Np*Isc[n]*(1-C1[n]*(math.exp(expo)-1))
        e=(vpv-vref[n])/Vdc_ref
        dz[k]=(ipv-iL)/Cpv
        if gates[n]>0.5:
            dz[k+1]=vpv/Ldco; dz[k+2]=-i_string/Cdc
        else:
            dz[k+1]=(vpv-vout)/Ldco; dz[k+2]=(iL-i_string)/Cdc
        dz[k+3]=e; dz[k+4]=(-xlag+e)/T2
        vprime += vout
    dz[5*N]=(vprime-vDC-2*Rline*iDC)/(2*Lline)
    cs=math.cos(theta); sn=math.sin(theta)
    vgd=Vs_mag*cs+Rg*id_-Xg*iq_; vgq=-Vs_mag*sn+Xg*id_+Rg*iq_; vmag=math.sqrt(vgd*vgd+vgq*vgq)
    Pac=1.5*(vgd*id_+vgq*iq_); iinv=Pac/max(vDC,1.0)
    dz[5*N+1]=(iDC-iinv)/Cinv
    evdc=(vDC-Vdc_ref)/Vdc_ref; dz[5*N+2]=evdc; idref=(Kp1*evdc+Ki1*x1)*Ipk_base
    ev=(vmag-Vpcc_ref*Vdq_base)/Vdq_base; dz[5*N+3]=ev; iqref=(Kp2*ev+Ki2*x2)*Ipk_base
    eid=(idref-id_)/Ipk_base; eiq=(iqref-iq_)/Ipk_base; dz[5*N+4]=eid; dz[5*N+5]=eiq
    ud=(Kp3*eid+Ki3*x3)*Vdq_base; uq=(Kp4*eiq+Ki4*x4)*Vdq_base
    dz[5*N+8]=(ud-Rf*id_)/Lf; dz[5*N+9]=(uq-Rf*iq_)/Lf
    epll=vgq/Vdq_base; dz[5*N+6]=epll; dz[5*N+7]=Kppll*epll+Kipll*x5
    return dz

@njit(cache=True)
def rk4_gate(z,h,g,C1,C2,Isc,Voc,vref,p):
    if h<=0: return z
    k1=rhs_gate(z,g,C1,C2,Isc,Voc,vref,p)
    k2=rhs_gate(z+0.5*h*k1,g,C1,C2,Isc,Voc,vref,p)
    k3=rhs_gate(z+0.5*h*k2,g,C1,C2,Isc,Voc,vref,p)
    k4=rhs_gate(z+h*k3,g,C1,C2,Isc,Voc,vref,p)
    return z+(h/6)*(k1+2*k2+2*k3+k4)

@njit(cache=True)
def one_period(z,C1,C2,Isc,Voc,vref,p,subdiv=1):
    N=C1.size; d=duty_from_state(z,N,p)
    # collect and sort all edges plus 0,1; duplicates okay then skipped
    edges=np.empty(N+2); edges[0]=0.0; edges[N+1]=1.0
    for n in range(N): edges[n+1]=d[n]
    edges=np.sort(edges)
    for j in range(edges.size-1):
        a=edges[j]; b=edges[j+1]
        if b-a<1e-12: continue
        mid=0.5*(a+b); g=np.empty(N)
        for n in range(N): g[n]=1.0 if mid<d[n] else 0.0
        h=(b-a)*TSW/subdiv
        for q in range(subdiv): z=rk4_gate(z,h,g,C1,C2,Isc,Voc,vref,p)
    return z

@njit(cache=True)
def simulate(z0,C1,C2,Isc,Voc,vref,p,tstop,save_dt=1e-3,subdiv=1):
    nper=int(round(tstop*FSW)); save_per=max(1,int(round(save_dt*FSW))); ns=nper//save_per+1
    out=np.empty((ns,z0.size)); tt=np.empty(ns); z=z0.copy(); out[0]=z; tt[0]=0.; j=1
    for k in range(nper):
        z=one_period(z,C1,C2,Isc,Voc,vref,p,subdiv)
        if (k+1)%save_per==0:
            out[j]=z; tt[j]=(k+1)*TSW; j+=1
    return tt[:j],out[:j]

if __name__=='__main__':
    for xg in [0.62,0.663515]:
      c,z,C1,C2,Isc,Voc,vref,p=prepare_case(xg,0.2)
      t0=time.time(); tt,yy=simulate(z,C1,C2,Isc,Voc,vref,p,0.01,1e-4,2); print('compile/run',xg,time.time()-t0,np.isfinite(yy).all())
      for sd in [1,2,4]:
       t0=time.time(); tt,yy=simulate(z,C1,C2,Isc,Voc,vref,p,2.0,1e-3,sd); good=np.isfinite(yy).all(); print(' xg',xg,'subdiv',sd,'time',time.time()-t0,'good',good,'vdc',np.nanmin(yy[:,51]),np.nanmax(yy[:,51]),yy[-1,51])
