from __future__ import annotations
import math,numpy as np
import p001_mxn_reconstructed as pm


def coherent_state_from_full(y,c):
    a,g=pm.unpack(y,c)
    return np.r_[a[0].reshape(-1),g]

def coherent_rhs(z,c):
    """Exact nonlinear dynamics on the string-coherent invariant manifold.

    Each of M strings has the same N-position DCO state profile. The 5N DCO
    coordinates store one representative string; 10 global states are exact.
    """
    if not hasattr(c,'_pv_const'): pm._prepare_pv(c)
    N=c.N; a=z[:5*N].reshape(N,5); g=z[5*N:]
    vpv=a[:,0];iL=a[:,1];vout=a[:,2];xint=a[:,3];xlag=a[:,4]
    iDC,vDC,x1,x2,x3,x4,x5,theta,id_,iq_=g
    C1,C2,Isc,Voc=c._pv_const[0][0],c._pv_const[1][0],c._pv_const[2][0],c._pv_const[3][0]
    vref=c._vref[0]
    expo=vpv/(c.pv.Ns*C2*Voc)
    ipv=c.pv_multiplier*c.pv.Np*Isc*(1-C1*(np.exp(expo)-1))
    e=(vpv-vref)/c.DCO_voltage_error_base
    dxint=e;dxlag=(-xlag+e)/c.T2
    ucomp=c.Ka*xint+c.Ka*(c.T1-c.T2)*xlag
    d=np.clip(c.Kpwm*ucomp,1e-6,.999999)
    idc_in=(1-d)*iL
    dvpv=(ipv-iL)/c.Cpv
    diL=(vpv-(1-d)*vout)/c.L_dco
    i_string=iDC/c.M
    dvout=(idc_in-i_string)/c.Cdc
    vprime=float(np.sum(vout))
    diDC=(vprime-vDC-2*c.Rline*iDC)/(2*c.Lline)
    I=id_+1j*iq_;V=c.Vs_mag*np.exp(-1j*theta)+(c.Rg+1j*c.Xg)*I
    vgd=float(V.real);vgq=float(V.imag);vmag=abs(V)
    Pac=1.5*(vgd*id_+vgq*iq_);iinv=Pac/max(vDC,1.0);dvDC=(iDC-iinv)/c.Cinv
    evdc=(vDC-c.Vdc_ref)/c.Vdc_ref;dx1=evdc;idref=(c.Kp1*evdc+c.Ki1*x1)*c.Ipk_base
    ev=(vmag-c.Vpcc_ref_pu*c.Vdq_base)/c.Vdq_base;dx2=ev;iqref=(c.Kp2*ev+c.Ki2*x2)*c.Ipk_base
    eid=(idref-id_)/c.Ipk_base;eiq=(iqref-iq_)/c.Ipk_base;dx3=eid;dx4=eiq
    ud=(c.Kp3*eid+c.Ki3*x3)*c.Vdq_base;uq=(c.Kp4*eiq+c.Ki4*x4)*c.Vdq_base
    did=(ud-c.Rf*id_)/c.Lf;diq=(uq-c.Rf*iq_)/c.Lf
    epll=vgq/c.Vdq_base;dx5=epll;dtheta=c.Kppll*epll+c.Kipll*x5
    return np.r_[np.stack([dvpv,diL,dvout,dxint,dxlag],axis=1).reshape(-1),[diDC,dvDC,dx1,dx2,dx3,dx4,dx5,dtheta,did,diq]]

def numerical_jacobian(z,c,rel=2e-7,absstep=1e-9):
    z=np.asarray(z,float);n=len(z);J=np.empty((n,n))
    for j in range(n):
        h=max(absstep,rel*max(1.,abs(z[j])));zp=z.copy();zm=z.copy();zp[j]+=h;zm[j]-=h
        J[:,j]=(coherent_rhs(zp,c)-coherent_rhs(zm,c))/(2*h)
    return J
