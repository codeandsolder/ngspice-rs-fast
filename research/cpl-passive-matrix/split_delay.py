"""Passive physical RLGC splitting with exact lossless modal propagation.

For homogeneous telegrapher propagation U(x)=exp(A(s)x), split
A(s)=A0(s)+Al where A0(s) couples sL and sC (lossless, pure delayed
modal propagation), and Al couples R and G (static resistive two-port).

A symmetric Strang product of exact lossless delay segments and exact
lossy *static* transfer matrices is passive, reciprocal, and has the
correct earliest possible arrival (minimum full-length LC delay).
This is a frequency-domain research validation, NOT a time backend.
"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.linalg import expm

from shared_pole_core import Material

def telegrapher(mat,s,zref,kind="full"):
    n=len(mat.r)
    if kind=="lossless":
        z=s*mat.l
        y=s*mat.c
    elif kind=="static":
        z=mat.r
        y=mat.g
    else:
        z=mat.r+s*mat.l
        y=mat.g+s*mat.c
    out=np.zeros((2*n,2*n),dtype=complex)
    out[:n,n:]=-z/zref
    out[n:,:n]=-y*zref
    return out

def line_y(t,zref,n):
    a,b,c,d=t[:n,:n],t[:n,n:],t[n:,:n],t[n:,n:]
    try:
        ib=np.linalg.inv(b)
    except np.linalg.LinAlgError:
        return None
    top=np.c_[-ib@a,ib]
    bottom=np.c_[d@ib@a-c,-d@ib]
    return np.r_[top,bottom]/zref

def split_transfer(model,length,rate,zref,sections):
    dx=length/sections
    lossless=expm(telegrapher(model,rate,zref,"lossless")*(dx/2))
    resistive=expm(telegrapher(model,rate,zref,"static")*dx)
    part=lossless@resistive@lossless
    return np.linalg.matrix_power(part,sections)

def mk(a):return np.asarray(a,dtype=float)
examples={}
examples["scalar_lossy"]=(Material(mk([[3.]]),mk([[1e-6]]),mk([[.01]]),mk([[1e-8]])),1.)
examples["scalar_zero_G"]=(Material(mk([[3.]]),mk([[1e-6]]),mk([[0.]]),mk([[1e-8]])),1.)
examples["scalar_lossless"]=(Material(mk([[0.]]),mk([[1e-6]]),mk([[0.]]),mk([[1e-8]])),1.)
examples["noncommuting_three"]=(Material(
  mk([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
  mk([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
  mk([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
  mk([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]])),.5)
rng=np.random.default_rng(20261124)
n=8
def spd(scale,off):
    b=rng.normal(0.,off,(n,n))
    return scale*(b@b.T+np.eye(n))
examples["noncommuting_eight"]=(Material(spd(2.,.12),spd(3e-7,.09),
    spd(.02,.15),spd(1e-10,.10)),.2)

OMEGAS=np.geomspace(1e3,1e10,42)
SECTIONS=[1,2,4,8,16,32]
def main():
    results={}
    for name,(model,length) in examples.items():
        model.validate()
        n=len(model.r)
        zref=np.sqrt(np.linalg.norm(model.l,np.inf)/np.linalg.norm(model.c,np.inf))
        bounds=np.linalg.eigvalsh(np.linalg.cholesky(model.c).T@model.l@np.linalg.cholesky(model.c))
        # Exact causal minimum travel based on eig(C^(1/2) L C^(1/2)).
        delays=np.sqrt(bounds)*length
        result={
            "count":n,
            "line_length":length,
            "min_delay_seconds":float(min(delays)),
            "max_delay_seconds":float(max(delays)),
            "operator_split_strang_order":2,
            "n_sections":{},
        }
        tstart=time.perf_counter()
        cases=[]
        for omega in OMEGAS:
            rate=1j*omega
            t=expm(telegrapher(model,rate,zref)*length)
            y=line_y(t,zref,n)
            if y is not None and np.isfinite(y).all():
                cases.append((rate,y,t))
        for segments in SECTIONS:
            errors=[]
            passivity=[]
            symmetry=[]
            for rate,exact,expt in cases:
                ts=split_transfer(model,length,rate,zref,segments)
                fitted=line_y(ts,zref,n)
                if fitted is None or not np.isfinite(fitted).all():
                    errors.append(float("inf"))
                    continue
                errors.append(float(np.linalg.norm(fitted-exact)/np.linalg.norm(exact)))
                passivity.append(float(np.linalg.eigvalsh((fitted+fitted.conj().T)/2).min()))
                symmetry.append(float(np.linalg.norm(fitted-fitted.T)/np.linalg.norm(fitted)))
            result["n_sections"][str(segments)]={
                "max_relative_Y_error":max(errors),
                "p95_relative_Y_error":float(np.quantile(errors,.95)),
                "worst_passivity_eigen":min(passivity) if passivity else None,
                "worst_reciprocity_error":max(symmetry) if symmetry else None,
            }
        result["elapsed_s"]=time.perf_counter()-tstart
        results[name]=result
        print(json.dumps({name:result},indent=2),flush=True)
    Path(__file__).with_name("split_delay.json").write_text(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
