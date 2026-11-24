"""Wave-domain Redheffer cascade of exact LC delays and passive static R/G.

All scattering matrices use the *same power-normalized*, reciprocal
high-frequency LC modal reference. Static loss sections are memoryless.
The LC sections transmit each mode with exactly its physical delay,
so there is no premature wavefront in any finite split network.
Cascaded passivity is algebraic; no large full-line ABCD matrices.
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from modal_propagation import LosslessModal
from split_delay import examples, telegrapher

def modal_transforms(modal):
    roots=np.sqrt(modal.delay_per_length)
    pvoltage=modal.voltage*roots[None,:]
    pcurrent=modal.current/roots[None,:]
    # Power normalization makes each port's energy flux equal to
    # ||incoming||² - ||outgoing||².
    assert np.linalg.norm(pvoltage.T@pcurrent-np.eye(len(roots)))<1e-8
    return pvoltage,pcurrent

def scattering_from_transfer(transfer,modal):
    n=len(modal.delay_per_length)
    pv,pi=modal_transforms(modal)
    t00=transfer[:n,:n]
    t01=transfer[:n,n:]
    t10=transfer[n:,:n]
    t11=transfer[n:,n:]
    a=pi.T@t00@pv
    b=pi.T@t01@pi
    c=pv.T@t10@pv
    d=pv.T@t11@pi
    operator=a-b-c+d
    left=np.linalg.solve(operator,c+d-a-b)
    cross=np.linalg.solve(operator,2*np.eye(n))
    right=(a+b)+(a-b)@left
    far=(a-b)@cross-np.eye(n)
    return np.block([[left,cross],[right,far]])

def pure_delay(modal,length,s):
    n=len(modal.delay_per_length)
    phase=np.diag(np.exp(-s*length*modal.delay_per_length))
    null=np.zeros_like(phase)
    return np.block([[null,phase],[phase,null]])

def star(left,right,n):
    a11,a12,a21,a22=left[:n,:n],left[:n,n:],left[n:,:n],left[n:,n:]
    b11,b12,b21,b22=right[:n,:n],right[:n,n:],right[n:,:n],right[n:,n:]
    m=np.eye(n)-a22@b11
    x=np.linalg.solve(m,a21)
    y=np.linalg.solve(m,a22@b12)
    return np.block([
        [a11+a12@b11@x, a12@(b11@y+b12)],
        [b21@x, b22+b21@y]
    ])

def split_scattering(model,length,s,modal,sections):
    n=len(model.r)
    dx=length/sections
    zref=1. # physical, unscaled voltage/current generator
    loss_generator=telegrapher(model,0.,zref,"static")
    static_transfer=expm(loss_generator*dx)
    static_scatter=scattering_from_transfer(static_transfer,modal)
    total=pure_delay(modal,dx/2,s)
    for i in range(sections):
        total=star(total,static_scatter,n)
        total=star(total,pure_delay(modal,dx if i+1<sections else dx/2,s),n)
    return total

def validate():
    report={}
    for name in ["scalar_lossless","scalar_zero_G","noncommuting_three","noncommuting_eight"]:
        model,length=examples[name]
        modal=LosslessModal(model.l,model.c)
        n=len(model.r)
        worst_relative=0.
        worst_passivity=0.
        worst_reciprocity=0.
        for sections in [1,2,4,8,16]:
            for omega in [1e3,1e5,1e7,1e9,1e10]:
                s=1j*omega
                scatter=split_scattering(model,length,s,modal,sections)
                # This full-line transfer is well-conditioned for the
                # selected moderate-length verification cases.
                dx=length/sections
                half=modal.transfer(s,dx/2,1.)
                static=expm(telegrapher(model,0.,1.,"static")*dx)
                direct=np.linalg.matrix_power(half@static@half,sections)
                expected=scattering_from_transfer(direct,modal)
                error=np.linalg.norm(scatter-expected)/max(np.linalg.norm(expected),1e-16)
                worst_relative=max(worst_relative,float(error))
                h=(scatter+scatter.T)/2
                worst_reciprocity=max(worst_reciprocity,float(np.linalg.norm(scatter-scatter.T)))
                largest=np.linalg.eigvalsh(scatter.conj().T@scatter).max()
                worst_passivity=max(worst_passivity,float(largest-1.))
        report[name]={
            "stages_checked":[1,2,4,8,16],
            "max_scatter_vs_transfer_relative":worst_relative,
            "max_reciprocity_error":worst_reciprocity,
            "max_sdagger_s_eigen_minus_one":worst_passivity,
            "minimum_physical_arrival_seconds":modal.min_delay_per_length*length
        }
        print(json.dumps({name:report[name]},indent=2),flush=True)

    long_model,_=examples["scalar_lossy"]
    # Moderately high but finite static loss over length 1000 m.
    # Static R=3,G=.01 gives gamma=sqrt(.03)=0.173/m => gamma L=173.
    # The full ABCD still fits, but a much longer case crosses exp overflow.
    long_mode=LosslessModal(long_model.l,long_model.c)
    factor=1000.
    extremely_long=split_scattering(long_model,factor,0.,long_mode,64)
    reference_z=np.sqrt(3./.01)
    characteristic=np.sqrt(1e-6/1e-8)
    rho=(reference_z-characteristic)/(reference_z+characteristic)
    assert np.all(np.isfinite(extremely_long))
    assert abs(extremely_long[0,0]-rho) < 1e-8,(extremely_long[0,0],rho)
    assert abs(extremely_long[0,1])<1e-10
    report["long_lossy_dc"]={
        "length":factor,
        "expected_reflection":float(rho),
        "computed_reflection":float(extremely_long[0,0].real),
        "computed_transmission":float(abs(extremely_long[0,1])),
        "finite":bool(np.all(np.isfinite(extremely_long))),
    }
    print(json.dumps({"long_lossy_dc":report["long_lossy_dc"]},indent=2),flush=True)
    # High-attenuation regression: gamma*length=1000. The full 2N
    # transfer contains exp(+1000) and is not representable in f64,
    # yet reciprocal input reflection and transmitted waves are finite.
    from shared_pole_core import Material
    highloss = Material(
        np.array([[1.]]), np.array([[1e-6]]),
        np.array([[1.]]), np.array([[1e-8]])
    )
    mode=LosslessModal(highloss.l,highloss.c)
    extreme=split_scattering(highloss,1000.,0.,mode,64)
    rho=(1.-10.)/(1.+10.)
    assert np.all(np.isfinite(extreme))
    assert abs(extreme[0,0]-rho)<1e-8, (extreme[0,0],rho)
    assert abs(extreme[0,1])<1e-200
    assert np.linalg.eigvalsh(extreme.T@extreme).max()<=1.+1e-12
    report["gamma_1000_DC"]={
        "loss_exponent":1000.,
        "series_resistance":1.,
        "shunt_conductance":1.,
        "length":1000.,
        "expected_reflection":float(rho),
        "computed_reflection":float(extreme[0,0].real),
        "transmission":float(abs(extreme[0,1])),
        "finite":True
    }
    print(json.dumps({"gamma_1000_DC":report["gamma_1000_DC"]},indent=2),flush=True)
    Path(__file__).with_name("scattering_split.json").write_text(json.dumps(report,indent=2))

if __name__=="__main__":
    validate()
