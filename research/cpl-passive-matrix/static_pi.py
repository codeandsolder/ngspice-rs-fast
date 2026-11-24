"""Passive coupled static-R/G pi stage without a matrix exponential or R inverse.

Eliminate the two 2N modal port scattering waves and N auxiliary
series currents by a (3N)x(3N) real linear solve. The shunts each
have G*dx/2, and the series branch carries R*dx, including singular
R=0/G=0. Compare with the exact matrix-exponential static stage.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from modal_propagation import LosslessModal
from scattering_split import scattering_from_transfer, modal_transforms, pure_delay, star
from split_delay import examples,telegrapher,OMEGAS

def coupled_static_pi(material,dx,modal):
    n=len(material.r)
    pv,pi=modal_transforms(modal)
    shunt=(material.g*dx/2)@pv
    series=material.r*dx
    size=3*n
    A=np.zeros((size,size))
    B=np.zeros((size,2*n))
    # current conservation at two R/G shunt nodes.
    A[:n,:n]=pi+shunt
    A[:n,2*n:]=np.eye(n)
    B[:n,:n]=pi-shunt
    A[n:2*n,n:2*n]=pi+shunt
    A[n:2*n,2*n:]=-np.eye(n)
    B[n:2*n,n:]=pi-shunt
    # physical branch-voltage equation: R*Iseries=Vleft-Vright.
    A[2*n:,:n]=-pv
    A[2*n:,n:2*n]=pv
    A[2*n:,2*n:]=series
    B[2*n:,:n]=pv
    B[2*n:,n:]=-pv
    return np.linalg.solve(A,B)[:2*n,:]

def main():
    cases={}
    for name,(mat,length) in examples.items():
        mat.validate()
        modal=LosslessModal(mat.l,mat.c)
        n=len(mat.r)
        stage=length/32
        spi=coupled_static_pi(mat,stage,modal)
        reference=scattering_from_transfer(
            expm(telegrapher(mat,0.,1.,"static")*stage),modal)
        err=np.linalg.norm(spi-reference)/max(np.linalg.norm(reference),1e-20)
        contraction=max(np.linalg.eigvalsh(spi.T@spi))
        symmetry=np.linalg.norm(spi-spi.T)
        assert err<0.005,(name,err)
        assert contraction<=1.+1e-10,(name,contraction)
        assert symmetry<1e-11,(name,symmetry)
        cases[name]={"static_relative_error":float(err),"smax_sq":float(contraction),
            "reciprocity_error":float(symmetry)}
        print(name,cases[name],flush=True)
    # Check complete cable using passive pi static stage and exact LC phases.
    full={}
    for name in ["scalar_lossy","scalar_zero_G","noncommuting_three","noncommuting_eight"]:
        mat,length=examples[name]
        modal=LosslessModal(mat.l,mat.c)
        n=len(mat.r)
        rows={}
        for sections in [8,16,32,64]:
            static=coupled_static_pi(mat,length/sections,modal)
            errors=[]
            for omega in OMEGAS:
                delay=pure_delay(modal,length/sections/2,1j*omega)
                total=delay
                for index in range(sections):
                    total=star(total,static,n)
                    total=star(total,pure_delay(modal,
                            length/sections if index+1<sections else length/sections/2,
                            1j*omega),n)
                exact=scattering_from_transfer(
                    expm(telegrapher(mat,1j*omega,1.)*length),modal)
                errors.append(float(np.linalg.norm(total-exact)/
                    max(np.linalg.norm(exact),1e-20)))
            rows[str(sections)]={"max_s_error":max(errors),
               "p95_s_error":float(np.quantile(errors,.95))}
        full[name]=rows
        print("FULL",name,rows,flush=True)
    Path(__file__).with_name("static_pi.json").write_text(
        json.dumps({"static":cases,"full":full},indent=2))


if __name__=="__main__":
    main()
