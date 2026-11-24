"""Lossless matched pi-ladder spatial convergence independently of the simulator.

We use *exact time integration* via sparse matrix exponential, so any
pre-arrival signal is entirely the finite-volume spatial approximation,
not the SPICE integration timestep/integrator. The exact distributed
line has a strict wavefront at tau=length sqrt(L C).
"""
import json
import time
import warnings
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply

R0=50.
L_PER_LENGTH=250e-9
C_PER_LENGTH=100e-12
LENGTH=.2
TAU=LENGTH*np.sqrt(L_PER_LENGTH*C_PER_LENGTH)
VS=1.

def matrix(n):
    dx=LENGTH/n
    cnode=np.full(n+1,C_PER_LENGTH*dx)
    cnode[0]=cnode[-1]=C_PER_LENGTH*dx/2
    inductance=L_PER_LENGTH*dx
    # Variables V_0..V_n, I_0..I_{n-1}, and affine 1
    dimension=2*n+2
    mat=lil_matrix((dimension,dimension),dtype=float)
    for j in range(n+1):
        cap=cnode[j]
        if j==0:
            mat[j,j] += -1/(R0*cap)
            mat[j,2*n+1] += VS/(R0*cap)
            mat[j,n+1] += -1/cap
        elif j==n:
            mat[j,j] += -1/(R0*cap)
            mat[j,n+j] += 1/cap
        else:
            mat[j,n+j] += 1/cap
            mat[j,n+1+j] += -1/cap
    for j in range(n):
        row=n+1+j
        mat[row,j]+=1/inductance
        mat[row,j+1]+=-1/inductance
    return csc_matrix(mat)

CHECKS=np.array([.05,.25,.5,.75,.8,.85,.9,.925,.95,.975,.99,.995,1.,1.005,1.01,1.025,1.05,1.075,1.1,1.2,1.5,2.])
def response(n,checks=CHECKS):
    a=matrix(n)*TAU
    initial=np.zeros(a.shape[0]); initial[-1]=1.
    out=[]
    for t in checks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=UserWarning)
            y=expm_multiply(a*float(t),initial)
        out.append(float(y[n]))
    return out

def main():
    models={}
    for n in (4,8,16,32,64,128,256):
        start=time.perf_counter()
        voltages=response(n)
        entry={
            "sections":n,
            "duration_seconds":time.perf_counter()-start,
            "time_over_tau":CHECKS.tolist(),
            "far_end_volts":voltages,
            "at_0_9_tau":voltages[list(CHECKS).index(.9)],
            "at_0_99_tau":voltages[list(CHECKS).index(.99)],
            "at_1_0_tau":voltages[list(CHECKS).index(1.)],
            "at_1_1_tau":voltages[list(CHECKS).index(1.1)],
            "after_2_tau":voltages[-1],
        }
        models[str(n)]=entry
        print(json.dumps({k:v for k,v in entry.items() if k not in ('time_over_tau','far_end_volts')},indent=2),flush=True)

    out={"source_voltage":VS,"termination_ohms":R0,"wavefront_seconds":TAU,
         "ideal_step_volts_before_tau":0.,
         "ideal_step_volts_after_tau":0.5,
         "models":models}
    Path(__file__).with_name("spatial_causality.json").write_text(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
