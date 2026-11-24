"""Prototype error-gated selection of exact-delay passive split sections.

Do not Richardson-extrapolate the *matrices*: extrapolation can destroy
passivity. Compare two nested discretizations using contractive
power-normalized scattering S, then select the finer physical network.

The sample grid is finite. This is research, not a certified global
transient error bound; evaluate independent holdout frequencies and use
timestep/bandwidth-aware checking before production.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm

from split_delay import examples, OMEGAS, telegrapher, line_y
from modal_propagation import LosslessModal
from scattering_split import split_scattering, scattering_from_transfer

TRAIN=OMEGAS[::2]
HOLDOUT=OMEGAS
LIMIT=512
TARGET=1e-3

def select(model,length,modal,target=TARGET):
    # R=G=0 is exactly represented by one pure modal LC delay.
    if not np.any(model.r) and not np.any(model.g):
        return 1,True,[{"analytic_lossless_exact":True}]
    n=1
    history=[]
    while n<LIMIT:
        doubled=n*2
        worst=0.
        for omega in TRAIN:
            s=1j*omega
            coarse=split_scattering(model,length,s,modal,n)
            fine=split_scattering(model,length,s,modal,doubled)
            diff=np.linalg.norm(fine-coarse)/max(np.linalg.norm(fine),1e-14)
            worst=max(worst,float(diff))
        history.append({"coarse":n,"fine":doubled,"max_scatter_difference":worst})
        if worst<target:
            return doubled,True,history
        n=doubled
    return n,False,history

def main():
    results={}
    for name in ["scalar_lossless","scalar_zero_G","scalar_lossy",
                 "noncommuting_three","noncommuting_eight"]:
        model,length=examples[name]
        modal=LosslessModal(model.l,model.c)
        n=len(model.l)
        selected,converged,history=select(model,length,modal)
        errors_s=[]
        errors_y=[]
        passivity=[]
        for omega in HOLDOUT:
            s=1j*omega
            actual_transfer=expm(telegrapher(model,s,1.)*length)
            exact=scattering_from_transfer(actual_transfer,modal)
            fitted=split_scattering(model,length,s,modal,selected)
            errors_s.append(float(np.linalg.norm(fitted-exact)/max(np.linalg.norm(exact),1e-14)))
            passivity.append(float(np.linalg.eigvalsh(fitted.conj().T@fitted).max()-1.))
        results[name]={
            "selected_cells":selected,
            "converged_on_training_grid":converged,
            "training_estimates":history,
            "independent_max_scatter_error":max(errors_s),
            "max_scattering_passivity_eigen_excess":max(passivity),
            "first_arrival_lower_bound_seconds":float(modal.min_delay_per_length*length),
        }
        print(json.dumps({name:results[name]},indent=2),flush=True)
    Path(__file__).with_name("adaptive_split.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":
    main()
