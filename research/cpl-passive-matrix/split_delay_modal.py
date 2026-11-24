"""Efficient Strang-split exact-delay coupled RLGC frequency benchmark.

Unlike the 2N matrix exponential prototype, lossless transmission uses
one SPD modal eigensystem and per-mode sin/cos (or Laplace cosh/sinh).
Static R/G two-ports are computed only once per section count.

This is a frequency-domain passivity/accuracy study, not yet a history-
aware transient state-machine. No sample clipping or artificial delay.
"""
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import expm

from modal_propagation import LosslessModal
from split_delay import examples, OMEGAS, telegrapher, line_y

SECTIONS=[1,2,4,8,16,32,64,128]

def run():
    results={}
    for name,(material,length) in examples.items():
        n=len(material.r)
        mode=LosslessModal(material.l,material.c)
        zref=np.sqrt(np.linalg.norm(material.l,np.inf)/np.linalg.norm(material.c,np.inf))
        references=[]
        for omega in OMEGAS:
            s=1j*omega
            full=expm(telegrapher(material,s,zref)*length)
            y=line_y(full,zref,n)
            if y is not None and np.isfinite(y).all():
                references.append((s,y))
        start=time.perf_counter()
        report={}
        for segments in SECTIONS:
            dx=length/segments
            passive_loss=expm(telegrapher(material,0.,zref,"static")*dx)
            errors=[]
            passivity=[]
            reciprocity=[]
            for s,reference in references:
                half=mode.transfer(s,dx/2,zref)
                stage=half@passive_loss@half
                transfer=np.linalg.matrix_power(stage,segments)
                y=line_y(transfer,zref,n)
                if y is None or not np.isfinite(y).all():
                    raise ValueError(f"{name}: ill-conditioned split Y with {segments} stages")
                errors.append((float(abs(s.imag)),
                    float(np.linalg.norm(y-reference)/np.linalg.norm(reference))))
                passivity.append(float(np.linalg.eigvalsh((y+y.conj().T)/2).min()))
                reciprocity.append(float(np.linalg.norm(y-y.T)/np.linalg.norm(y)))
            report[str(segments)]={
                "max_relative_Y_error":max(value for _,value in errors),
                "at_omega":max(errors,key=lambda pair:pair[1])[0],
                "p95_relative_Y_error":float(np.quantile([e for _,e in errors],.95)),
                "minimum_passivity_eigen":min(passivity),
                "maximum_reciprocity_error":max(reciprocity),
            }
        results[name]={
            "conductors":n,
            "first_arrival_lower_bound_seconds":mode.min_delay_per_length*length,
            "elapsed_seconds":time.perf_counter()-start,
            "fits":report,
        }
        print(json.dumps({name:results[name]},indent=2),flush=True)
    Path(__file__).with_name("split_delay_modal.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":
    run()
