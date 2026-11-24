"""Finite-band rational approximation of passive sqrt(s/(s+a)) zero-G cable."""
import json, time
import numpy as np
from scipy.optimize import nnls

relax=3e6
low, high=3e2,3e10
def exact(s):
    if np.isinf(s):
        return .1
    return .1*np.sqrt(s/(s+relax))
points=np.geomspace(low,high,140)
training=np.r_[0.,points,np.inf]
imag=1j*points
def basis(s,poles):
    if np.isinf(s):
        return np.r_[np.ones(len(poles)),np.zeros(len(poles))]
    return np.r_[s/(s+poles), poles/(s+poles)]
report=[]
for mult_low, mult_hi, count in [(1e-2,1e2,16),(1e-4,1e4,24),(1e-6,1e4,32),(1e-8,1e6,48),(1e-4,1e4,64)]:
    start=time.perf_counter()
    poles=np.geomspace(low*mult_low,high*mult_hi,count)
    rows=[];values=[]
    for s in list(training)+list(imag):
        b=basis(s,poles)
        val=exact(s)
        w=10. if s==0. or np.isinf(s) else 1.
        rows.append(b.real*w)
        values.append(val.real*w)
        if np.iscomplexobj(s) and not np.isreal(s):
            rows.append(b.imag*w)
            values.append(val.imag*w)
    weights,residual=nnls(np.asarray(rows),np.asarray(values),maxiter=10000)
    real_err=[abs(weights@basis(s,poles)-exact(s))/max(abs(exact(s)),1e-12) for s in points]
    imag_err=[abs(weights@basis(1j*s,poles)-exact(1j*s))/max(abs(exact(1j*s)),1e-12) for s in points]
    case={"poles":count,"low_factor":mult_low,"high_factor":mult_hi,"active":int((weights>1e-12).sum()),
        "max_real":float(max(real_err)),"max_imag":float(max(imag_err)),
        "p95_imag":float(np.quantile(imag_err,.95)),"seconds":time.perf_counter()-start}
    report.append(case)
    print(json.dumps(case),flush=True)
from pathlib import Path
Path(__file__).with_name("fractional_scalar.json").write_text(json.dumps(report,indent=2))
