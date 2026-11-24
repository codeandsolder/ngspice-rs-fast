"""π-ladder wavefront pre-response for finite-rise linear voltage ramps.

The linear-ramp response is the moving average of the step response
over the rise interval. Obtain its integrated step response exactly
(up to the sparse matrix exponential), not by timestep quadrature.
"""
import json, warnings
from pathlib import Path
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import expm_multiply
from spatial_causality import matrix, TAU

TIMES=[.5,.8,.9,.95,.97,.98,.99,.995,1.,1.025,1.05,1.1,1.2,1.4,2.]
RISES=[.02,.05,.1,.2,.5]
SECTIONS=[16,32,64,128,256]
out={}
for n in SECTIONS:
    a=matrix(n)*TAU
    d=a.shape[0]
    augmented=lil_matrix((d+1,d+1))
    augmented[:d,:d]=a
    augmented[d,n]=1.
    augmented=csc_matrix(augmented)
    init=np.zeros(d+1);init[d-1]=1.
    inputs=set(float(t) for t in TIMES)
    inputs.update(max(0.,float(t)-rise) for t in TIMES for rise in RISES)
    def state(t):
        if t<=0.:
            return 0.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=UserWarning)
            return float(expm_multiply(augmented*t,init)[d])
    integrated={t:state(t) for t in sorted(inputs)}
    record={}
    for rise in RISES:
        values=[]
        errors=[]
        for t in TIMES:
            actual=(integrated[float(t)]-integrated[max(0.,float(t)-rise)])/rise
            ideal=0.5*min(max((t-1.)/rise,0.),1.)
            values.append(actual)
            errors.append(actual-ideal)
        before=[v for t,v in zip(TIMES,values) if t<1.]
        record[str(rise)]={
            "rise_fraction_tau":rise,
            "max_pre_delay_volts":max(before),
            "at_0_99_tau":values[TIMES.index(.99)],
            "at_1_0_tau":values[TIMES.index(1.)],
            "max_abs_error_volts":max(abs(v) for v in errors),
            "responses":values,
        }
    out[str(n)]=record
    print(json.dumps({"sections":n,
        "at_0_99_tau":{str(r):record[str(r)]["at_0_99_tau"] for r in RISES},
        "max_pre_delay":{str(r):record[str(r)]["max_pre_delay_volts"] for r in RISES}
    },indent=2),flush=True)
report={"continuous_delay_tau_seconds":TAU,
    "ideal_before_tau":0.,
    "time_over_tau":TIMES,
    "rise_fractions_of_tau":RISES,
    "sections":out}
Path(__file__).with_name("spatial_rise_time.json").write_text(json.dumps(report,indent=2))
