"""Independent check of direct expm transfer vs physical passive 2-port.

The source CPL branch uses expm of the physical 2N transfer generator and
inverts its B block.  A high-loss finite cable has a finite 2-port
admittance but exponentially unbounded transfer blocks.
"""
import math
import json
import warnings
import numpy as np
from scipy.linalg import expm
cases = [
    {"name":"moderate_attn","R":1.,"L":1e-6,"G":1.,"C":1e-9,"length":2.},
    {"name":"long_lossy_valid","R":1.,"L":1e-6,"G":1.,"C":1e-9,"length":1000.},
    {"name":"lossless_dc","R":0.,"L":1e-6,"G":0.,"C":1e-9,"length":1.},
]
outputs=[]
for item in cases:
    # Physical s=0; first-order telegrapher transfer T=exp([[0,-R],[-G,0]]*length).
    Z,Y = item["R"], item["G"]
    length=item["length"]
    scale=math.sqrt(Z/Y) if Z>0 and Y>0 else 1.
    a=np.array([[0., -Z*length/scale],[-Y*length*scale, 0.]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t=expm(a)
    finite=bool(np.isfinite(t).all())
    if Z>0 and Y>0:
        gamma=math.sqrt(Z*Y)*length
        e=math.exp(-gamma)
        den=-math.expm1(-2*gamma)
        yc=math.sqrt(Y/Z)
        input_y=yc*(1+e*e)/den
        cross_y=yc*(2*e)/den
    else:
        input_y=None
        cross_y=None
    outputs.append({
        "name":item["name"],"positive_physical_R_G":bool(Z>0 and Y>0),
        "generator_eigenvalue_magnitude":math.sqrt(Z*Y)*length,
        "direct_expm_all_finite":finite,
        "physical_admittance_y11":input_y,
        "physical_admittance_y12":-cross_y if cross_y is not None else None,
        "direct_transfer_B":float(t[0,1]) if finite else None})
print(json.dumps(outputs,indent=2))
from pathlib import Path
Path(__file__).with_name("transfer_stability.json").write_text(json.dumps(outputs,indent=2))
