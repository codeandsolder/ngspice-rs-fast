"""Independent Riccati equation residual validation for complex RLGC ground truth."""
import json
import numpy as np
from shared_pole_core import Material
def mat(x): return np.asarray(x,float)
models={
  "scalar": Material(mat([[3.]]),mat([[1e-6]]),mat([[.01]]),mat([[1e-8]])),
  "three": Material(
    mat([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
    mat([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
    mat([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
    mat([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]])
  )
}
rng=np.random.default_rng(20261124)
n=8
def make_spd(scale,off):
    a=rng.normal(0.,off,(n,n))
    return scale*(a@a.T+np.eye(n))
models["eight"]=Material(make_spd(2.,.12),make_spd(3e-7,.09),
   make_spd(.02,.15),make_spd(1e-10,.10))
results={}
for name,model in models.items():
    residual=[]
    symmetry=[]
    passivity=[]
    for omega in np.geomspace(1e1,1e13,50):
        Z=model.r+1j*omega*model.l
        Y=model.g+1j*omega*model.c
        H=model.characteristic(1j*omega)
        residual.append(float(np.linalg.norm(H@Z@H-Y)/np.linalg.norm(Y)))
        symmetry.append(float(np.linalg.norm(H-H.T)/np.linalg.norm(H)))
        passivity.append(float(np.linalg.eigvalsh((H+H.conj().T)/2).min()))
    results[name]={"n":len(model.r),
      "max_riccati_relative_residual":max(residual),
      "max_reciprocal_symmetry_error":max(symmetry),
      "min_hermitian_eig":min(passivity)}
    print(json.dumps({name:results[name]},indent=2))
from pathlib import Path
Path(__file__).with_name("validate_reference.json").write_text(json.dumps(results,indent=2))
