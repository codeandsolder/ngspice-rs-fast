import json, math
from pathlib import Path
import numpy as np
from shared_pole_core import Material, fit_passive, evaluate

def mat(v):
    return np.asarray(v,dtype=float)
examples={}
examples["scalar_general"] = Material(
    mat([[3.]]),mat([[1e-6]]),mat([[.01]]),mat([[1e-8]]))
examples["scalar_zero_shunt_loss"] = Material(
    mat([[3.]]),mat([[1e-6]]),mat([[0.]]),mat([[1e-8]]))
examples["scalar_lossless"] = Material(
    mat([[0.]]),mat([[1e-6]]),mat([[0.]]),mat([[1e-8]]))
examples["coupled_pair"]=Material(
    mat([[2.,.2],[.2,3.]]),
    mat([[4e-7,.6e-7],[.6e-7,6e-7]]),
    mat([[.01,-.002],[-.002,.02]]),
    mat([[2e-10,-.02e-10],[-.02e-10,1e-10]]))
examples["coupled_three"]=Material(
    mat([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
    mat([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
    mat([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
    mat([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]]))
rng=np.random.default_rng(20261124)
n=8
def make_spd(scale, off):
    b=rng.normal(0.,off,(n,n))
    return scale*(b@b.T + np.eye(n))
examples["coupled_eight"]=Material(
    make_spd(2.,.12), make_spd(3e-7,.09),
    make_spd(.02,.15), make_spd(1e-10,.10))
results={}
for name,model in examples.items():
    print(f"Benchmarking {name}, n={len(model.r)}",flush=True)
    try:
        fitted,stats=fit_passive(model,
            pole_count=8 if len(model.r)>3 else 12,
            samples_per_axis=36 if len(model.r)>3 else 60,
            pair_atoms=len(model.r)<=3)
        accuracy=evaluate(model,fitted)
        results[name]={**stats,**accuracy}
        print(json.dumps({name:results[name]},indent=2),flush=True)
    except (ValueError,np.linalg.LinAlgError) as e:
        results[name]={"error":str(e)}
        print(f"Fit error: {e}",flush=True)
Path(__file__).with_name("matrix_benchmarks.json").write_text(json.dumps(results,indent=2))
