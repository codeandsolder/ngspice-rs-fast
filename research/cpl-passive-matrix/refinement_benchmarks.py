import json
from pathlib import Path
import numpy as np
from shared_pole_core import Material,fit_passive,evaluate
def m(a): return np.asarray(a,dtype=float)
rng=np.random.default_rng(20261124)
def make_spd(scale,off):
    n=8
    b=rng.normal(0.,off,(n,n))
    return scale*(b@b.T+np.eye(n))
models={
    "scalar_zero_g":Material(m([[3.]]),m([[1e-6]]),m([[0.]]),m([[1e-8]])),
    "three": Material(
        m([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
        m([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
        m([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
        m([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]])
    ),
    "eight":Material(make_spd(2.,.12),make_spd(3e-7,.09),
             make_spd(.02,.15),make_spd(1e-10,.10))
}
configs=[
    ("scalar_zero_g",32,60,True,1e8,1e8),
    ("scalar_zero_g",64,60,True,1e8,1e8),
    ("three",16,65,True,1e3,1e3),
    ("three",24,65,True,1e3,1e3),
    ("eight",12,55,False,1e3,1e3),
    ("eight",16,55,False,1e3,1e3),
]
results={}
for name,poles,samples,pairs,low,high in configs:
    key=f"{name}_p{poles}"
    print("Running",key,flush=True)
    try:
        fit, info=fit_passive(models[name],pole_count=poles,
            samples_per_axis=samples,pair_atoms=pairs,
            low_margin=low, high_margin=high)
        info.update(evaluate(models[name],fit))
        results[key]=info
        print(json.dumps({key:info},indent=2),flush=True)
    except Exception as e:
        results[key]={"error":repr(e)}
        print(f"Fit failed: {e!r}",flush=True)
    Path(__file__).with_name("refinement_benchmarks.json").write_text(json.dumps(results,indent=2))
