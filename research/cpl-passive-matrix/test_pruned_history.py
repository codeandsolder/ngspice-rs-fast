"""Pruning respects the exact delayed-history interpolation support."""
import json
from pathlib import Path
import numpy as np
from shared_pole_core import Material
from time_scattering import PassiveDelayCascade,run

def array(x):return np.asarray(x,float)
def fixtures():
    n=1
    yield "scalar_zero_losses",Material(array([[0.]]),array([[250e-9]]),
         array([[0.]]),array([[100e-12]])),.2,8
    yield "scalar_RG",Material(array([[3.]]),array([[250e-9]]),
         array([[.01]]),array([[100e-12]])),.2,8
    yield "coupled_three",Material(
        array([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
        array([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
        array([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
        array([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]]),
    ),.2,8

def execute():
    report={}
    for name,model,length,sections in fixtures():
        reference=PassiveDelayCascade(model,length,sections)
        tau=length*reference.modal.min_delay_per_length
        def source(t):
            v=np.zeros(reference.n)
            if t<tau*.13:
                v[0]=min(1.,t/(.06*tau))
            else:
                v[0]=np.sin(t/tau*2.5)
            return v
        dt=np.min(reference.half_delays)*.3
        t,a,b,c,full=run(model,length,sections,source,15*tau,
                         dt=dt,prune_history=False)
        t2,a2,b2,c2,short=run(model,length,sections,source,15*tau,
                          dt=dt,prune_history=True)
        assert np.array_equal(t,t2)
        diff=float(max(np.max(abs(b-b2)),np.max(abs(c-c2))))
        assert diff<2e-13,(name,diff)
        assert abs(short.energy_in-full.energy_in)<1e-18
        assert abs(short.energy_out-full.energy_out)<1e-18
        # Each trace retains only a few LC-delay windows, not the
        # full accepted timestep count.
        full_count=full.history_sample_count()
        kept=short.history_sample_count()
        assert kept<full_count/10,(name,kept,full_count)
        assert np.max(abs(c2[t<tau]))==0.
        report[name]={
            "accepted_steps":len(t),
            "unpruned_history_samples":full_count,
            "pruned_history_samples":kept,
            "sample_reduction":full_count/kept,
            "max_waveform_delta":diff,
            "strict_prearrival":True,
        }
        print(name,report[name],flush=True)
    return report
if __name__=="__main__":
    Path(__file__).with_name("pruned_history.json").write_text(
        json.dumps(execute(),indent=2)+"\n"
    )
