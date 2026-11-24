"""Exact-front and passivity checks for the research delay scattering solver."""
import json
from pathlib import Path
import numpy as np

from shared_pole_core import Material
from scattering_split import split_scattering
from time_scattering import PassiveDelayCascade, run

def arr(v):
    return np.asarray(v,float)

def scalar(r,g):
    return Material(arr([[r]]),arr([[250e-9]]),arr([[g]]),arr([[100e-12]]))

def test_lossless_step():
    model=scalar(0.,0.)
    results={}
    for cells in [1,4,16,32]:
        cascade=PassiveDelayCascade(model,.2,cells,far_front=[float("inf")])
        minimum=float(np.min(cascade.modal.delay_per_length)*.2)
        ramp=.1*minimum
        t,source,back,far,net=run(model,.2,cells,
          lambda t: arr([min(1.,t/ramp)]),2*minimum,
          dt=float(np.min(cascade.half_delays))*.75)
        assert np.max(abs(back))<1e-12,(cells,"reflection")
        assert np.max(abs(far[t<minimum]))<1e-12,(cells,"prearrival")
        assert abs(far[-1,0]-1.)<1e-10,(cells,"postarrival")
        assert net.energy_out<=net.energy_in+1e-8
        results[str(cells)]={
          "max_prearrival":float(np.max(abs(far[t<minimum]))),
          "max_reflection":float(np.max(abs(back))),
          "transmission_at_2tau":float(far[-1,0]),
          "energy_in":net.energy_in,
          "energy_out":net.energy_out,
          "steps":len(t),
          "tau":minimum,
        }
        print("LOSSLESS_STRICT_WAVEFRONT",cells,results[str(cells)],flush=True)
    return results

def test_losses():
    models={
      "R_only":scalar(3.,0.),
      "RG":scalar(3.,.01),
      "3wire":Material(
        arr([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
        arr([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
        arr([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
        arr([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]])
      )
    }
    output={}
    for name,model in models.items():
        length=.2
        cells=8
        net=PassiveDelayCascade(model,length,cells)
        tau=length*net.modal.min_delay_per_length
        incoming=lambda t: np.r_[min(t/(.2*tau),1.),np.zeros(net.n-1)]
        t,source,back,far,runner=run(model,length,cells,incoming,3*tau)
        assert np.max(abs(far[t<tau]))<=1e-12,(name,"prearrival")
        assert runner.energy_out<=runner.energy_in+2e-8,(name,"energy")
        assert np.max(abs(far))<1.+1e-12
        output[name]={
          "exact_front_bound":tau,
          "maximum_prearrival":float(np.max(abs(far[t<tau]))),
          "near_reflection_peak":float(np.max(abs(back))),
          "far_transmission_peak":float(np.max(abs(far))),
          "near_crosstalk_peak":float(np.max(abs(back[:,1:]))) if net.n>1 else 0.,
          "far_crosstalk_peak":float(np.max(abs(far[:,1:]))) if net.n>1 else 0.,
          "energy_input":runner.energy_in,
          "energy_out":runner.energy_out,
        }
        print("LOSSY_EXACT_FRONT",name,output[name],flush=True)
    assert output['3wire']['far_crosstalk_peak']>1e-8
    return output

def test_stale_and_step_limit():
    model=scalar(1.,.001)
    net=PassiveDelayCascade(model,.2,4)
    half=np.min(net.half_delays)
    trial=net.trial(half*.5,arr([1.]),arr([0.]))
    other=net.trial(half*.5,arr([.5]),arr([0.]))
    foreign=PassiveDelayCascade(model,.2,4)
    try:
        foreign.accept(trial)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted trial from a different line")
    for bad_time in [0.,float("nan"),2.*half]:
        try:
            net.trial(bad_time,arr([1.]),arr([0.]))
        except ValueError:
            pass
        else:
            raise AssertionError("accepted invalid or future-reading trial step")
    net.accept(trial)
    try:
        net.accept(other)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted stale trial from same old state")
    # A failure in the middle of a proposed model update must not
    # append any of the valid preceding cells' samples.
    next_time=1.5*half
    corrupted=net.trial(next_time,arr([1.]),arr([0.]))
    corrupted["static_out_far"][-1][0]=float("nan")
    snapshot=(net.time,net.revision,net.history_sample_count(),
              net.energy_in,net.energy_out)
    try:
        net.accept(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted NaN static block as partial transaction")
    assert (net.time,net.revision,net.history_sample_count(),
            net.energy_in,net.energy_out)==snapshot
    # A new valid trial with the same step is still accepted.
    fixed=net.trial(next_time,arr([1.]),arr([0.]))
    net.accept(fixed)
    assert net.revision==2
    return True

if __name__=="__main__":
    print(json.dumps({
       "lossless":test_lossless_step(),
       "lossy":test_losses(),
       "rollback":test_stale_and_step_limit(),
    },indent=2))
