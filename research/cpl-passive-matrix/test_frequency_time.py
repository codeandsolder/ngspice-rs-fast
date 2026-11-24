"""Compare causal accepted-wave transient to independently composed AC scattering.

At a fixed angular frequency, a sine drive must converge to exactly
the matched-line scattering S21 phasor. This catches sign/transposition
mistakes not visible to pure wavefront tests.
"""
import json
from pathlib import Path
import numpy as np
from shared_pole_core import Material
from time_scattering import PassiveDelayCascade, run
from scattering_split import split_scattering

def mat(x): return np.asarray(x,float)

CASES={
 "scalar_lossless": (
    Material(mat([[0.]]),mat([[250e-9]]),mat([[0.]]),mat([[100e-12]])),0.2,8,2e9),
 "scalar_RG": (
    Material(mat([[3.]]),mat([[250e-9]]),mat([[.01]]),mat([[100e-12]])),0.2,8,2e9),
 "three_coupled": (
    Material(
        mat([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
        mat([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
        mat([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
        mat([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]])
    ),0.2,8,2e9),
}
results={}
for name,(model,length,sections,omega) in CASES.items():
    modal=PassiveDelayCascade(model,length,sections)
    tau_min=length*modal.modal.min_delay_per_length
    dt=float(min(np.min(modal.half_delays)/4.,2*np.pi/omega/80.))
    T=2*np.pi/omega
    initial=np.zeros(modal.n)
    def sine(t):
        v=initial.copy()
        v[0]=np.sin(omega*t)
        return v
    times,input,back,far,state=run(model,length,sections,sine,20*T,dt=dt)
    wave=far[-int(np.ceil(4*T/dt)):]
    ts=times[-len(wave):]
    basis=np.stack([np.sin(omega*ts),np.cos(omega*ts)],axis=1)
    predicted=np.linalg.lstsq(basis,wave,rcond=None)[0]
    phasor=predicted[0,:]+1j*predicted[1,:]
    y=split_scattering(model,length,1j*omega,modal.modal,sections)
    exact=y[modal.n:,:modal.n][:,0]
    error=np.linalg.norm(phasor-exact)/np.linalg.norm(exact)
    results[name]={
       "n":modal.n,
       "sections":sections,
       "omega":omega,
       "dt":dt,
       "steps":len(times),
       "max_relative_phasor_error":float(error),
       "phase_exact_rad":np.angle(exact).tolist(),
       "phase_transient_rad":np.angle(phasor).tolist(),
       "front_tau_min_seconds":float(tau_min),
       "energy_input":float(state.energy_in),
       "energy_output":float(state.energy_out)
    }
    print(json.dumps({name:results[name]},indent=2),flush=True)
    if error>0.03:
        raise AssertionError(f"{name} time-domain scatter differs from AC")
    if state.energy_out>state.energy_in+1e-12:
        raise AssertionError("passivity failed")
Path(__file__).with_name("frequency_time.json").write_text(json.dumps(results,indent=2))
