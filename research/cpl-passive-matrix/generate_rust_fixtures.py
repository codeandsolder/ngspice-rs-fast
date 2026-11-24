"""Deterministic Cpl WaveNetwork Python-to-Rust differential fixture generator.

All fixture coefficients and expected accepted wave values are generated
from the *independent* SciPy coupled R/L/G/C matrix-exponential static
scattering and the Python exact-delay accepted-history solver. Output
is ordinary Rust source in the same standalone research crate.
"""
from pathlib import Path
import json
import numpy as np
from time_scattering import PassiveDelayCascade
from shared_pole_core import Material

def array(values):
    return np.asarray(values,dtype=float)

CASES=[
    ("physical_scalar_RG", Material(
        array([[3.]]),array([[250e-9]]),
        array([[.01]]),array([[100e-12]])), .2,8),
    ("physical_noncommuting_three", Material(
        array([[1.5,.11,.08],[.11,2.,.07],[.08,.07,2.5]]),
        array([[4e-7,1e-8,5e-9],[1e-8,5e-7,8e-9],[5e-9,8e-9,6e-7]]),
        array([[.03,-.002,-.001],[-.002,.02,-.001],[-.001,-.001,.01]]),
        array([[1e-10,-1e-11,-5e-12],[-1e-11,1.5e-10,-1e-11],[-5e-12,-1e-11,2e-10]])
    ),.2,8),
]
def rust_float(value):
    import struct
    bits=struct.unpack("<Q",struct.pack("<d",float(value)))[0]
    hex_digits=f"{bits:016x}"
    groups="_".join(hex_digits[i:i+4] for i in range(0,16,4))
    return f"f64::from_bits(0x{groups})"

def rust_floats(values):
    return '&['+', '.join(rust_float(x) for x in values)+']'
def emit_fixture(name,model,length,stages):
    model.validate()
    network=PassiveDelayCascade(model,length,stages,
       near_front=np.zeros(len(model.r)),
       far_front=np.full(len(model.r),np.inf),
       prune_history=True)
    n=len(model.r)
    tau=length*network.modal.min_delay_per_length
    total=3*tau
    stepmax=float(np.min(network.half_delays))*.25
    steps=int(np.ceil(total/stepmax))
    times=np.linspace(0,total,steps+1)
    pick=sorted(set([1,steps//7,steps//3,steps//2,2*steps//3,5*steps//6,steps]))
    checkpoints=[]
    for index,t in enumerate(times[1:],start=1):
        incoming=np.zeros(n)
        incoming[0]=min(1.,t/(.2*tau))
        candidate=network.advance(float(t),incoming,np.zeros(n))
        if index in pick:
            checkpoints.append((index,candidate['outgoing_near'].copy(),candidate['outgoing_far'].copy()))
    data={
        "name":name,
        "modes":n,
        "stages":stages,
        "half_delays":network.half_delays.tolist(),
        "static_scattering":network.static.reshape(-1).tolist(),
        "first_arrival":float(tau),
        "final_time":float(total),
        "steps":steps,
        "checkpoints":[{"step":i,"reflected":near.tolist(),"transmitted":far.tolist()}
                       for i,near,far in checkpoints],
        "input_energy":float(network.energy_in),
        "output_energy":float(network.energy_out),
    }
    fragments=[
       'PhysicalFixture {',
       f'    name: "{name}",',
       f'    stages: {stages},',
       f'    modes: {n},',
       f'    half_delays: {rust_floats(network.half_delays)},',
       f'    static_scattering: {rust_floats(network.static.reshape(-1))},',
       f'    earliest: {rust_float(tau)},',
       f'    final_time: {rust_float(total)},',
       f'    steps: {steps},',
       '    checkpoints: &[',
    ]
    for i,near,far in checkpoints:
        fragments.append(f'        ({i}, {rust_floats(near)}, {rust_floats(far)}),')
    fragments+=['    ],','},']
    return data,'\n'.join(fragments)
def main():
    data={}
    expressions=[]
    for args in CASES:
        datum,text=emit_fixture(*args)
        data[datum['name']]=datum
        expressions.append(text)
        print(datum['name'],
            datum['modes'],datum['steps'],len(datum['checkpoints']),
            'energy_out/input',datum['output_energy']/datum['input_energy'])
    root=Path(__file__).with_name('rust').joinpath('src','wave_delay')
    rust='''
//! Physical reciprocal coupled-line differential fixture, generated
//! by an independent Python matrix-exponential reference solver.

struct PhysicalFixture {
    name: &'static str,
    stages: usize,
    modes: usize,
    half_delays: &'static [f64],
    static_scattering: &'static [f64],
    earliest: f64,
    final_time: f64,
    steps: u32,
    checkpoints: &'static [(u32, &'static [f64], &'static [f64])],
}

static CASES: &[PhysicalFixture] = &[
'''+ '\n'.join(expressions)+ '''
];

#[test]
fn physical_rlgc_scattering_matches_independent_python_accepted_history()
-> Result<(), &'static str> {
    use super::WaveNetwork;
    for fixture in CASES {
        let mut network=WaveNetwork::new(
            fixture.half_delays.to_vec(),
            fixture.stages,
            fixture.static_scattering.to_vec(),
            vec![0.;fixture.modes],
            vec![f64::INFINITY;fixture.modes],
            true,
        )?;
        let mut check=0;
        for step in 1..=fixture.steps {
            let time=fixture.final_time * (f64::from(step)/f64::from(fixture.steps));
            let mut incident=vec![0.;fixture.modes];
            incident[0]=(time/(0.2*fixture.earliest)).min(1.);
            let trial=network.trial(time,&incident,&vec![0.;fixture.modes])?;
            if check < fixture.checkpoints.len() {
                let (wanted_step,expected_near,expected_far)=fixture.checkpoints[check];
                if step==wanted_step {
                    for (&actual,&expected) in trial.reflected_near().iter().zip(expected_near) {
                        assert!((actual-expected).abs()<5e-10,
                            "{} near mismatch at step {}: {} vs {}",
                            fixture.name,step,actual,expected);
                    }
                    for (&actual,&expected) in trial.transmitted_far().iter().zip(expected_far) {
                        assert!((actual-expected).abs()<5e-10,
                            "{} far mismatch at step {}: {} vs {}",
                            fixture.name,step,actual,expected);
                    }
                    check+=1;
                }
            }
            network.accept(trial)?;
        }
        assert_eq!(check,fixture.checkpoints.len());
        let (input,output)=network.energies();
        assert!(output<=input, "{} violated power-wave energy passivity",fixture.name);
    }
    Ok(())
}
'''
    (root/'tests'/'physical_fixtures.rs').write_text(rust)
    Path(__file__).with_name('physical_fixtures.json').write_text(json.dumps(data,indent=2)+'\n')
if __name__=="__main__":
    main()
