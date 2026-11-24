"""Independent physical SciPy-to-Rust static R/G pi network fixtures."""
import json
from pathlib import Path
import numpy as np
from modal_propagation import LosslessModal
from scattering_split import modal_transforms
from split_delay import examples
from static_pi import coupled_static_pi
from generate_rust_fixtures import rust_float, rust_floats

ROOT=Path(__file__).parent
CASES=[]
for name,(model,length) in examples.items():
    modal=LosslessModal(model.l,model.c)
    pv,pi=modal_transforms(modal)
    dx=length/32
    s=coupled_static_pi(model,dx,modal)
    entry={
        "name":name,
        "modes":len(model.r),
        "cell_length":dx,
        "r":model.r.reshape(-1).tolist(),
        "g":model.g.reshape(-1).tolist(),
        "pv":pv.reshape(-1).tolist(),
        "pi":pi.reshape(-1).tolist(),
        "s":s.reshape(-1).tolist(),
    }
    CASES.append(entry)
def rust_case(e):
    return '\n'.join([
      '    StaticFixture {',
      f'        name: "{e["name"]}",',
      f'        modes: {e["modes"]},',
      f'        cell_length: {rust_float(e["cell_length"])},',
      f'        resistance: {rust_floats(e["r"])},',
      f'        conductance: {rust_floats(e["g"])},',
      f'        voltage_basis: {rust_floats(e["pv"])},',
      f'        current_basis: {rust_floats(e["pi"])},',
      f'        expected_scattering: {rust_floats(e["s"])},',
      '    },'
    ])
source='''//! Independent cross-language physical symmetric static-R/G fixtures.
//! Generated deterministically by an independent Python modal solver.

use super::*;

struct StaticFixture {
    name: &'static str,
    modes: usize,
    cell_length: f64,
    resistance: &'static [f64],
    conductance: &'static [f64],
    voltage_basis: &'static [f64],
    current_basis: &'static [f64],
    expected_scattering: &'static [f64],
}

static CASES: &[StaticFixture] = &[
'''+ '\n'.join(rust_case(e) for e in CASES)+ '''
];

#[test]
fn full_coupled_static_pi_agrees_with_independent_scipy_matrix_solve()
-> Result<(), &'static str>{
    for fixture in CASES {
        let got=coupled_static_rg_pi(
            fixture.resistance, fixture.conductance,
            fixture.voltage_basis, fixture.current_basis,
            fixture.modes, fixture.cell_length,
        )?;
        assert_eq!(got.len(), fixture.expected_scattering.len());
        let largest=got.iter().zip(fixture.expected_scattering)
            .map(|(a,b)|(a-b).abs()).fold(0.0_f64,f64::max);
        assert!(largest<3e-10,
            "physical static scattering mismatch in {}: {}",fixture.name,largest);
        matrix_is_passive(&got,2*fixture.modes)?;
    }
    Ok(())
}

#[test]
fn zero_series_and_shunt_are_the_exact_modal_through_connection()
-> Result<(), &'static str>{
    let modes=3;
    let zeros=vec![0.;modes*modes];
    let identity=(0..modes*modes)
        .map(|k| if k/modes==k%modes {1.}else{0.})
        .collect::<Vec<_>>();
    let s=coupled_static_rg_pi(&zeros,&zeros,&identity,&identity,modes,1.)?;
    let width=2*modes;
    for row in 0..width{
        for col in 0..width{
            let expected=if row%modes==col%modes
                &&(row/modes)!=(col/modes){1.}else{0.};
            assert!((s[row*width+col]-expected).abs()<1e-13);
        }
    }
    Ok(())
}

#[test]
fn negative_resistance_or_conductance_is_not_passive(){
    let n=1;
    let voltage=[1.];
    let current=[1.];
    assert!(coupled_static_rg_pi(&[-1.],&[0.],&voltage,&current,n,1.).is_err());
    assert!(coupled_static_rg_pi(&[0.],&[-1.],&voltage,&current,n,1.).is_err());
}

#[test]
fn invalid_recipient_basis_and_geometry_are_rejected(){
    let basis=[1.];
    assert!(coupled_static_rg_pi(&[1.],&[1.],&basis,&[2.],1,1.).is_err());
    assert!(coupled_static_rg_pi(&[1.],&[1.],&basis,&basis,1,0.).is_err());
    assert!(coupled_static_rg_pi(&[1.],&[1.],&basis,&basis,1,f64::NAN).is_err());
    assert!(coupled_static_rg_pi(&[1.],&[f64::NAN],&basis,&basis,1,1.).is_err());
    assert!(coupled_static_rg_pi(&[1.],&[1.],&[],&basis,1,1.).is_err());
}
'''
dst=ROOT/'rust/src/wave_delay/static_pi'
dst.mkdir(parents=True,exist_ok=True)
(dst/'tests.rs').write_text(source)
(ROOT/'static_pi_fixtures.json').write_text(json.dumps(CASES,indent=2)+'\n')
print('generated',len(CASES),'scalar/3-wire/8-wire static matrix fixtures')
