//! End-to-end model/instance temperature preparation benchmark for #194.
//! Parser and model resolution run before the timer; model bin interpolation,
//! model temperature, instance geometry, validation and material-mode branches
//! are counted together, as they run in a real physical circuit.
use std::{hint::black_box, path::Path, time::Instant};

use ngspice_rs::devices::{ModelResolver, bsim4};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

#[test]
#[ignore = "opt-in whole BSIM4 temperature setup benchmark on native and generic targets"]
fn benchmark_bsim4_model_bin_instance_preparation() {
    let models = [
        ("default_sparse", ".model nn nmos level=54 version=4.8.3"),
        (
            "dense_bin",
            ".model nn nmos level=54 version=4.8.3 binunit=1 \
             vth0=0.45 lvth0=1e-9 wvth0=2e-9 pvth0=1e-15 \
             a0=1.0 la0=2e-7 wa0=1e-7 pa0=1e-14 \
             nfactor=1.4 lnfactor=4e-8 wnfactor=5e-8 pnfactor=1e-15 \
             dsub=0.1 ldsub=2e-8 wdsub=3e-8 pdsub=1e-14 \
             cgsl=1e-10 lcgsl=1e-16 wcgsl=2e-16 pcgsl=1e-23 \
             xj=1e-8 lxj=2e-15 wxj=1e-15 pxj=1e-22",
        ),
        (
            "cancelled_threshold",
            ".model nn nmos level=54 version=4.8.3 binunit=1 \
             vth0=1e16 lvth0=0.5 wvth0=-1e16 pvth0=0",
        ),
    ];
    for (label, model) in models {
        let deck = format!(
            "BSIM4 model bin setup benchmark\nv1 d 0 1.2\nvg g 0 0.7\n\
             m1 d g 0 0 nn l=1u w=1u\n{model}\n.op\n.end\n"
        );
        let parsed = Parser::new()
            .parse_deck(&parse_deck_text(Path::new("bin-benchmark.cir"), &deck))
            .unwrap();
        let device = parsed.device("m1").unwrap();
        let resolver = ModelResolver::new(&parsed.models).unwrap();
        let model = bsim4::model_parameters(&resolver.resolve(device).unwrap().unwrap()).unwrap();
        let mut durations = Vec::new();
        let mut observed = None;
        for trial in 0..5 {
            let mut elapsed = 0_u128;
            let mut checksum = 0_u64;
            for _ in 0..96 {
                let started = Instant::now();
                let (members, asks, warnings) = black_box(
                    model
                        .temperature_values(device, black_box(27.), black_box(27.))
                        .unwrap_or_else(|error| panic!("{label}: {error}")),
                );
                elapsed += started.elapsed().as_nanos();
                for (_, value) in members.into_iter().chain(asks) {
                    checksum = checksum.wrapping_add(value.to_bits().rotate_left(7));
                }
                checksum = checksum.wrapping_add(u64::try_from(warnings.len()).unwrap());
            }
            if let Some(expected) = observed {
                assert_eq!(
                    expected, checksum,
                    "{label}: different physical preparation output"
                );
            }
            observed = Some(checksum);
            if trial != 0 {
                durations.push(elapsed);
            }
        }
        durations.sort_unstable();
        println!(
            "bsim4_model_bin_{label}: median_ns_per_instance={} checksum={:#x}",
            durations[durations.len() / 2] / 96,
            observed.unwrap(),
        );
    }
}
