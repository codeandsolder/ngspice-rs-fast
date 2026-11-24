use ngspice_rs::analysis::{
    harmonic_balance::{HarmonicLattice, LatticeLimits, ToneAxis},
    periodic_small_signal::SidebandPlan,
};

#[test]
fn signed_sidebands_preserve_integer_pump_coordinates() {
    let lattice = HarmonicLattice::new(
        &[
            ToneAxis {
                frequency: 10.,
                order: 1,
            },
            ToneAxis {
                frequency: 23.,
                order: 1,
            },
        ],
        2,
        LatticeLimits::default(),
    )
    .unwrap();
    let plan = SidebandPlan::new(&lattice, 2., 9).unwrap();
    assert_eq!(plan.bands().len(), 9);
    assert_eq!(
        plan.bands()[plan.locate(&[-1, 1]).unwrap()].frequency(),
        15.
    );
    assert_eq!(
        plan.bands()[plan.locate(&[1, -1]).unwrap()].frequency(),
        -11.
    );
    assert_eq!(plan.bands()[plan.locate(&[0, 0]).unwrap()].frequency(), 2.);
    assert!(!plan.conjugate_overlap());
    assert!(
        SidebandPlan::new(&lattice, 0., 9)
            .unwrap()
            .conjugate_overlap()
    );
    assert!(
        SidebandPlan::new(&lattice, 5., 9)
            .unwrap()
            .conjugate_overlap()
    );
    assert!(SidebandPlan::new(&lattice, 1e30, 9).is_err());
    assert!(SidebandPlan::new(&lattice, 2., 8).is_err());
}
