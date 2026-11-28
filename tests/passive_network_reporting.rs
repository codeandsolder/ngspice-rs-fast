//! #228: physical current, reported current and excitation/control identities.
use std::{fmt::Write as _, path::Path};

use ngspice_rs::{
    analysis::{AnalysisContext, AnalysisInput, Plot, PointSink, RunConfig},
    devices::Circuit,
    netlist::{Parser, source::parse_deck_text},
    primitives::{AnalysisKind, Complex, SpiceResult},
};

fn circuit(text: &str) -> SpiceResult<Circuit> {
    let deck = parse_deck_text(Path::new("network.cir"), text);
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    config.circuit(&netlist)
}

fn close(got: Complex, want: Complex, relative: f64, absolute: f64) {
    let error = (got - want).magnitude();
    assert!(
        error <= absolute + relative * want.magnitude(),
        "{got:?} != {want:?}; error {error:e}"
    );
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, arguments: &[&str]) -> SpiceResult<Plot> {
    circuit.finalize()?;
    AnalysisInput::with_arguments(kind, arguments.iter().copied())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

/// LTspice 24.0.12, four multiplicities, each with resistor/capacitor/both
/// parallel parasitics, 3 AC frequencies, OP, 3 DC points. The reference
/// columns are terminal voltage, current through the driving resistor, and
/// reported storage current (real/imag pairs).
#[test]
fn independent_oracle_distinguishes_physical_and_reported_currents() -> SpiceResult<()> {
    let mut deck = String::from("storage reporting\n");
    for family in ['l', 'c'] {
        for (m_name, m) in [("1", 1.), ("2", 2.), ("3", 3.), ("p5", 0.5)] {
            for par in ["r", "c", "rc"] {
                let s = format!("{family}{m_name}{par}");
                let value = if family == 'l' { "3m" } else { "2u" };
                let rp = if par.contains('r') { "100" } else { "0" };
                let cp = if par.contains('c') { "1u" } else { "0" };
                let factor = if family == 'l' { m - 1. } else { 0. };
                writeln!(deck, "v{s} s{s} 0 dc 1 ac 1\nr{s} s{s} p{s} 2\n{family}{s} p{s} 0 {value} m={m} pn_rseries=5 pn_rparallel={rp} pn_cparallel={cp} pn_report_ac_rparallel_factor={factor}").unwrap();
            }
        }
    }
    deck.push_str(".end\n");
    let mut circuit = circuit(&deck)?;
    let ac = run(
        &mut circuit,
        AnalysisKind::Ac,
        &["lin", "3", "100", "10000"],
    )?;
    let op = run(&mut circuit, AnalysisKind::OperatingPoint, &[])?;
    let dc = run(
        &mut circuit,
        AnalysisKind::DcSweep,
        &["vl3rc", "0", "1", ".5"],
    )?;
    let mut count = 0;
    for line in include_str!("fixtures/passive_network/ltspice-24.0.12-current.tsv").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let phase = fields[0];
        let point = fields[1].parse::<usize>().unwrap();
        let s = fields[2];
        let name = format!("{}{}", &s[..1], s);
        let plot = match phase {
            "ac" => &ac,
            "op" => &op,
            "dc" => &dc,
            _ => panic!("unexpected reference phase {phase}"),
        };
        for (variable, offset) in [
            (format!("v(p{s})"), 3),
            (format!("i(v{name}#port)"), 5),
            (format!("i({name})"), 7),
        ] {
            let want = Complex::new(
                fields[offset].parse().unwrap(),
                fields[offset + 1].parse().unwrap(),
            );
            let got = plot.value(&variable, point).unwrap();
            let error = (got - want).magnitude();
            assert!(
                error <= 5e-9 + 1e-9 * want.magnitude(),
                "{phase} point {point}, {variable}: {got:?} != {want:?}, error {error:e}"
            );
            count += 1;
        }
    }
    assert_eq!(count, 504);
    Ok(())
}

#[test]
fn source_control_uses_external_branch_without_changing_excitation_identity() -> SpiceResult<()> {
    let mut circuit = circuit(
        "source network\nv1 out 0 1 ac 1 pn_rseries=2 pn_cseries_shunt=1u pn_control=1 pn_report_ac=1\nr1 out 0 3\nf1 0 f v1 1\nrf f 0 1\nb1 b 0 v=i(v1)\n.end\n",
    )?;
    let ac = run(
        &mut circuit,
        AnalysisKind::Ac,
        &["lin", "1", "10000", "10000"],
    )?;
    let voltage = Complex::new(0.602261102728891, 0.02998880634513171);
    let terminal = -voltage / Complex::real(3.);
    close(ac.value("v(out)", 0).unwrap(), voltage, 1e-10, 1e-12);
    for name in ["i(v1)", "i(vv1#port)", "v(f)", "v(b)"] {
        close(ac.value(name, 0).unwrap(), terminal, 1e-10, 1e-12);
    }
    let dc = run(&mut circuit, AnalysisKind::DcSweep, &["v1", "2", "2", "1"])?;
    close(
        dc.value("v(out)", 0).unwrap(),
        Complex::real(1.2),
        1e-10,
        1e-12,
    );
    Ok(())
}

#[test]
fn winding_control_is_not_the_reported_current() -> SpiceResult<()> {
    let mut circuit = circuit(
        "winding identity\nv1 drive 0 1 ac 1\nr1 drive p 2\nl1 p 0 3m m=3 pn_rseries=5 pn_rparallel=100 pn_cparallel=1u pn_report_ac_rparallel_factor=2 pn_control=0\nb1 b 0 v=i(l1)\n.end\n",
    )?;
    let ac = run(
        &mut circuit,
        AnalysisKind::Ac,
        &["lin", "1", "5050", "5050"],
    )?;
    let want = Complex::new(-0.001964082964207116, -0.02932248854810942);
    close(ac.value("v(b)", 0).unwrap(), want, 1e-10, 1e-12);
    close(ac.value("i(l1#intrinsic)", 0).unwrap(), want, 1e-10, 1e-12);
    close(
        ac.value("i(l1)", 0).unwrap(),
        Complex::new(0.09206209339041573, 0.04892418749764822),
        1e-10,
        1e-12,
    );
    Ok(())
}

#[test]
fn capacitor_primary_probe_keeps_differential_voltage_and_current_identities() -> SpiceResult<()> {
    let mut circuit = circuit(
        "capacitor identity\nv1 a 0 1 ac 1\nv2 b 0 0 ac .2\nc1 a b 2u pn_rseries=5 pn_rparallel=100 pn_cparallel=1u pn_report_ac=0 pn_report_ac_g=.03 pn_control=0\nb1 q 0 v=i(c1)\n.end\n",
    )?;
    let ac = run(
        &mut circuit,
        AnalysisKind::Ac,
        &["lin", "1", "5050", "5050"],
    )?;
    let primary = Complex::real(0.8) * Complex::new(0.018294118115039, 0.0576554322280287);
    close(ac.value("v(q)", 0).unwrap(), primary, 1e-10, 1e-12);
    close(
        ac.value("i(c1)", 0).unwrap(),
        primary + Complex::real(0.024),
        1e-10,
        1e-12,
    );
    Ok(())
}

#[derive(Default)]
struct CurrentOnly {
    selected: Vec<usize>,
    values: Vec<Complex>,
}
impl PointSink for CurrentOnly {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        self.selected = vec![header.variable_index("i(l1)").unwrap()];
        Ok(())
    }
    fn selected_columns(&self) -> Option<&[usize]> {
        Some(&self.selected)
    }
    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        assert_eq!(values.len(), 1);
        self.values.push(values[0]);
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        Ok(())
    }
}

#[test]
fn selected_report_columns_match_full_rows() -> SpiceResult<()> {
    let text = "selected reporting\nv1 drive 0 pulse(0 1 1u 0.1u 0.1u 20u 40u) ac 1\nr1 drive p 2\nl1 p 0 3m m=3 pn_rseries=5 pn_rparallel=100 pn_cparallel=1u pn_report_ac_rparallel_factor=2 pn_report_tran_g=.02\n.end\n";
    for (kind, args) in [
        (AnalysisKind::Ac, vec!["lin", "3", "100", "10000"]),
        (AnalysisKind::DcSweep, vec!["v1", "0", "1", ".5"]),
        (AnalysisKind::Transient, vec!["100n", "3u", "0", "100n"]),
    ] {
        let mut full = circuit(text)?;
        let mut selected = circuit(text)?;
        full.set_observations(&["i(l1)".to_owned()])?;
        selected.set_observations(&["i(l1)".to_owned()])?;
        let expected = run(&mut full, kind, &args)?;
        let plan = AnalysisInput::with_arguments(kind, args.iter().copied()).resolve()?;
        let mut sink = CurrentOnly::default();
        plan.run_into(&mut selected, &AnalysisContext::default(), &mut sink)?;
        assert_eq!(sink.values.len(), expected.point_count());
        for (index, got) in sink.values.iter().copied().enumerate() {
            close(got, expected.value("i(l1)", index).unwrap(), 1e-12, 1e-14);
        }
    }
    Ok(())
}

#[test]
fn transient_storage_network_matches_independent_switching_endpoint() -> SpiceResult<()> {
    let deck = "switching storage
.options reltol=1e-8 abstol=1e-12 vntol=1e-10
v1 s1 0 pulse(0 1 1u 0.1u 0.1u 10u 20u)
r1 s1 p1 2
l1 p1 0 3m pn_rseries=5 pn_rparallel=100 pn_cparallel=1u m=3 pn_report_ac_rparallel_factor=2
v2 s2 0 pulse(0 1 1u 0.1u 0.1u 10u 20u)
r2 s2 p2 2
c2 p2 0 2u pn_rseries=5 pn_rparallel=100 pn_cparallel=1u pn_lseries=3m pn_gseries_dc={1/(20*3)} pn_gseries_ac=.05 pn_gseries_tran={1/(20*3)} m=3 pn_report_tran=1
.tran 100n 30u 0 100n
.end
";
    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("network.cir"), deck))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let plot = config
        .plan_for(&netlist.analyses[0])?
        .run(&mut circuit, &config.context())?;
    let last = plot.point_count() - 1;
    close(
        plot.value("time", last).unwrap(),
        Complex::real(30e-6),
        1e-12,
        1e-14,
    );
    for (name, expected) in [
        ("v(p1)", 0.7584027154835773),
        ("i(l1)", 0.1207986422582508),
        ("v(p2)", 0.7260855908107419),
        ("i(c2)", 0.1369572045946370),
    ] {
        close(
            plot.value(name, last).unwrap(),
            Complex::real(expected),
            2e-5,
            2e-7,
        );
    }
    for (row, t) in plot.rows().enumerate() {
        for (report, port) in [("i(l1)", "i(vl1#port)"), ("i(c2)", "i(vc2#port)")] {
            assert_eq!(
                t[plot.variable_index(report).unwrap()],
                t[plot.variable_index(port).unwrap()],
                "row {row}"
            );
        }
    }
    Ok(())
}

#[test]
fn uic_damping_uses_transient_coefficients_without_bias_constraints() -> SpiceResult<()> {
    let text = "uic damping\nv1 s 0 1\nr1 s p 10\nc1 p 0 1u ic=0 pn_bias_tran=1000 pn_gterminal_dc=.01 pn_gterminal_tran=.03 pn_report_tran=1\n.end\n";
    let mut circuit = circuit(text)?;
    let mut input =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["100n", "100u", "0", "100n"]);
    input.uic = true;
    circuit.finalize()?;
    let plot = input
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())?;
    let last = plot.point_count() - 1;
    let equilibrium = 1. / 1.3;
    let expected = equilibrium * (1. - (-100e-6_f64 * 0.13 / 1e-6).exp());
    close(
        plot.value("v(p)", last).unwrap(),
        Complex::real(expected),
        2e-5,
        1e-7,
    );
    Ok(())
}

#[test]
fn series_inductance_shunt_supports_distinct_multiplicity_laws() -> SpiceResult<()> {
    let mut deck = String::from(
        "series shunt multiplicity\n.options reltol=1e-8 abstol=1e-12 vntol=1e-10\nv1 s 0 pulse(0 1 1u .1u .1u 10u 20u) ac 1\n",
    );
    for (index, m) in [1., 2., 3., 0.5].into_iter().enumerate() {
        let n = index + 1;
        writeln!(
            deck,
            "r{n} s p{n} 2\nc{n} p{n} 0 2u m={m} pn_rseries=5 pn_rparallel=100 pn_cparallel=1u pn_lseries=3m pn_gseries_ac=.05 pn_gseries_tran={{1/(20*{m})}}"
        )
        .unwrap();
    }
    deck.push_str(".ac lin 3 100 10000\n.tran 100n 30u 0 100n\n.end\n");
    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("network.cir"), &deck))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let ac = config
        .plan_for(&netlist.analyses[0])?
        .run(&mut circuit, &config.context())?;
    let tran = config
        .plan_for(&netlist.analyses[1])?
        .run(&mut circuit, &config.context())?;
    // Independent LTspice 24.0.12 reference: each column uses a different m.
    let reference = [
        [
            (0.9803629380716185, -0.003628936428701225),
            (0.961457406132857, -0.006980919486193901),
            (0.943244677387502, -0.01007904087107339),
            (0.9900875007544451, -0.001850624543540288),
        ],
        [
            (0.9128491594308511, -0.08062262101802195),
            (0.8296089265627619, -0.1356829812321496),
            (0.7537167234428188, -0.1721498485127054),
            (0.9562933410095975, -0.04399006895389805),
        ],
        [
            (0.896629327911288, -0.1175458510453922),
            (0.7922861612565576, -0.1909383740850804),
            (0.6975657707736985, -0.2333041598145967),
            (0.9495325892464954, -0.06510396310725285),
        ],
    ];
    for (point, row) in reference.into_iter().enumerate() {
        for (index, (re, im)) in row.into_iter().enumerate() {
            close(
                ac.value(&format!("v(p{})", index + 1), point).unwrap(),
                Complex::new(re, im),
                1e-10,
                1e-12,
            );
        }
    }
    let last = tran.point_count() - 1;
    for (index, expected) in [
        0.9153207353710472,
        0.8201352172432387,
        0.7260842981738106,
        0.9497307705099741,
    ]
    .into_iter()
    .enumerate()
    {
        close(
            tran.value(&format!("v(p{})", index + 1), last).unwrap(),
            Complex::real(expected),
            2e-5,
            2e-7,
        );
    }
    Ok(())
}

#[test]
fn physical_damping_ignores_temporary_junction_continuation() -> SpiceResult<()> {
    use ngspice_rs::{
        devices::{AnalysisMode, LoadRequest, ModelContext},
        maths::{SparseMatrix, Vector},
    };
    let mut circuit = circuit(
        "continuation damping\nv1 s 0 1\nr1 s p 10\nc1 p 0 1u pn_gterminal_dc_gmin=1e10\n.end\n",
    )?;
    let configured = ModelContext::default().with_gmin(1e-12);
    let continuation = ModelContext {
        gmin: 1e-6,
        ..configured
    };
    let normal = circuit.linear_system_with_context(&configured)?;
    let continued = circuit.linear_system_with_context(&continuation)?;
    assert_eq!(normal.a, continued.a);
    let n = circuit.unknown_count();
    let solution = Vector::zeros(n);
    let history = circuit.state_history()?;
    let stamp = |model_context| -> SpiceResult<SparseMatrix> {
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context,
            integration: None,
            history: &history,
            forcing: None,
        };
        let mut matrix = SparseMatrix::new(n, n);
        circuit.load(
            &request,
            &mut matrix,
            &mut Vector::zeros(n),
            &mut history.trial(),
        )?;
        Ok(matrix)
    };
    assert_eq!(stamp(&configured)?, stamp(&continuation)?);
    let overridden = circuit.linear_system_with_context(&configured.with_gmin(2e-12))?;
    assert_ne!(normal.a, overridden.a);
    Ok(())
}

#[test]
fn generated_probe_is_not_an_excitation_source() -> SpiceResult<()> {
    let mut circuit =
        circuit("probe identity\nv1 s 0 1\nr1 s p 2\nc1 p 0 2u pn_rseries=5\n.end\n")?;
    let system = circuit.linear_system()?;
    assert_eq!(system.sources.len(), 1);
    assert_eq!(system.sources[0].name.as_ref(), "v1");
    assert!(circuit.bind_source_target("vc1#port").is_err());
    assert!(
        run(
            &mut circuit,
            AnalysisKind::TransferFunction,
            &["v", "(", "p", ")", "vc1#port"],
        )
        .is_err()
    );
    let tf = run(
        &mut circuit,
        AnalysisKind::TransferFunction,
        &["v", "(", "p", ")", "v1"],
    )?;
    assert_eq!(tf.point_count(), 1);
    Ok(())
}

#[test]
fn series_inductor_shunt_noise_uses_frequency_domain_multiplicity() -> SpiceResult<()> {
    let mut circuit = circuit(
        "series shunt noise\nv1 s 0 ac 1\nr1 s p 2\n\
         c1 p 0 2u m=3 pn_rseries=5 pn_rparallel=100 pn_cparallel=1u \
         pn_lseries=3m pn_gseries_ac=.05 pn_gseries_tran={1/(20*3)} pn_gseries_noise=1\n.end\n",
    )?;
    let noise = run(
        &mut circuit,
        AnalysisKind::Noise,
        &["v(p)", "v1", "lin", "3", "100", "10000"],
    )?;
    for (name, reference) in [
        (
            "onoise_spectrum",
            [
                1.768349869679772e-10,
                1.580738710815810e-10,
                1.520717676270345e-10,
            ],
        ),
        (
            "inoise_spectrum",
            [
                1.874645005376138e-10,
                2.044605719103892e-10,
                2.067466055045742e-10,
            ],
        ),
    ] {
        for (index, expected) in reference.into_iter().enumerate() {
            close(
                noise.value(name, index).unwrap(),
                Complex::real(expected),
                3e-4,
                0.,
            );
        }
    }
    Ok(())
}
