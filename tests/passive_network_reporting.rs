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
