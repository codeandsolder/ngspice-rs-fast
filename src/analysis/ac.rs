//! Linear complex small-signal analysis on the SPARSE 1.3 complex LU.
//!
//! The frequency grid ([`frequency_grid`]) and the linearized system
//! ([`SmallSignal`]) are shared with the `.sp` driver
//! (`crate::analysis::sparam`), because `span.c` steps frequencies and
//! prepares its operating point exactly as `acan.c` does.
use crate::analysis::frequency::{Advance, FrequencyGrid};
use crate::analysis::linear::{number, plot, unsupported};
use crate::analysis::{AnalysisPlan, Plot, PointSink};
use crate::devices::Circuit;
use crate::devices::linear::LinearSystem;
use crate::maths::Vector;
use crate::maths::complex::ComplexMatrix;
use crate::primitives::{Complex, Real, SpiceResult};

pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &crate::analysis::AnalysisContext,
    grid: &[Real],
    include_bias: bool,
    sink: &mut dyn PointSink,
) -> SpiceResult<()> {
    let settings = request.dc_settings().clone();
    let mut small_signal = SmallSignal::prepare(circuit, request, context, settings)?;
    for bias in small_signal.retained_bias(circuit, request, context, include_bias)? {
        sink.complete(bias)?;
    }
    let plot = plot(
        circuit,
        "ac1",
        "AC Analysis",
        Some(("frequency", "frequency")),
        true,
    )?;
    sink.begin(&plot)?;
    // NIacIter on the circuit's own matrix: the first AC solve of the
    // circuit reorders (`SMPcReorder`, starting from the real pivot order),
    // later frequencies refactor (`SMPcLUfac`) and reorder only after a zero
    // pivot. See [`ac_matrix`].
    let shared: &Circuit = circuit;
    let mut point = Vec::with_capacity(plot.variable_count());
    for &f in grid {
        let x = small_signal.at_frequency(shared, context, f, |system| {
            ac_matrix(shared, system, 2. * std::f64::consts::PI * f)?
                .factorize()?
                .solve(&system.ac_rhs())
        })?;
        point.clear();
        point.push(Complex::real(f));
        point.extend(x);
        point.extend(circuit.observe_parameters(&context.model_context().with_frequency(f))?);
        plot.check_point(&point)?;
        sink.point(&point)?;
    }
    sink.end()
}

/// Which C frequency loop a grid follows: `acan.c` has a special decade delta
/// that `span.c` lacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GridKind {
    /// `.ac` (`ACan`).
    Ac,
    /// `.sp` (`SPan`).
    Sp,
}

/// The `acan.c`/`span.c` frequency grid of `lin|dec|oct points start stop`,
/// the first four positional arguments of the card, generated as C does:
/// iteratively (`freq *= delta` or `freq += delta`) while
/// `freq <= stop + freqTol`, with `freqTol = delta * stop * reltol` (log) or
/// `delta * reltol` (linear).
///
/// C quirks kept on purpose: a linear sweep of fewer than three points uses
/// `delta = 0` and so yields only the start frequency (Richard `McRoberts`'
/// patch tests `ACnumberSteps - 1 > 1`); `.ac dec` spanning a decade or more
/// uses `delta = exp(ln(stop/start) / floor(|log10(stop/start)| * points))`
/// so the last point lands on `stop`.
pub(super) fn frequency_grid(
    arguments: &[String],
    reltol: Real,
    kind: GridKind,
) -> SpiceResult<Vec<Real>> {
    let argument = |index: usize| arguments.get(index).map(String::as_str);
    let mode = argument(0).unwrap_or_default().to_ascii_lowercase();
    // ACnumberSteps is IF_INTEGER: floor(0.5 + value).
    let points = argument(1)
        .and_then(crate::primitives::parse_spice_number)
        .and_then(crate::primitives::c_integer)
        .filter(|n| *n > 0 && *n <= 100_000)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| unsupported("AC points must be in 1..=100000"))?;
    let start = number(argument(2), "start frequency")?;
    let stop = number(argument(3), "stop frequency")?;
    let steps = Real::from(points);
    let logarithmic = mode != "lin";
    if logarithmic && start <= 0. {
        // acan.c/span.c: "ERROR: AC startfreq <= 0", E_PARMVAL.
        return Err(unsupported("AC start frequency must be > 0 for dec/oct"));
    }
    if start < 0. || stop < start {
        return Err(unsupported("invalid AC frequency bounds"));
    }
    let delta = match mode.as_str() {
        "dec" if kind == GridKind::Ac && stop / 10. >= start => {
            let num_steps = ((stop / start).log10().abs() * steps).floor();
            ((stop / start).ln() / num_steps).exp()
        }
        "dec" if stop <= start && kind == GridKind::Ac => 1.,
        "dec" => (10f64.ln() / steps).exp(),
        "oct" => (2f64.ln() / steps).exp(),
        "lin" if points - 1 > 1 => (stop - start) / (steps - 1.),
        "lin" => 0.,
        _ => return Err(unsupported("AC sweep mode must be lin, dec or oct")),
    };
    let tolerance = if logarithmic {
        delta * stop * reltol
    } else {
        delta * reltol
    };
    let advance = if logarithmic {
        match delta {
            1. => Advance::Single,
            _ => Advance::Multiply(delta),
        }
    } else {
        match delta {
            0. => Advance::Single,
            _ => Advance::Add(delta),
        }
    };
    let analysis = match kind {
        GridKind::Ac => ".ac",
        GridKind::Sp => ".sp",
    };
    FrequencyGrid::new(start, stop, tolerance, advance, analysis)?.collect()
}

/// `CKTacLoad`: the small-signal system at `omega` loaded into the circuit's
/// own matrix, so that factoring it shares SPARSE's frame and pivot order
/// with the real solves (`NIacIter`). Falls back to a standalone matrix when
/// a small-signal stamp lies outside the circuit's allocated elements.
///
/// # Errors
/// Invalid assembly.
pub(super) fn ac_matrix(
    circuit: &Circuit,
    system: &LinearSystem,
    omega: Real,
) -> SpiceResult<ComplexMatrix> {
    if let Some(pattern) = circuit.matrix_pattern()
        && let Some(matrix) = ComplexMatrix::from_operators_grouped_on(
            &pattern,
            &system.a,
            &system.e,
            &system.e_groups,
            omega,
        )?
    {
        return Ok(matrix);
    }
    ComplexMatrix::from_operators_grouped(&system.a, &system.e, &system.e_groups, omega)
}

/// The circuit linearized at its DC operating point, as `acan.c` and `span.c`
/// prepare it before their frequency loops.
pub(super) struct SmallSignal {
    settings: crate::analysis::bias::DcSettings,
    start: crate::analysis::bias::OperatingPointStart,
    state1: crate::devices::StateSnapshot,
    system: LinearSystem,
    /// A `hertz`-dependent device forces a new operating point per frequency.
    varies: bool,
    /// A device whose small-signal stamp depends on the frequency (a
    /// transmission line) forces a reassembly, not a new operating point,
    /// per frequency.
    reassemble: bool,
    previous: Vector,
}

impl SmallSignal {
    /// Solves the operating point and assembles the small-signal system.
    pub(crate) fn prepare(
        circuit: &mut Circuit,
        request: &AnalysisPlan,
        context: &crate::analysis::AnalysisContext,
        settings: crate::analysis::bias::DcSettings,
    ) -> SpiceResult<Self> {
        circuit.finalize()?;
        let start = crate::analysis::bias::OperatingPointStart::resolve(circuit, request)?;
        let solved = start.solve(circuit, &context.model_context(), &settings)?;
        // The MODEINITSMSIG load's state writes outlive the analysis (C keeps
        // one CKTstate0 per deck): later analyses read them back.
        circuit.record_small_signal_state(&context.model_context(), &solved.values)?;
        // acan.c reloads with MODEINITSMSIG after CKTop, and SWload/CSWload then
        // copy CKTstate1 into CKTstate0. CKTop never accepts or rotates states, so
        // CKTstate1 is still the zero ("really off") vector: the AC conductance of
        // a switch is that of the accepted history, not of the converged
        // operating point (`swacload.c`/`cswacld.c`). The operating point's own
        // trial state is therefore deliberately not used here.
        let history = circuit.state_history()?;
        let state1 = circuit.bind_state(
            history
                .accepted(1)
                .map_or_else(|| vec![0.; history.len()], <[Real]>::to_vec),
        )?;
        let system = circuit.small_signal_system_at(
            &context.model_context(),
            &solved.values,
            Some(&state1),
        )?;
        // acan.c: with a `hertz`-dependent device (CKTvarHertz) the operating
        // point is re-solved at every frequency, warm-started from the previous.
        let varies = circuit
            .devices()
            .iter()
            .any(|device| device.depends_on_frequency());
        let reassemble = circuit
            .devices()
            .iter()
            .any(|device| device.small_signal_depends_on_frequency());
        Ok(Self {
            settings,
            start,
            state1,
            system,
            varies,
            reassemble,
            previous: solved.values,
        })
    }

    /// Publish the original bias of this analysis, not a second `CKTop`.
    /// Only the multi-plot interface requests the optional retained plot.
    pub(super) fn retained_bias(
        &self,
        circuit: &Circuit,
        request: &AnalysisPlan,
        context: &crate::analysis::AnalysisContext,
        include: bool,
    ) -> SpiceResult<Vec<Plot>> {
        if !include {
            return Ok(Vec::new());
        }
        let title = match request.kind() {
            crate::primitives::AnalysisKind::Noise => "NOISE Operating Point",
            crate::primitives::AnalysisKind::Distortion
            | crate::primitives::AnalysisKind::PoleZero => "Distortion Operating Point",
            _ => "AC Operating Point",
        };
        let mut plot = plot(circuit, "op1", title, None, false)?;
        let mut point = Vec::with_capacity(plot.variables().len());
        point.extend(self.bias().as_slice().iter().copied().map(Complex::real));
        // Spectral observations support scalar instance parameters. Include
        // them in the bias row as well; its header already names them.
        point.extend(circuit.observe_parameters(&context.model_context())?);
        plot.push_point(&point)?;
        Ok(vec![plot])
    }

    /// The system linearized at the operating point (re-solved at the last
    /// requested frequency for a `hertz`-dependent circuit).
    pub(crate) const fn system(&self) -> &LinearSystem {
        &self.system
    }

    /// The operating point the system is linearized at (re-solved at the last
    /// requested frequency for a `hertz`-dependent circuit).
    pub(crate) const fn bias(&self) -> &Vector {
        &self.previous
    }

    /// The small-signal state (C's `CKTstate0` after the `MODEINITSMSIG`
    /// load): the zero accepted state.
    pub(crate) const fn state(&self) -> &crate::devices::StateSnapshot {
        &self.state1
    }

    /// Whether the operating point is re-solved at every frequency.
    pub(crate) const fn varies(&self) -> bool {
        self.varies
    }

    /// Whether some device's small-signal load is reassembled at every
    /// frequency ([`crate::devices::Device::small_signal_depends_on_frequency`]).
    pub(crate) const fn reassembles(&self) -> bool {
        self.reassemble
    }

    /// Calls `with` on the small-signal system valid at frequency `f`.
    pub(crate) fn at_frequency<R>(
        &mut self,
        circuit: &Circuit,
        context: &crate::analysis::AnalysisContext,
        f: Real,
        with: impl FnOnce(&LinearSystem) -> SpiceResult<R>,
    ) -> SpiceResult<R> {
        if self.varies {
            let model = context.model_context().with_frequency(f);
            self.previous = self
                .start
                .solve_from(
                    &circuit.prepare_run(),
                    &model,
                    &self.settings,
                    &self.previous,
                )?
                .values;
            // C repeats MODEINITSMSIG after every frequency-dependent CKTop.
            // Preserve its carried device slots for the next bias and analysis.
            circuit.record_small_signal_state(&model, &self.previous)?;
            self.system =
                circuit.small_signal_system_at(&model, &self.previous, Some(&self.state1))?;
        } else if self.reassemble {
            // traacld.c loads exp(-j omega TD) at every frequency; the
            // operating point itself does not depend on it.
            let model = context.model_context().with_frequency(f);
            self.system =
                circuit.small_signal_system_at(&model, &self.previous, Some(&self.state1))?;
        }
        with(&self.system)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisContext, AnalysisInput};
    use crate::netlist::{Parser, source::parse_deck_text};
    use crate::primitives::AnalysisKind;
    use std::path::Path;

    fn grid(mode: &str, points: &str, start: &str, stop: &str, kind: GridKind) -> Vec<f64> {
        frequency_grid(
            &[mode.into(), points.into(), start.into(), stop.into()],
            1e-3,
            kind,
        )
        .unwrap()
    }

    #[test]
    fn singleton_sweeps_keep_reference_exactness() {
        assert_eq!(grid("dec", "10", "42", "42", GridKind::Ac), [42.]);
        assert_eq!(grid("lin", "2", "1", "100", GridKind::Ac), [1.]);
        assert_eq!(grid("lin", "3", "42", "42", GridKind::Sp), [42.]);
    }

    #[test]
    fn linear_grid_keeps_repeated_addition_and_inclusive_endpoint() {
        assert_eq!(grid("lin", "3", "1", "3", GridKind::Ac), [1., 2., 3.]);
        assert_eq!(grid("lin", "3", "1", "3", GridKind::Sp), [1., 2., 3.]);
    }

    fn carried(circuit: &Circuit) -> Vec<Option<Real>> {
        let history = circuit.state_history().unwrap();
        let mut trial = history.trial();
        let mut values = Vec::new();
        for index in 0..circuit.device_count() {
            let state = history.device_of(&mut trial, index).unwrap();
            values.extend((0..state.len()).map(|slot| state.carried(slot)));
        }
        values
    }

    #[test]
    fn ordinary_operating_point_records_small_signal_state_for_later_analyses() {
        let netlist = Parser::new().parse_deck(&parse_deck_text(
            Path::new("op-state.cir"),
            "BJT operating point\nvcc c 0 5\nvbase b 0 0.3\nq1 c b 0 qmod\n.model qmod npn (is=1e-15 bf=100 rb=100 cjc=2p xcjc=0.5 cjs=3p)\n.end\n",
        )).unwrap();
        let mut circuit = Circuit::from_netlist(&netlist).unwrap();
        let mut same_bias = Circuit::from_netlist(&netlist).unwrap();
        let context = AnalysisContext::default();
        let request =
            AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
                .resolve()
                .unwrap();
        crate::analysis::linear::op(&mut circuit, &request, &context).unwrap();
        let settings = request.dc_settings().clone();
        SmallSignal::prepare(&mut same_bias, &request, &context, settings).unwrap();
        let expected = carried(&same_bias);
        assert!(expected.iter().flatten().any(|value| *value > 0.));
        assert_eq!(carried(&circuit), expected);
    }

    #[test]
    fn retained_bias_contains_observations_without_changing_the_sweep() {
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("retained-bias.cir"),
                "AC with retained bias\nv1 in 0 dc 2 ac 1\nr1 in out 1k\nr2 out 0 1k\n.end\n",
            ))
            .unwrap();
        let mut circuit = Circuit::from_netlist(&netlist).unwrap();
        let mut other = Circuit::from_netlist(&netlist).unwrap();
        let observations = ["@r1[resistance]".to_owned()];
        circuit.set_observations(&observations).unwrap();
        other.set_observations(&observations).unwrap();
        let context = AnalysisContext::default();
        let mut request =
            AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "100", "300"]);
        request.keep_op_info = Some(true);
        let request = request.resolve().unwrap();
        let plots = request.run_plots(&mut circuit, &context).unwrap();
        assert_eq!(plots.len(), 2);
        assert_eq!(plots[0].plotname, "AC Operating Point");
        assert_eq!(plots[0].value("v(out)", 0), Some(Complex::real(1.)));
        assert_eq!(
            plots[0].value("@r1[resistance]", 0),
            Some(Complex::real(1000.))
        );
        assert_eq!(plots[1], request.run(&mut other, &context).unwrap());
    }

    #[test]
    fn frequency_dependent_rebias_reapplies_nodesets_like_c() {
        // At 100 Hz the negative bias offset leaves only the negative branch.
        // At 150 Hz the positive branch exists again. C's next MODEINITJCT
        // forces the same nodeset, selecting that branch despite the warm start.
        // Multiplication by the unit AC input exposes the DC bias as gain.
        let netlist = Parser::new().parse_deck(&parse_deck_text(
            Path::new("rebias-nodeset.cir"),
            "hertz latch\nr1 a 0 1k\nb1 0 a i=(2*tanh(4*v(a))-2.5*exp(-((hertz-100)/20)^2)+v(in))/1000\nvin in 0 dc 0 ac 1\nb2 out 0 v=v(a)*v(in)\nr2 out 0 1k\n.nodeset v(a)=2\n.ac lin 3 100 200\n.end\n",
        )).unwrap();
        let config = crate::analysis::RunConfig::from_netlist(&netlist).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let request = config.plan_for(&netlist.analyses[0]).unwrap();
        let plot = request.run(&mut circuit, &config.context()).unwrap();
        for (row, reference) in [-4.5, 1.9951733971375833, 1.9999995498230105]
            .into_iter()
            .enumerate()
        {
            let gain = plot.value("v(out)", row).unwrap().re;
            assert!(
                (gain - reference).abs() < 1e-6,
                "row {row}: {gain} vs C {reference}"
            );
        }
    }

    #[test]
    fn frequency_dependent_rebias_refreshes_carried_small_signal_state() {
        let netlist = Parser::new().parse_deck(&parse_deck_text(
            Path::new("rebias.cir"),
            "frequency dependent BJT bias\nb1 c 0 v=5+hertz/100\nvbase b 0 dc 0.3 ac 1\nq1 c b 0 qmod\n.model qmod npn (is=1e-15 bf=100 rb=100 cjc=2p xcjc=0.5 cjs=3p)\n.end\n",
        )).unwrap();
        let mut circuit = Circuit::from_netlist(&netlist).unwrap();
        let context = AnalysisContext::default();
        let request = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "100", "300"])
            .resolve()
            .unwrap();
        let settings = request.dc_settings().clone();
        let mut small = SmallSignal::prepare(&mut circuit, &request, &context, settings).unwrap();
        assert!(small.varies());
        let initial = carried(&circuit);
        small
            .at_frequency(&circuit, &context, 300., |_| Ok(()))
            .unwrap();
        let actual = carried(&circuit);
        circuit
            .record_small_signal_state(&context.model_context().with_frequency(300.), small.bias())
            .unwrap();
        let expected = carried(&circuit);
        assert_ne!(initial, expected);
        assert_eq!(actual, expected);
    }
}
