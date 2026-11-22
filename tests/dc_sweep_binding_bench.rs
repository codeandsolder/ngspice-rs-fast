//! Manual phase-separated DC binding/preflight/solve benchmark. Compare the
//! same source and checksum in independent baseline and candidate processes.
use ngspice_rs::analysis::{Plot, PointSink, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{Complex, SpiceResult};
use std::fmt::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

struct TimedSink {
    start: Instant,
    prepared: Duration,
    solve: Duration,
    rows: usize,
    checksum: u64,
}
impl PointSink for TimedSink {
    fn begin(&mut self, _: &Plot) -> SpiceResult<()> {
        self.prepared = self.start.elapsed();
        Ok(())
    }
    fn point(&mut self, row: &[Complex]) -> SpiceResult<()> {
        self.rows += 1;
        for value in row {
            self.checksum = self.checksum.rotate_left(7) ^ value.re.to_bits();
            self.checksum = self.checksum.rotate_left(7) ^ value.im.to_bits();
        }
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        self.solve = self.start.elapsed() - self.prepared;
        Ok(())
    }
}

fn measure(devices: usize, args: &str, points: usize) {
    let mut text = "DC sweep binding benchmark\n".to_owned();
    for index in 0..devices {
        writeln!(text, "r{index} a 0 rm 1000").unwrap();
    }
    writeln!(
        text,
        "r_target a 0 rm 1500\nvdrive a 0 1\n.model rm r(tc1=0.001)\n.dc {args}\n.end"
    )
    .unwrap();
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("binding.cir"), &text))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let mut sink = TimedSink {
        start: Instant::now(),
        prepared: Duration::ZERO,
        solve: Duration::ZERO,
        rows: 0,
        checksum: 0,
    };
    request
        .run_into(&mut circuit, &config.context(), &mut sink)
        .unwrap();
    assert_eq!(sink.rows, points);
    println!(
        "devices={},rows={},prepare_us={},solve_us={},checksum={:016x}",
        circuit.device_count(),
        sink.rows,
        sink.prepared.as_micros(),
        sink.solve.as_micros(),
        sink.checksum
    );
}

#[test]
#[ignore = "manual phase-separated benchmark"]
fn large_source_sweep() {
    measure(8192, "vdrive 0 511 1", 512);
}

#[test]
#[ignore = "manual phase-separated benchmark"]
fn large_resistor_sweep() {
    measure(8192, "r_target 1500 2011 1", 512);
}

#[test]
#[ignore = "manual phase-separated benchmark"]
fn mixed_physical_sweep() {
    measure(1024, "r_target 1500 1627 1 temp 27 42 1", 2048);
}
