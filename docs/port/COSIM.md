# Compiled HDL co-simulation

HDL is a separately built, application-selected executable, not code evaluated
by a SPICE deck. Bind a `devices::digital_bridge::Bridge` to the native digital
network and use existing ADC/DAC bindings for analog connectivity. No native
simulator library is loaded into the Rust numerical process.

## Scheduler and checkpoints

A simulator declares ordered flattened input/output bits and a maximum state
size. Its decimal precision is explicit (1 fs to 1 s); the bridge floors
physical time to the latest representable tick. Absolute ticks are limited to
2^52-1 to preserve adjacent tick resolution. The four wire values are 0, 1,
X and Z; Z drives high impedance, X drives a strong unknown. Verilator's
two-state adapter rejects X/Z inputs instead of inventing a binary value.
GHDL normalizes H/L to 1/0 and U/W/- to X.

The native event queue supports model-owned autonomous wakeups. A simulator
can return its next time slot (scheduled mode), or advance with unchanged
inputs to the first settled output change or a requested horizon (lookahead).
Predictions are retained across unrelated events but invalidated by a resolved
input change. At coincident time, incoming digital drives take precedence over
the speculative autonomous prediction; the model settles again with new inputs.
Existing delta-cycle and event-work limits also bound autonomous evaluation.

Each trial clones immutable model checkpoints. Rejecting a trial does not
alter the accepted state or counters. The process transport supports complete
opaque snapshots (NSH1), or deterministic replay (NSR1) for simulators that do
not serialize their scheduler. Replay keeps a shared immutable history, runs
forward in a persistent process, and restarts/replays a prior checkpoint on
rollback. Replay verifies all recorded reached times, next times and outputs.
History length and byte budgets are explicit. Frequent rewinds of long
histories can be expensive; there is no constant-cost rollback guarantee.

## Version 1 binary protocol

All integers are little-endian. Every message begins with a u32 payload length
(excluding the length field). The request payload is:

| Field | Type |
|---|---|
| Magic | 4 bytes: NSH1 or NSR1 |
| Absolute tick | u64 |
| Precision exponent (seconds) | i32 |
| Operation | u32: 0 apply inputs, 1 advance held |
| SPICE delta | u32 |
| Input width | u32 |
| Output width | u32 |
| Snapshot length | u32; MAX means uninitialized/absent |
| Inputs | one byte per bit |
| Snapshot | opaque bytes, if present |

A successful response contains magic, u32 status=0, u64 reached tick, u64 next
tick (MAX means absent), u32 output width, u32 checkpoint length, output bit
bytes and checkpoint bytes. The bit encoding is 0=zero, 1=one, 2=X, 3=Z.
A failure response contains magic, u32 status=1 and up to 4096 UTF-8 diagnostic
bytes. NSR1 always sends absent snapshots and receives zero-length snapshots.
An apply must reach its requested tick; a lookahead may stop earlier only at
an output change. Lookahead must not provide scheduled wakeups.

The transport bounds frame sizes before allocation (up to 65536 bits per
direction, 16 MiB checkpoint), validates dimensions/version/logic/times,
and enforces a per-exchange timeout. On failure it terminates and reaps the
direct process. The executable and working directory must be absolute;
arguments are passed directly, not through a shell. The adapter is trusted:
this is process separation, not a sandbox for hostile code or descendants.

## Example adapters

POSIX C++20 wrappers live in `examples/cosim`.

- Verilator 5.054: timed compiled HDL, explicit C++ mapping of flattened ports,
  uses nextTimeSlot/eventsPending. Example clocked_filter.sv and clocked_filter.cpp.
- GHDL 6.0.0 mcode: public VPI callbacks, horizon/early-output mode, scalar
  logic and logic-vector ports, inputs and outputs separately in declaration
  order, vector bits in declaration order. No bidirectional ports or
  mixed-language hierarchy in this adapter. Use std_logic for four-state
  signals. GHDL's VHPI next-time query is unimplemented in this release.
- Icarus and other adapters are not implemented here.

From a separate build directory, with absolute source paths:

```sh
verilator --cc --exe --build --timing --top-module clocked_filter \
  --Mdir obj_dir -CFLAGS -std=c++20 /path/examples/cosim/clocked_filter.sv \
  /path/examples/cosim/clocked_filter.cpp
ghdl -a --std=08 /path/examples/cosim/clocked_filter.vhd
ghdl --vpi-compile g++ -std=c++20 -fPIC -c /path/examples/cosim/ghdl_bridge.cpp -o bridge.o
ghdl --vpi-link g++ -shared -o bridge.so bridge.o
ghdl -r --std=08 clocked_filter --vpi=./bridge.so --stop-delta=1024
```

The GHDL command is the executable/argument sequence for ReplaySimulator
in lookahead mode; it expects framed requests on stdin, not interactive text.
Verilator's executable uses scheduled mode. For GHDL default precision use
-15; the provided Verilator model uses -9. Both adapters check precision.

## Verification

`cargo test --test verilog_vhdl_cosim` exercises native bridge behavior without
external compilers. Run with `-- --include-ignored --nocapture` to compile and
execute the optional Verilator/GHDL tests (both tools on PATH). All generated
files belong in temporary test directories, not the source tree.

The 50 us ADC–clocked filter–DAC benchmark with Verilator measured 137
accepted analog steps versus 10080 forced 5 ns steps, with 10003 HDL
evaluations in either case (intermediate implementation: 143 ms vs 181 ms).
The analog output transitions at 35 ns and 45 ns; the internal HDL clock does
not force an analog solve at each edge.

The GHDL backend achieved the same 137 versus 10080 accepted analog steps,
with 10007 delivered native events and the same output transitions.
The adaptive run used 10134 horizon probes versus 20077 for fixed stepping;
wall time was 445 ms versus 836 ms. Timings vary with concurrent build load.
Live GHDL tests also exercise a 65-bit four-state bus (including X and Z)
and deterministic rejected-trial recovery. The native competing-clock
regression checks prediction reuse and coincident input cancellation.

The closed-loop RC oscillator example (feedback_inverter in both HDLs)
uses a 1 kohm/100 nF RC network, hysteretic ADC (1 V/2 V), 500 ns ADC
delay, 500 ns HDL inverter delay and 1 us DAC slew. Both compiled models
produced seven transitions over 600 us, with a 71.555 us half-period
and maximum cross-language midpoint difference 3.48 ns.

Independent reference observations using unchanged upstream ngspice
examples: Verilator delay.v (five-bit autonomous ring, 100 ns precision)
matched all 12 states of the live ngspice-47 d_cosim eprint trace from
0 to 11 us; the maximum C event timestamp offset was 28.2 ns. GHDL
pwm.vhd matched all 21 settled transitions through 10 ms against a
standalone GHDL testbench, including exact integer-femtosecond times.
The upstream shared-library VHDL integration was not runnable on the
test host: its portable LLVM compiler required libLLVM.so.18.1 and
libgnat-13.so, while the mcode adapter needed neither.

The larger Verilog benchmark adds a 64-stage native D-flip-flop
circulating register driven by the compiled clock. The same 50 us interval
delivered 660135 events and 970131 model evaluations in both runs.
Every register made 5001 transitions including initialization and
returned to its expected initial state after 5000 rising edges.
Adaptive: 137 accepted analog steps, 1.226 s; fixed 5 ns: 10080 steps,
2.938 s (2.40x wall-time improvement on this run).
