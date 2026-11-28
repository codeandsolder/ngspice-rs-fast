Independent synthetic operating-point witness for the prepared VDMOS consumer.
op.expected contains the three source currents printed by ngspice 47 with
KLU (2026-11-28). Regenerate on a host with ngspice 47:

    ngspice -b op.cir | rg '^i\(vd[012]\)\s*=' > op.expected

The regular engine test uses the recorded data. The optional live test runs
the exact same deck when the independent executable is present in the test
runtime. No native solver results contribute to the reference.
