# Independent storage-current reference (LTspice 24.0.12)

The synthetic source deck is ltspice-24.0.12-current.cir. Run it in batch
ASCII-raw mode using LTspice 24.0.12. For the operating-point and DC-sweep
references replace its .ac card with .op and .dc Vl3rc 0 1 .5,
respectively. These are separate runs of the same circuit. Temperature is
27 °C (default). No vendor implementation code or model library is used.

ltspice-24.0.12-current.tsv contains 168 records, one per storage device
per analysis point. Its tab-separated fields are:

    analysis, zero-based point, suffix, terminal voltage (re, im),
    driving resistor current (re, im), reported device current (re, im)

The drive current is an independent physical witness of external current;
it is not inferred from the reported storage current. Capacitor and
inductor cards each have four multiplicities (0.5, 1, 2, 3) and three
parallel parasitic combinations (R only, C only, both). The AC points
are 100, 5050, and 10000 Hz. All complex values retain the rawfile's
decimal precision. The test compares terminal voltage, independent
physical port current, and reported current separately.

ltspice-24.0.12-transient.cir independently exercises a switching stimulus
over 30 us for both full parasitic networks. At the final accepted point
the measured terminal voltages are 0.7584027154835773 and
0.7260855908107419 V; the reported currents are 0.1207986422582508 and
0.1369572045946370 A, respectively. Both currents agree with their
individual driving-resistor currents. The test uses a relative numerical
tolerance rather than requiring identical adaptive time grids.

The independent AC experiment for V source output-current sensing uses
a 1 V AC source with Rser=2, Cpar=1u and a 3-ohm load. At 10 kHz, the
unsensed output is 0.5966083459066635-j0.04498320951769758 V; with an F or
behavioral I(V) control reference it is
0.602261102728891+j0.02998880634513171 V, consistent with capacitance
spanning the series resistance instead of the output. This is a
source-normalization distinction; the engine's physical topology is
explicitly selected. Controlled-source transient behavior is not
established by these AC data.

### Series-inductor shunt and omission experiments

The synthetic shunt-transient and shunt-AC decks measure four multiplicities (1, 2, 3, 0.5). At 30 us the terminal voltages are 0.9153207353710472, 0.8201352172432387, 0.7260842981738106 and 0.9497307705099741 V. Independent explicit ngspice-47 circuits reproduce these endpoints when the aggregate RLshunt is 20 ohms, independent of multiplicity, rather than 20/m ohms. The AC measurements at 100, 5050 and 10000 Hz instead follow a fully scaled parallel-copy admittance. The normalized test uses per-copy pn_gseries_ac=1/20 and pn_gseries_tran=1/(20*m).

The Cpar-default deck at 1 GHz establishes that omitted Cpar with nonzero Lser is C*1e-6 per copy: C=2u yields 2p, C=4u yields 4p, and m=3 yields 6p aggregate. Explicit Cpar=0, 0.1p and 3p are preserved, not floored. Lser=0 does not trigger the default. These are source-normalization decisions, not native coefficient defaults.

The additional shunt-noise experiment uses m=3, a 2-ohm driver and the same complete capacitor network. LTspice output-noise amplitude spectral density at 100, 5050 and 10000 Hz is 1.768349869679772e-10, 1.580738710815810e-10 and 1.520717676270345e-10 V/sqrt(Hz); input noise is 1.874645005376138e-10, 2.044605719103892e-10 and 2.067466055045742e-10 V/sqrt(Hz). An explicit ngspice-47 reference with aggregate RLshunt=20/3 ohm matches to 1e-6 relative, whereas aggregate RLshunt=20 ohm differs by approximately 5–6% at 5050 and 10000 Hz. Thus the noise law uses the AC multiplicity scaling, not the transient scaling.

### Controlled voltage-source transient qualification boundary

The synthetic LTspice controlled-source transient deck uses a smooth 10 kHz sine, Rser=2, Cpar=1u and a 3-ohm load. At 2 us the unsensed output is 0.0386425695 V; the current-sensed output is 0.0000646877 V and reported current +0.00976394 A at reltol=1e-4. At reltol=1e-3 the sensed output is 0.000792292 V and reported current -0.0102100 A. Gear and reltol=1e-5 exceeded a deliberately bounded runtime/output size. F and behavioral I(V) references gave identical trajectories at reltol=1e-4. With Cpar removed, sensed and unsensed voltages are both 0.0751999401 V and current -0.0250666467 A.

The independently explicit physical ngspice-47 deck (output shunt unsensed, series-resistor shunt sensed, oriented external ideal probe) yields 0.03864237066517964 V and 0.09957165312085969 V, with sensed external current -0.03319055104028656 A. The native engine represents this physical topology explicitly. The LTspice current-sensed transient reference is tolerance-sensitive/nonconvergent; these data do not establish a stable additional physical rule or transient vendor parity. The AC measurements remain independently qualified.

The zero-multiplicity OP deck measures V(live)=1, V(ghost)=0 and I(C1)=I(C2)=0 in LTspice 24.0.12. The reference reports source-visible identities even though physical storage is absent. Native zero-multiplicity removal creates no storage devices; source-visible zero-current output is a separate producer qualification obligation (#80).
