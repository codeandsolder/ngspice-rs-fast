# Independent prepared-waveform witnesses

Measured with LTspice 24.0.12 (Windows), using the included input deck and
Gear integration, `plotwinsize=0`, `numdgt=15`. The TSV has the unmodified
decoded binary64 values from 139 of the transient samples: first 14,
every 13th source row, windows around the key edges and the final row.
Columns: time, V(a), V(b), V(c), V(d), V(e), V(f), V(g), V(h), V(i).
The 1-ohm load makes V(c) equal to the independent pulse current.

The companion trapezoidal runs smooth the resistor voltage around the
fractional-count current-source cutoff, whereas Gear and direct voltage
source witnesses exhibit the hard transition. Use Gear here to avoid
qualifying the reference program's output processing as a source law.
At a negative-frequency finite sine's startup release, LTspice records
the left landing value at the exact release time. The solver explicitly
represents both external limits, with prepared transient startup using
the physical right limit. The witness test compares the recorded left
value at the two exact landing samples and the actual law everywhere else.
