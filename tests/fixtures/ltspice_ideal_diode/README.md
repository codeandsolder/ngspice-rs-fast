# Independent common-region witness

The authored circuit uses ngspice's XSPICE sidiode with no smoothing or
finite current limits. Those regions have the same piecewise-linear law
as the LTspice idealized diode. The finite-limit and smooth-region laws
are not assumed equivalent.

Regenerate with:

```sh
ngspice -n -b common.cir | awk '$1 ~ /^[0-9]+$/ && NF==3 { printf "%s\t%.17g\n",$2,-$3 }' > common.tsv
```

Columns: applied anode-to-cathode voltage (V), terminal current (A).
The minus sign converts source current to diode current.
