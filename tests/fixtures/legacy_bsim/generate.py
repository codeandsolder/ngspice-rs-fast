from pathlib import Path
import argparse
import itertools
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description="Generate independent legacy BSIM witnesses")
parser.add_argument("--ngspice", default="ngspice")
args = parser.parse_args()
out = Path(__file__).resolve().parent
base='tox=.024 vdd=5 vfb=-.9 phi=.75 k1=.75 k2=.02 n0=1.2 dl=0 dw=0 rsh=52 js=2e-5 cj=1e-3 cjsw=2e-10 pb=.74 pbsw=.83 mj=.5 mjsw=.33 cgdo=1.5e-10 cgso=2e-10 cgbo=4e-11 kf=2e-25 af=1.2'
family={4:'mus=510 muz=400 eta=.02 u0=.1 u1=.06',5:'mus0=510 mu0=400 eta0=.02 ua0=.1 u10=.06 vbb=-3 ai0=.05 bi0=.5'}
geometry='l=8.3u w=17.6u m=2 ad=30p as=40p pd=13u ps=17u nrd=2.3 nrs=3.1'
def build_deck(forward_equivalent, transient=False):
    lines = ["Independent legacy BSIM1/2 physical terminal-current witnesses"]
    names = []
    for level, polarity in itertools.product((4, 5), ("n", "p")):
        model = f"mod{level}{polarity}"
        lines.append(f".model {model} {polarity}mos(level={level} {base} {family[level]})")
        if forward_equivalent:
            reversed_base = base.replace("cgdo=1.5e-10 cgso=2e-10", "cgdo=2e-10 cgso=1.5e-10")
            lines.append(f".model {model}rev {polarity}mos(level={level} {reversed_base} {family[level]})")
        sign = 1 if polarity == "n" else -1
        for direction in ("f", "r"):
            label = f"{level}{polarity}{direction}"
            d = sign * (.8 if direction == "f" else -.6)
            g = sign * (2. if direction == "f" else 1.5)
            bulk = sign * (-.2 if direction == "f" else -1.)
            lines.extend([
                f"vd{label} d{label} 0 {d:.12g}",
                (f"vg{label} g{label} 0 PWL(0 {g-sign*.5:.12g} 1u {g-sign*.5:.12g} 1.1u {g:.12g} 3u {g:.12g})" if transient else f"vg{label} g{label} 0 {g:.12g} ac 1"),
                f"vb{label} b{label} 0 {bulk:.12g}",
            ])
            if forward_equivalent and direction == "r":
                reversed_geometry = "l=8.3u w=17.6u m=2 ad=40p as=30p pd=17u ps=13u nrd=3.1 nrs=2.3"
                lines.append(f"m{label} 0 g{label} d{label} b{label} {model}rev {reversed_geometry}")
            else:
                lines.append(f"m{label} d{label} g{label} 0 b{label} {model} {geometry}")
            names.extend([f"i(vd{label})", f"i(vg{label})", f"i(vb{label})"])
    lines.extend([".options gmin=1e-15 reltol=1e-9 abstol=1e-13 vntol=1e-9", ".end"])
    return "\n".join(lines) + "\n", names


def run_oracle(deck, names):
    oracle = deck.removesuffix(".end\n") + ".control\nset numdgt=17\nop\nprint " + " ".join(names) + "\nac lin 1 1meg 1meg\nprint " + " ".join(names) + "\nquit\n.endc\n.end\n"
    with tempfile.TemporaryDirectory(prefix="legacy-bsim-") as directory:
        ref = Path(directory)
        source = ref / "dc-ac-oracle.cir"
        source.write_text(oracle)
        proc = subprocess.run(
            [args.ngspice, "-b", str(source)], cwd=ref,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
        )
    found = {key: [] for key in names}
    for line in proc.stdout.splitlines():
        match = re.match(r"(i\(v[dgb][^)]*\))\s*=\s*(.+)", line)
        if match and match[1] in found:
            found[match[1]].append(match[2].strip())
    assert all(len(value) == 2 for value in found.values()), found
    rows = ["# ngspice 47 KLU: source, DC A, AC real A, AC imag A"]
    for name, values in found.items():
        real, imag = values[1].split(",")
        rows.append(f"{name}\t{float(values[0]):.17e}\t{float(real):.17e}\t{float(imag):.17e}")
    return "\n".join(rows) + "\n"


for equivalent in (False, True):
    deck, names = build_deck(equivalent)
    stem = "dc-ac-forward-equivalent" if equivalent else "dc-ac"
    (out / f"{stem}.cir").write_text(deck)
    (out / f"{stem}.tsv").write_text(run_oracle(deck, names))

# Compare physical transient currents away from PWL slope discontinuities.
import bisect
times = [0.5e-6, 1.025e-6, 1.05e-6, 1.075e-6, 1.2e-6, 2.5e-6]
direct, names = build_deck(False, transient=True)
equivalent, _ = build_deck(True, transient=True)
(out / "tran.cir").write_text(direct)
(out / "tran-forward-equivalent.cir").write_text(equivalent)
with tempfile.TemporaryDirectory(prefix="legacy-bsim-tran-") as directory:
    ref = Path(directory)
    source = ref / "tran-oracle.cir"
    source.write_text(equivalent.removesuffix(".end\n") + ".control\nset wr_singlescale\nset numdgt=17\ntran 0.5n 3u 0 0.5n\nwrdata currents.dat " + " ".join(names) + "\nquit\n.endc\n.end\n")
    proc = subprocess.run([args.ngspice, "-b", str(source)], cwd=ref, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    data = [list(map(float, line.split())) for line in (ref / "currents.dat").read_text().splitlines()]
    assert all(len(row) == len(names)+1 for row in data), data[:2]
    grid = [row[0] for row in data]
    rows = ["# ngspice 47 KLU: time s, source, current A"]
    for time in times:
        index = bisect.bisect_left(grid, time)
        left, right = data[index-1], data[index]
        fraction = (time-left[0])/(right[0]-left[0])
        for column, name in enumerate(names, 1):
            current = left[column] + fraction*(right[column]-left[column])
            rows.append(f"{time:.17e}\t{name}\t{current:.17e}")
(out / "tran-forward-equivalent.tsv").write_text("\n".join(rows)+"\n")

# Independent physical noise laws use measured DC current and gate slope.
rows=['# ngspice 47 KLU: level, polarity, orientation, drain A, dId/dVg S']
for level,polarity,direction in itertools.product((4,5),('n','p'),('f','r')):
    sign=1 if polarity=='n' else -1
    d=sign*(.8 if direction=='f' else -.6)
    g=sign*(2 if direction=='f' else 1.5)
    bulk=sign*(-.2 if direction=='f' else -1.)
    # Direct DC is reliable in both orientations. Measure source current
    # at three independent gate biases, not C's cached gm/noise state slot.
    noise_model=base.replace('rsh=52', 'rsh=0')
    deck=f'noise physical witness\nvd d 0 {d}\nvg g 0 {g}\nvb b 0 {bulk}\nm1 d g 0 b mm {geometry}\n.model mm {polarity}mos(level={level} {noise_model} {family[level]})\n.options gmin=1e-15 reltol=1e-10 abstol=1e-15 vntol=1e-11\n.control\nset numdgt=17\n'
    for v in [g,g-1e-5,g+1e-5]:
        deck+=f'alter vg {v:.17e}\nop\nprint i(vd)\n'
    deck+='quit\n.endc\n.end\n'
    with tempfile.TemporaryDirectory(prefix='legacy-bsim-noise-') as td:
        path=Path(td)/'noise.cir';path.write_text(deck)
        result=subprocess.run([args.ngspice,'-b',str(path)],cwd=td,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
    values=[float(x) for x in re.findall(r'^i\(vd\)\s*=\s*(\S+)',result.stdout,re.M)]
    assert len(values)==3,result.stdout
    rows.append(f'{level}\t{polarity}\t{direction}\t{-values[0]:.17e}\t{-(values[2]-values[1])/2e-5:.17e}')
(out / "noise-dc.tsv").write_text('\n'.join(rows)+'\n')
