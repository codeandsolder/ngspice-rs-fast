// Usage: node experiments/circuit-lto/summarize.mjs benchmark.log
import fs from 'node:fs';
const samples = new Map();
const median = xs => [...xs].sort((a,b)=>a-b)[Math.floor(xs.length/2)];
for (const line of fs.readFileSync(process.argv[2], 'utf8').split('\n')) {
  const m = line.match(/^(\S+) (generic|dispatch|baked) count=(\d+) setup_ns=(\d+) solve_ns=(\d+) result_ns=(\d+) points=(\d+) digest=([0-9a-f]+)/);
  if (!m || +m[3] === 1) continue;
  const [,name,mode,count,setup,solve,result,points,digest] = m;
  const cases = samples.get(name) ?? new Map();
  const records = cases.get(mode) ?? [];
  records.push({count:+count, setup:+setup/+count, solve:+solve/+count, result:+result/+count, points:+points, digest});
  cases.set(mode,records);
  samples.set(name,cases);
}
console.log('| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |');
console.log('|---|---:|---:|---:|---:|---:|');
for (const [name, modes] of samples) {
  const records = [...modes.values()].flat();
  if (new Set(records.map(r=>r.count+":"+r.points+":"+r.digest)).size !== 1) throw Error('result mismatch '+name);
  const g=median(modes.get('generic').map(r=>r.solve))/1e6;
  const d=median(modes.get('dispatch').map(r=>r.solve))/1e6;
  const b=median(modes.get('baked').map(r=>r.solve))/1e6;
  console.log('| '+[name,g.toFixed(3),d.toFixed(3),b.toFixed(3),(g/d).toFixed(3)+'x',(g/b).toFixed(3)+'x'].join(' | ')+' |');
}
