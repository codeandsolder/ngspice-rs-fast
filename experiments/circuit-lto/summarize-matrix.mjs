// Usage: node summarize-matrix.mjs matrix.log
import fs from 'node:fs';
const data=new Map();
for(const line of fs.readFileSync(process.argv[2],'utf8').split('\n')) {
  const m=line.match(/^(\S+) (\S+) (generic|baked) count=(\d+) setup_ns=(\d+) solve_ns=(\d+) result_ns=(\d+) points=(\d+) digest=(\w+)/);
  if(!m) continue;
  const [,v,c,mode,n,,solve,,pts,digest]=m;
  const k=[c,v,mode].join('|');
  const a=data.get(k)||[];
  a.push({ms:+solve/+n/1e6,points:+pts,digest});
  data.set(k,a);
}
const med=a=>a.map(x=>x.ms).sort((a,b)=>a-b)[Math.floor(a.length/2)];
const cases=[...new Set([...data.keys()].map(k=>k.split('|')[0]))];
const variants=[...new Set([...data.keys()].map(k=>k.split('|')[1]))];
console.log('| Case | Variant | Generic ms | Baked ms | Baked speedup | Generic relative to znver2 native |');
console.log('|---|---|---:|---:|---:|---:|');
for(const c of cases)for(const v of variants){
  const g=data.get([c,v,'generic'].join('|'));
  const b=data.get([c,v,'baked'].join('|'));
  if(g?.length!==5||b?.length!==5) throw Error('incomplete '+c+' '+v);
  for(const a of [g,b]) if(new Set(a.map(x=>x.points+':'+x.digest)).size!==1) throw Error('unstable result '+c+' '+v);
  const x=med(g),y=med(b),base=med(data.get([c,'znver2/native','generic'].join('|')));
  console.log('| '+[c,v,x.toFixed(4),y.toFixed(4),(x/y).toFixed(3)+'x',(base/x).toFixed(3)+'x'].join(' | ')+' |');
}
