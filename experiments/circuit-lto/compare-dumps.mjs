// Compare full output, including independent-axis alignment. No interpolation
// is used: a different grid is a separate observation, not an equal result.
import fs from 'node:fs';
function parse(path) {
  const lines=fs.readFileSync(path,'utf8').trim().split('\n');
  const plots=[];
  while (lines.length) {
    const [tag,n,...columns]=lines.shift().split('\t');
    if(tag!=='plot') throw Error('bad plot header: '+tag);
    const rows=lines.splice(0,+n).map(line=>line.split('\t').flatMap(cell=>cell.split(',').map(Number)));
    if(rows.length!==+n || rows.some(r=>r.length!==columns.length*2)) throw Error('bad plot shape');
    plots.push({columns,rows});
  }
  return plots;
}
const [reference,candidate]=process.argv.slice(2).map(parse);
if(reference.length!==candidate.length) throw Error('plot count changed');
let maxAbsolute=0,maxScaled=0,changed=0,cells=0,axisMax=0;
for(let i=0;i<reference.length;i++){
  const a=reference[i],b=candidate[i];
  if(JSON.stringify(a.columns)!==JSON.stringify(b.columns)) throw Error('columns changed');
  if(a.rows.length!==b.rows.length){
    console.log(JSON.stringify({gridChanged:true,reference:a.rows.length,candidate:b.rows.length}));
    process.exit(2);
  }
  for(let j=0;j<a.rows.length;j++) for(let k=0;k<a.rows[j].length;k++){
    const x=a.rows[j][k],y=b.rows[j][k];
    if(!Number.isFinite(x)||!Number.isFinite(y)) throw Error('nonfinite output');
    const d=Math.abs(x-y);
    cells++;
    if(!Object.is(x,y)) changed++;
    maxAbsolute=Math.max(maxAbsolute,d);
    // Scale by max(1e-12, |reference|, |candidate|), report raw error.
    maxScaled=Math.max(maxScaled,d/Math.max(1e-12,Math.abs(x),Math.abs(y)));
    if(k<2 && /^(time|frequency|v-sweep|i-sweep|temp-sweep)/i.test(a.columns[0])) axisMax=Math.max(axisMax,d);
  }
}
console.log(JSON.stringify({cells,changed,maxAbsolute,maxScaled,axisMax}));
