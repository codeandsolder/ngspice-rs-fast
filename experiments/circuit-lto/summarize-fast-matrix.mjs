import fs from 'node:fs';
const data=new Map();
for(const line of fs.readFileSync(process.argv[2],'utf8').split('\n')){
 const m=line.match(/^(\S+) (\S+) (generic|baked) count=(\d+) setup_ns=(\d+) solve_ns=(\d+) result_ns=(\d+) points=(\d+) digest=(\w+)/);
 if(!m)continue;
 const [,v,c,mode,n,,solve,,pts,digest]=m,k=[c,v,mode].join('|');
 const a=data.get(k)||[];a.push({ms:+solve/+n/1e6,pts,digest});data.set(k,a);
}
const med=a=>a.map(x=>x.ms).sort((a,b)=>a-b)[Math.floor(a.length/2)];
const cases=[...new Set([...data.keys()].map(k=>k.split('|')[0]))];
const variants=[...new Set([...data.keys()].map(k=>k.split('|')[1]))];
console.log('| Case | Variant | Generic ms | Baked ms | Generic speedup vs strict | Baked speedup vs strict | Baked vs generic |');
console.log('|---|---|---:|---:|---:|---:|---:|');
for(const c of cases) for(const v of variants){
 const g=data.get([c,v,'generic'].join('|')),b=data.get([c,v,'baked'].join('|'));
 if(g?.length!==5||b?.length!==5)throw Error('incomplete '+c+' '+v);
 for(const a of [g,b])if(new Set(a.map(x=>x.pts+':'+x.digest)).size!==1)throw Error('unstable '+c+' '+v);
 const strict=v.replace(/-fast$/,'-strict');
 const gs=med(data.get([c,strict,'generic'].join('|'))),bs=med(data.get([c,strict,'baked'].join('|')));
 const x=med(g),y=med(b);
 console.log('| '+[c,v,x.toFixed(4),y.toFixed(4),(gs/x).toFixed(3)+'x',(bs/y).toFixed(3)+'x',(x/y).toFixed(3)+'x'].join(' | ')+' |');
}
