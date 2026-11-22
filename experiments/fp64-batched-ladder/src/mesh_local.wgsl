// Four systems per 64-invocation workgroup: 16 cooperative workers per
// system. Store factors and triangular RHS in LDS, not global VRAM.
// Every column update owns a distinct destination. Compute multipliers
// separately before parallel updates; only replace the original column
// after every dependent read is complete.
alias Z=vec2<f64>;
const SYSTEMS=4u;
const WORKERS=16u;
const ROWS=ROW_COUNTu;
const ENTRIES=ENTRY_COUNTu;

@group(0) @binding(0) var<storage,read> original:array<f64>;
@group(0) @binding(1) var<storage,read> rhs:array<f64>;
@group(0) @binding(2) var<storage,read_write> solution:array<f64>;
@group(0) @binding(3) var<storage,read> plan:array<vec4<u32>>;
var<workgroup> factors:array<Z,MATRIX_CELLS>;
var<workgroup> values:array<Z,RHS_CELLS>;
var<workgroup> multipliers:array<Z,MULTIPLIER_CELLS>;

fn mul(a:Z,b:Z)->Z{
    return Z(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}
fn inverse(a:Z)->Z{
    let n=a.x*a.x+a.y*a.y;
    return Z(a.x/n,-a.y/n);
}
fn index(row:u32,sys:u32)->u32 { return row*SYSTEMS+sys; }

@compute @workgroup_size(64)
fn solve(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
    let sys=lane%SYSTEMS;
    let worker=lane/SYSTEMS;
    // Input remains packed in 64-system tiles, as in the other variants.
    let tile=group.x/WORKERS;
    let input_lane=(group.x%WORKERS)*SYSTEMS+sys;
    let mb=tile*ENTRIES*128u+input_lane;
    let xb=tile*ROWS*128u+input_lane;
    for(var i=worker;i<ENTRIES;i+=WORKERS){
        let offset=mb+i*128u;
        factors[index(i,sys)]=Z(original[offset],original[offset+64u]);
    }
    for(var r=worker;r<ROWS;r+=WORKERS){
        let offset=xb+r*128u;
        values[index(r,sys)]=Z(rhs[offset],rhs[offset+64u]);
    }
    workgroupBarrier();

    for(var k=0u;k<ROWS;k++){
        let col=plan[1u+k];
        let pivot=index(col.x,sys);
        if worker==0u { factors[pivot]=inverse(factors[pivot]); }
        workgroupBarrier();
        let inv=factors[pivot];
        let count=col.z-col.y;
        for(var j=worker;j<count;j+=WORKERS){
            let edge=plan[col.y+j];
            multipliers[index(j,sys)]=mul(factors[index(edge.y,sys)],inv);
        }
        workgroupBarrier();
        // Update records are contiguous before the neighbor records.
        // update.z identifies the neighbor whose multiplier is needed.
        for(var u=col.w+worker;u<col.y;u+=WORKERS){
            let update=plan[u];
            let dest=index(update.x,sys);
            factors[dest]-=mul(multipliers[index(update.z,sys)],factors[index(update.y,sys)]);
        }
        workgroupBarrier();
        for(var j=worker;j<count;j+=WORKERS){
            let edge=plan[col.y+j];
            factors[index(edge.y,sys)]=multipliers[index(j,sys)];
        }
        workgroupBarrier();
    }

    // L y = b. Leave the unscaled pivot RHS in place until backwards solve,
    // so updates can read it without a concurrent write.
    for(var k=0u;k<ROWS;k++){
        let y=values[index(k,sys)];
        let col=plan[1u+k];
        for(var j=col.y+worker;j<col.z;j+=WORKERS){
            let edge=plan[j];
            let dest=index(edge.x,sys);
            values[dest]-=mul(factors[index(edge.y,sys)],y);
        }
        workgroupBarrier();
    }
    // L^T x = D^-1 y; one worker per system, all workers synchronize at
    // the row boundary. No global storage accesses in factor or solve.
    for(var k=ROWS;k>0u;k--){
        let r=k-1u;
        if worker==0u {
            let col=plan[1u+r];
            var x=mul(factors[index(col.x,sys)],values[index(r,sys)]);
            for(var j=col.y;j<col.z;j++){
                let edge=plan[j];
                x-=mul(factors[index(edge.y,sys)],values[index(edge.x,sys)]);
            }
            values[index(r,sys)]=x;
        }
        workgroupBarrier();
    }
    for(var r=worker;r<ROWS;r+=WORKERS){
        let v=values[index(r,sys)];
        let offset=xb+r*128u;
        solution[offset]=v.x;
        solution[offset+64u]=v.y;
    }
}
